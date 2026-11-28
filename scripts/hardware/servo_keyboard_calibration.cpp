
#include <linux/i2c-dev.h>
#include <sys/ioctl.h>
#include <sys/select.h>
#include <termios.h>
#include <fcntl.h>
#include <unistd.h>
#include <csignal>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <array>
#include <cctype>

namespace fs = std::filesystem;

// WHISPRYN hardware configuration
constexpr int PCA_ADDR = 0x40;
constexpr int PAN_CH = 0;
constexpr int TILT_CH = 2;
constexpr int FIXED_TILT_US = 2583;
constexpr int PWM_HZ = 50;

// PWM signal test range. The mounted mechanism has its own travel limits.
constexpr double MIN_US = 500;
constexpr double MAX_US = 3500;
constexpr double CONFIRM_CHANGE_US = 100;

const std::array<double, 11> rates_us_per_second{
    3, 5, 10, 20, 30, 40, 60, 80, 100, 150, 200
};

const std::array<double, 4> steps_us{
    5, 10, 25, 50
};

volatile std::sig_atomic_t running = 1;

void stop_signal(int) {
    running = 0;
}

// Linux I2C interface
class I2C {
    int fd_ = -1;

public:
    I2C(const char* device, int address) {
        fd_ = ::open(device, O_RDWR);

        if (fd_ < 0 ||
            ::ioctl(fd_, I2C_SLAVE, address) < 0) {
            throw std::runtime_error(
                std::string("Cannot open/configure ") + device
            );
        }
    }

    void write_bytes(
        const unsigned char* bytes,
        size_t n
    ) {
        if (::write(fd_, bytes, n) !=
            static_cast<ssize_t>(n)) {
            throw std::runtime_error("I2C write failed");
        }
    }

    void reg(unsigned char reg, unsigned char value) {
        const unsigned char b[] = {reg, value};
        write_bytes(b, 2);
    }

    unsigned char read_reg(unsigned char reg) {
        write_bytes(&reg, 1);

        unsigned char value = 0;

        if (::read(fd_, &value, 1) != 1) {
            throw std::runtime_error("I2C read failed");
        }

        return value;
    }

    ~I2C() {
        if (fd_ >= 0) {
            ::close(fd_);
        }
    }

    I2C(const I2C&) = delete;
    I2C& operator=(const I2C&) = delete;
};

// PCA9685 servo controller
class PCA9685 {
    I2C dev_{"/dev/i2c-1", PCA_ADDR};

public:
    PCA9685() {
        const auto old = dev_.read_reg(0x00);

        // Enter sleep mode
        dev_.reg(
            0x00,
            (old & 0x7f) | 0x10
        );

        // Configure 50 Hz PWM
        dev_.reg(
            0xfe,
            static_cast<unsigned char>(
                std::lround(
                    25000000.0 / (4096 * PWM_HZ)
                ) - 1
            )
        );

        // Enable auto-increment
        const auto wake =
            static_cast<unsigned char>(
                (old & 0x6f) | 0x20
            );

        dev_.reg(0x00, wake);

        std::this_thread::sleep_for(
            std::chrono::milliseconds(10)
        );

        dev_.reg(0x00, wake | 0x80);

        // No automatic positioning
        off(PAN_CH);
        off(TILT_CH);
    }

    void pulse(int ch, double us) {
        if (!std::isfinite(us) ||
            us < MIN_US ||
            us > MAX_US) {
            throw std::runtime_error(
                "Pulse outside electrical safeguards"
            );
        }

        const int ticks =
            static_cast<int>(
                std::lround(
                    us * 4096 * PWM_HZ / 1e6
                )
            );

        const unsigned char b[] = {
            static_cast<unsigned char>(
                0x06 + 4 * ch
            ),
            0,
            0,
            static_cast<unsigned char>(ticks & 255),
            static_cast<unsigned char>(
                (ticks >> 8) & 15
            )
        };

        dev_.write_bytes(b, 5);
    }

    void off(int ch) {
        const unsigned char b[] = {
            static_cast<unsigned char>(
                0x06 + 4 * ch
            ),
            0,
            0,
            0,
            0x10
        };

        dev_.write_bytes(b, 5);
    }

    ~PCA9685() {
        try {
            off(PAN_CH);
            off(TILT_CH);
        } catch (...) {}
    }
};

// Interactive terminal handling
class Terminal {
    termios original_{};
    bool raw_ = false;

public:
    Terminal() {
        if (!isatty(STDIN_FILENO) ||
            tcgetattr(
                STDIN_FILENO,
                &original_
            ) != 0) {
            throw std::runtime_error(
                "Interactive terminal or SSH required"
            );
        }
    }

    void raw() {
        termios mode = original_;

        mode.c_lflag &= ~(ICANON | ECHO);
        mode.c_cc[VMIN] = 1;
        mode.c_cc[VTIME] = 0;

        if (tcsetattr(
            STDIN_FILENO,
            TCSANOW,
            &mode
        ) != 0) {
            throw std::runtime_error(
                "Cannot configure terminal"
            );
        }

        raw_ = true;
    }

    void normal() {
        if (raw_) {
            tcsetattr(
                STDIN_FILENO,
                TCSANOW,
                &original_
            );
        }

        raw_ = false;
    }

    std::string ask(const std::string& prompt) {
        normal();

        std::cout << "\n" << prompt << std::flush;

        std::string answer;
        std::getline(std::cin, answer);

        raw();
        return answer;
    }

    ~Terminal() {
        normal();
    }
};

struct Marks {
    std::optional<double> center, low, high;

    double minimum() const { return low.value_or(MIN_US); }
    double maximum() const { return high.value_or(MAX_US); }

    bool valid() const {
        for (auto pulse : {center, low, high}) {
            if (pulse && !(*pulse >= MIN_US && *pulse <= MAX_US)) return false;
        }
        return (!low || (center && *low < *center)) &&
               (!high || (center && *center < *high));
    }
};

struct Axis {
    const char* name;
    int channel;

    std::optional<double> current;
    std::optional<double> target;
    std::optional<double> saved;

    int last_ticks = -1;
    Marks marks{};
};

std::string show_axis(const Axis& axis) {
    if (!axis.current) return "OFF";
    std::string pulse = std::to_string(std::lround(*axis.current)) + "us";
    if (axis.channel == TILT_CH) return pulse + " (fixed)";
    if (!axis.marks.center) return pulse + " (center unset)";
    const auto offset = std::lround(*axis.current - *axis.marks.center);
    return pulse + " (" + (offset > 0 ? "+" : "") + std::to_string(offset) + "us from center)";
}

// Store calibration separately from the last position commanded before PWM was disabled.
void read_marks(const fs::path& file, Axis& pan, Axis& tilt) {
    if (!fs::exists(file)) return;
    std::ifstream in(file);
    if (!in) throw std::runtime_error("Cannot read " + file.string());
    std::string axis_name, key;
    double pulse;
    while (in >> axis_name) {
        if (!(in >> key >> pulse)) throw std::runtime_error("Incomplete servo mark");
        Axis* axis = axis_name == pan.name ? &pan : axis_name == tilt.name ? &tilt : nullptr;
        if (!axis) throw std::runtime_error("Unknown axis in servo marks");
        auto& marks = axis->marks;
        if (key == "center" && !marks.center) marks.center = pulse;
        else if (key == "low" && !marks.low) marks.low = pulse;
        else if (key == "high" && !marks.high) marks.high = pulse;
        else throw std::runtime_error("Unknown or duplicate servo mark");
    }
    if (!in.eof() || !pan.marks.valid() || !tilt.marks.valid())
        throw std::runtime_error("Invalid servo marks in " + file.string());
}

void save_marks(const fs::path& file, const Axis& pan, const Axis& tilt) {
    fs::create_directories(file.parent_path());
    const fs::path tmp = file.string() + ".tmp";
    std::ofstream out(tmp);
    out << std::setprecision(17);
    for (const Axis* axis : {&pan, &tilt}) {
        const auto& m = axis->marks;
        if (m.center) out << axis->name << " center " << *m.center << '\n';
        if (m.low) out << axis->name << " low " << *m.low << '\n';
        if (m.high) out << axis->name << " high " << *m.high << '\n';
    }
    out.close();
    if (!out) throw std::runtime_error("Cannot write " + tmp.string());
    fs::rename(tmp, file);
}

int ticks_for(double us) {
    return static_cast<int>(
        std::lround(
            us * 4096 * PWM_HZ / 1e6
        )
    );
}

void enable_axis(PCA9685& board, Axis& axis, double pulse) {
    board.pulse(axis.channel, pulse);
    axis.current = pulse;
    axis.target = pulse;
    axis.last_ticks = ticks_for(pulse);
}

// Load last commanded positions
void read_saved(
    const fs::path& file,
    Axis& pan,
    Axis& tilt
) {
    std::ifstream f(file);

    std::string name;
    double pulse;

    while (f >> name >> pulse) {
        if (!std::isfinite(pulse) ||
            pulse < MIN_US ||
            pulse > MAX_US) {
            continue;
        }

        if (name == "pan") {
            pan.saved = pulse;
        }

        if (name == "tilt") {
            tilt.saved = pulse;
        }
    }
}

// Save last commanded positions
void persist(
    const fs::path& file,
    Axis& pan,
    Axis& tilt
) {
    fs::create_directories(
        file.parent_path()
    );

    const auto tmp = fs::path(
        file.string() + ".tmp"
    );

    std::ofstream out(
        tmp,
        std::ios::trunc
    );

    if (!out) {
        throw std::runtime_error(
            "Cannot write calibration state"
        );
    }

    if (pan.current) {
        out << "pan " << *pan.current << "\n";
    } else if (pan.saved) {
        out << "pan " << *pan.saved << "\n";
    }

    if (tilt.current) {
        out << "tilt " << *tilt.current << "\n";
    } else if (tilt.saved) {
        out << "tilt " << *tilt.saved << "\n";
    }

    out.close();

    if (!out) {
        throw std::runtime_error(
            "Cannot finish calibration state write"
        );
    }

    fs::rename(tmp, file);
}

// Keyboard reading
char read_key() {
    unsigned char key = 0;

    if (::read(
        STDIN_FILENO,
        &key,
        1
    ) != 1) {
        return '\0';
    }

    if (key != 27) {
        return static_cast<char>(
            std::tolower(key)
        );
    }

    // Arrow-key escape sequence
    char seq[2] = {};

    for (int i = 0; i < 2; ++i) {
        fd_set rfds;

        FD_ZERO(&rfds);
        FD_SET(
            STDIN_FILENO,
            &rfds
        );

        timeval tv{0, 80000};

        if (select(
            STDIN_FILENO + 1,
            &rfds,
            nullptr,
            nullptr,
            &tv
        ) <= 0) {
            break;
        }

        if (::read(
            STDIN_FILENO,
            &seq[i],
            1
        ) != 1) {
            break;
        }
    }

    if (seq[0] != '[') {
        return '\0';
    }

    switch (seq[1]) {
        case 'A': return 'w';
        case 'B': return 's';
        case 'C': return 'd';
        case 'D': return 'a';
    }

    return '\0';
}

int main() {
    std::signal(
        SIGINT,
        stop_signal
    );

    std::signal(
        SIGTERM,
        stop_signal
    );

    const fs::path state =
        fs::path("runtime") /
        "servo_last_commands.txt";
    const fs::path marks_file = state.parent_path() / "servo_marks.txt";

    Axis pan{
        "PAN",
        PAN_CH,
        {},
        {},
        {},
        -1
    };

    Axis tilt{
        "TILT",
        TILT_CH,
        {},
        {},
        {},
        -1
    };

    try {
        Terminal keyboard;
        read_marks(marks_file, pan, tilt);
        PCA9685 board;

        read_saved(
            state,
            pan,
            tilt
        );

        int speed_idx = 1;
        int step_idx = 0;
        Axis* selected = &pan;

        keyboard.raw();

        std::cout
            << "\nWHISPRYN C++ KEYBOARD CALIBRATION\n"
            << "1 arm PAN (ch0), A/D jog | 2 arm fixed TILT (ch2) at " << FIXED_TILT_US << " us\n"
            << "PAN: Z/X pulse speed (us/s) | N/M pulse step (us) | P pulse offset from center (us)\n"
            << "PAN: C mark center | 0 return to center\n"
            << "[ mark lower limit | ] mark upper limit | R reset selected axis marks\n"
            << "H halt/hold | SPACE PWM off | Q quit\n"
            << "No auto-start command; no mechanical centre assumed.\n"
            << "First ARM may still jump if the servo moved unpowered.\n";

        std::cout
            << "Saved reference: PAN "
            << (
                pan.saved
                ? std::to_string(*pan.saved)
                : "unknown"
            )
            << " us; TILT "
            << (
                tilt.saved
                ? std::to_string(*tilt.saved)
                : "unknown"
            )
            << " us\n";

        auto last_time =
            std::chrono::steady_clock::now();

        auto last_report = last_time;
        auto last_save = last_time;
        std::string last_status;

        while (running) {
            fd_set rfds;

            FD_ZERO(&rfds);
            FD_SET(
                STDIN_FILENO,
                &rfds
            );

            timeval tv{0, 20000};

            if (select(
                STDIN_FILENO + 1,
                &rfds,
                nullptr,
                nullptr,
                &tv
            ) > 0) {

                const char key = read_key();

                Axis* a =
                    (
                        key == '1' ||
                        key == 'a' ||
                        key == 'd' ||
                        key == 'p'
                    )
                    ? &pan
                    : (
                        key == '2' ||
                        key == 'w' ||
                        key == 's' ||
                        key == 't'
                    )
                    ? &tilt
                    : nullptr;
                if (a) selected = a;

                if (key == 'q') {
                    break;
                }

                if (key == ' ') {
                    board.off(PAN_CH);
                    board.off(TILT_CH);

                    if (pan.current) {
                        pan.saved = pan.current;
                    }

                    if (tilt.current) {
                        tilt.saved = tilt.current;
                    }

                    pan.current.reset();
                    pan.target.reset();
                    pan.last_ticks = -1;

                    tilt.current.reset();
                    tilt.target.reset();
                    tilt.last_ticks = -1;

                    persist(
                        state,
                        pan,
                        tilt
                    );

                    std::cout
                        << "\nPWM OFF: tilt may move under gravity.\n";

                } else if (key == 'c' || key == '[' || key == ']' || key == '0' || key == 'r') {
                    Axis& axis = *selected;
                    if (selected == &tilt && key != 'r') {
                        std::cout << "\nTILT is fixed at " << FIXED_TILT_US << " us.\n";
                    } else if (key == 'r') {
                        axis.target = axis.current;
                        if (keyboard.ask(std::string("Type RESET to clear ") + axis.name + " marks: ") == "RESET") {
                            axis.marks = {};
                            save_marks(marks_file, pan, tilt);
                            std::cout << "\n" << axis.name << " marks cleared.\n";
                        }
                        last_time = std::chrono::steady_clock::now();
                    } else if (!axis.current) {
                        std::cout << "\nArm " << axis.name << " first.\n";
                    } else if (key == '0') {
                        if (axis.marks.center) axis.target = axis.marks.center;
                        else std::cout << "\nJog to center and press C first.\n";
                    } else {
                        axis.target = axis.current;
                        Marks marks = axis.marks;
                        if (key == 'c') marks.center = axis.current;
                        if (key == '[') marks.low = axis.current;
                        if (key == ']') marks.high = axis.current;
                        if (marks.valid()) {
                            axis.marks = marks;
                            save_marks(marks_file, pan, tilt);
                            const char* label = key == 'c' ? "center" : key == '[' ? "lower limit" : "upper limit";
                            std::cout << "\n" << axis.name << " " << label << " saved at "
                                      << *axis.current << " us.\n";
                        } else {
                            std::cout << "\nSet center first and keep lower < center < upper.\n";
                        }
                    }
                } else if (key == 'h') {

                    if (pan.current) {
                        pan.target = pan.current;
                    }

                    if (tilt.current) {
                        tilt.target = tilt.current;
                    }

                    std::cout
                        << "\nTargets halted; holding last commands.\n";

                } else if (key == 'z') {

                    speed_idx = std::max(
                        0,
                        speed_idx - 1
                    );

                } else if (key == 'x') {

                    speed_idx = std::min(
                        static_cast<int>(rates_us_per_second.size()) - 1,
                        speed_idx + 1
                    );

                } else if (key == 'n') {

                    step_idx = std::max(
                        0,
                        step_idx - 1
                    );

                } else if (key == 'm') {

                    step_idx = std::min(
                        static_cast<int>(steps_us.size()) - 1,
                        step_idx + 1
                    );

                } else if (key == '2') {
                    if (tilt.current) {
                        std::cout << "\nTILT already holding " << FIXED_TILT_US << " us.\n";
                    } else if (FIXED_TILT_US < tilt.marks.minimum() || FIXED_TILT_US > tilt.marks.maximum()) {
                        std::cout << "\nFixed TILT pulse is outside saved limits. TILT stays off.\n";
                    } else {
                        const std::string prompt = "Type ARM to hold TILT at " +
                            std::to_string(FIXED_TILT_US) + " us (first move may jump): ";
                        if (keyboard.ask(prompt) == "ARM" && running) {
                            enable_axis(board, tilt, FIXED_TILT_US);
                            persist(state, pan, tilt);
                            std::cout << "\nTILT holding " << FIXED_TILT_US << " us.\n";
                        } else {
                            std::cout << "\nCancelled.\n";
                        }
                        last_time = std::chrono::steady_clock::now();
                    }
                } else if (key == 'w' || key == 's' || key == 't') {
                    std::cout << "\nTILT is fixed at " << FIXED_TILT_US << " us.\n";
                } else if (key == '1' && a) {

                    if (a->current) {
                        std::cout
                            << "\n"
                            << a->name
                            << " already armed.\n";
                    } else {

                        keyboard.normal();

                        std::cout
                            << "\nLast saved "
                            << a->name
                            << ": "
                            << (
                                a->saved
                                ? std::to_string(*a->saved) + " us"
                                : "unknown"
                            )
                            << "\n";

                        std::cout
                            << "Servo may have moved while off. "
                            << "Enter INITIAL pulse "
                            << "(" << a->marks.minimum() << ".." << a->marks.maximum()
                            << " us, blank cancels): "
                            << std::flush;

                        std::string input;
                        std::getline(
                            std::cin,
                            input
                        );

                        if (!input.empty()) {
                            try {
                                size_t used = 0;

                                const double pulse =
                                    std::stod(input, &used);

                                if (
                                    used != input.size() ||
                                    !std::isfinite(pulse) ||
                                    pulse < a->marks.minimum() ||
                                    pulse > a->marks.maximum()
                                ) {
                                    throw std::invalid_argument(
                                        "pulse"
                                    );
                                }

                                std::cout
                                    << "Type ARM to send this first command "
                                    << "(may jump): "
                                    << std::flush;

                                std::string confirmation;

                                std::getline(
                                    std::cin,
                                    confirmation
                                );

                                if (confirmation == "ARM") {
                                    enable_axis(board, *a, pulse);

                                    persist(
                                        state,
                                        pan,
                                        tilt
                                    );

                                    std::cout
                                        << a->name
                                        << " ARMED at "
                                        << pulse
                                        << " us\n";
                                } else {
                                    std::cout
                                        << "Cancelled.\n";
                                }

                            } catch (...) {
                                std::cout
                                    << "Invalid initial pulse.\n";
                            }
                        }

                        keyboard.raw();

                        last_time =
                            std::chrono::steady_clock::now();
                    }

                } else if (key == 'p' && a) {

                    if (!a->current) {
                        std::cout
                            << "\nArm "
                            << a->name
                            << " first.\n";
                    } else if (!a->marks.center) {
                        std::cout << "\nJog to the model's center and press C first.\n";
                    } else {
                        keyboard.normal();
                        const double center = *a->marks.center;
                        std::cout
                            << "\n"
                            << a->name
                            << " pulse offset from center in us ("
                            << a->marks.minimum() - center << ".."
                            << a->marks.maximum() - center << ", blank cancels): "
                            << std::flush;

                        std::string input;
                        std::getline(
                            std::cin,
                            input
                        );

                        if (!input.empty()) {
                            try {
                                size_t used = 0;

                                const double offset_us =
                                    std::stod(input, &used);
                                const double target = center + offset_us;

                                if (
                                    used != input.size() ||
                                    !std::isfinite(offset_us) ||
                                    target < a->marks.minimum() ||
                                    target > a->marks.maximum()
                                ) {
                                    throw std::invalid_argument(
                                        "pulse offset"
                                    );
                                }

                                if (
                                    std::abs(target - *a->current) > CONFIRM_CHANGE_US
                                ) {
                                    std::cout
                                        << "Large change. "
                                        << "Type YES to proceed: "
                                        << std::flush;

                                    std::string confirm;
                                    std::getline(
                                        std::cin,
                                        confirm
                                    );

                                    if (confirm != "YES") {
                                        throw std::invalid_argument(
                                            "cancelled"
                                        );
                                    }
                                }

                                a->target = target;

                            } catch (...) {
                                std::cout
                                    << "Target cancelled or invalid.\n";
                            }
                        }

                        keyboard.raw();

                        last_time =
                            std::chrono::steady_clock::now();
                    }

                } else if (a && (key == 'a' || key == 'd')) {

                    if (!a->target) {
                        std::cout
                            << "\nArm "
                            << a->name
                            << " first.\n";
                    } else {

                        const int sign = key == 'd' ? 1 : -1;

                        a->target = std::clamp(
                            *a->current +
                            sign * steps_us[step_idx],
                            a->marks.minimum(),
                            a->marks.maximum()
                        );
                    }
                }
            }

            const auto now =
                std::chrono::steady_clock::now();

            const double dt = std::clamp(
                std::chrono::duration<double>(
                    now - last_time
                ).count(),
                0.0,
                0.1
            );

            last_time = now;

            const double us_per_second =
                rates_us_per_second[speed_idx];

            for (auto* a : {&pan, &tilt}) {
                if (!a->current || !a->target) {
                    continue;
                }

                const double remaining =
                    *a->target - *a->current;

                if (std::abs(remaining) < 0.0001) {
                    continue;
                }

                *a->current += std::copysign(
                    std::min(
                        std::abs(remaining),
                        us_per_second * dt
                    ),
                    remaining
                );

                const int ticks =
                    ticks_for(*a->current);

                if (ticks != a->last_ticks) {
                    board.pulse(
                        a->channel,
                        *a->current
                    );

                    a->last_ticks = ticks;
                }
            }

            if (
                now - last_report >
                std::chrono::milliseconds(500)
            ) {
                std::ostringstream status;
                status
                    << "PAN " << show_axis(pan)
                    << " | TILT " << show_axis(tilt)
                    << " | selected " << selected->name
                    << " | speed "
                    << rates_us_per_second[speed_idx]
                    << "us/s | step "
                    << steps_us[step_idx]
                    << "us";

                const auto line = status.str();
                if (line != last_status) {
                    std::cout << line << '\n' << std::flush;
                    last_status = line;
                }

                last_report = now;
            }

            if (
                now - last_save >
                std::chrono::seconds(2)
            ) {
                try {
                    persist(
                        state,
                        pan,
                        tilt
                    );
                } catch (...) {}

                last_save = now;
            }
        }

        persist(
            state,
            pan,
            tilt
        );

        keyboard.normal();

        std::cout
            << "\nStopped. Servo outputs disabled.\n";

    } catch (const std::exception& e) {
        std::cerr
            << "\nERROR: "
            << e.what()
            << "\n";

        return 1;
    }

    return 0;
}
