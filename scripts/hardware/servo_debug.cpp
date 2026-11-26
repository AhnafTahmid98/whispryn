
#include <linux/i2c-dev.h>
#include <sys/ioctl.h>
#include <sys/select.h>
#include <termios.h>
#include <fcntl.h>
#include <unistd.h>
#include <signal.h>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>

constexpr int PCA_ADDR = 0x40;
constexpr int PAN_CHANNEL = 0;
constexpr int TILT_CHANNEL = 2;
constexpr int PWM_HZ = 50;

// Temporary electrical test envelope.
// These are NOT calibrated mechanical limits.
constexpr int MIN_US = 1000;
constexpr int MAX_US = 2000;

volatile sig_atomic_t running = 1;

void stop_signal(int) {
    running = 0;
}

class PCA9685 {
    int fd = -1;

    void write_data(const uint8_t* data, size_t size) {
        if (::write(fd, data, size) !=
            static_cast<ssize_t>(size)) {
            throw std::runtime_error("I2C write failed");
        }
    }

    void write_reg(uint8_t reg, uint8_t value) {
        uint8_t data[] = {reg, value};
        write_data(data, 2);
    }

    uint8_t read_reg(uint8_t reg) {
        if (::write(fd, &reg, 1) != 1)
            throw std::runtime_error("I2C select failed");

        uint8_t value = 0;

        if (::read(fd, &value, 1) != 1)
            throw std::runtime_error("I2C read failed");

        return value;
    }

public:
    PCA9685() {
        fd = ::open("/dev/i2c-1", O_RDWR);

        if (fd < 0)
            throw std::runtime_error(
                "Cannot open /dev/i2c-1"
            );

        if (::ioctl(fd, I2C_SLAVE, PCA_ADDR) < 0)
            throw std::runtime_error(
                "PCA9685 not accessible"
            );

        uint8_t old = read_reg(0x00);

        write_reg(
            0x00,
            (old & 0x7F) | 0x10
        );

        int prescale = std::lround(
            25000000.0 / (4096 * PWM_HZ)
        ) - 1;

        write_reg(0xFE, prescale);

        uint8_t wake = (old & 0x6F) | 0x20;

        write_reg(0x00, wake);
        usleep(10000);
        write_reg(0x00, wake | 0x80);

        disable(PAN_CHANNEL);
        disable(TILT_CHANNEL);
    }

    void pulse(int channel, int us) {
        if (us < MIN_US || us > MAX_US)
            throw std::runtime_error(
                "Pulse outside test envelope"
            );

        int ticks = std::lround(
            us * 4096.0 * PWM_HZ / 1000000.0
        );

        uint8_t data[] = {
            static_cast<uint8_t>(0x06 + 4 * channel),
            0,
            0,
            static_cast<uint8_t>(ticks & 0xFF),
            static_cast<uint8_t>((ticks >> 8) & 0x0F)
        };

        write_data(data, 5);
    }

    void disable(int channel) {
        uint8_t data[] = {
            static_cast<uint8_t>(0x06 + 4 * channel),
            0, 0, 0, 0x10
        };

        write_data(data, 5);
    }

    ~PCA9685() {
        if (fd >= 0) {
            try {
                disable(PAN_CHANNEL);
                disable(TILT_CHANNEL);
            } catch (...) {}

            ::close(fd);
        }
    }
};

class Keyboard {
    termios original{};
    bool active = false;

public:
    Keyboard() {
        if (!isatty(STDIN_FILENO) ||
            tcgetattr(STDIN_FILENO, &original) != 0) {
            throw std::runtime_error(
                "Interactive terminal required"
            );
        }
    }

    void enable() {
        termios mode = original;
        mode.c_lflag &= ~(ICANON | ECHO);
        mode.c_cc[VMIN] = 0;
        mode.c_cc[VTIME] = 0;

        tcsetattr(STDIN_FILENO, TCSANOW, &mode);
        active = true;
    }

    void restore() {
        if (active)
            tcsetattr(
                STDIN_FILENO,
                TCSANOW,
                &original
            );

        active = false;
    }

    ~Keyboard() {
        restore();
    }
};

int main() {
    signal(SIGINT, stop_signal);
    signal(SIGTERM, stop_signal);

    try {
        Keyboard keyboard;
        PCA9685 servo;

        int pan = -1;
        int tilt = -1;
        int step = 20;

        keyboard.enable();

        std::cout
            << "\nWHISPRYN C++ SERVO TEST\n"
            << "PAN  : Channel 0\n"
            << "TILT : Channel 2\n"
            << "-----------------------------\n"
            << "1 / 2 : Arm PAN / TILT\n"
            << "A / D : Move PAN\n"
            << "W / S : Move TILT\n"
            << "N / M : Smaller / larger step\n"
            << "SPACE : Disable both outputs\n"
            << "Q     : Quit\n"
            << "-----------------------------\n"
            << "Initial step: 20 microseconds\n"
            << "No mechanical centre assumed.\n\n";

        while (running) {
            fd_set readset;
            FD_ZERO(&readset);
            FD_SET(STDIN_FILENO, &readset);

            timeval timeout{0, 100000};

            if (select(
                STDIN_FILENO + 1,
                &readset,
                nullptr,
                nullptr,
                &timeout
            ) <= 0) {
                continue;
            }

            char key = 0;

            if (read(STDIN_FILENO, &key, 1) != 1)
                continue;

            key = static_cast<char>(
                std::tolower(
                    static_cast<unsigned char>(key)
                )
            );

            if (key == 'q')
                break;

            if (key == ' ') {
                servo.disable(PAN_CHANNEL);
                servo.disable(TILT_CHANNEL);

                pan = -1;
                tilt = -1;

                std::cout
                    << "Both outputs disabled\n";
                continue;
            }

            if (key == 'n') {
                step = std::max(5, step - 5);

                std::cout
                    << "Step: " << step << " us\n";
                continue;
            }

            if (key == 'm') {
                step = std::min(50, step + 5);

                std::cout
                    << "Step: " << step << " us\n";
                continue;
            }

            if (key == '1' || key == '2') {
                bool is_pan = (key == '1');
                int& position = is_pan ? pan : tilt;

                int channel = is_pan
                    ? PAN_CHANNEL
                    : TILT_CHANNEL;

                keyboard.restore();

                std::cout
                    << "\nInitial "
                    << (is_pan ? "PAN" : "TILT")
                    << " pulse (1000-2000 us): "
                    << std::flush;

                std::string input;

                if (!std::getline(std::cin, input))
                    break;

                try {
                    if (!input.empty()) {
                        size_t used = 0;
                        int pulse = std::stoi(
                            input, &used
                        );

                        if (used != input.size() ||
                            pulse < MIN_US ||
                            pulse > MAX_US) {
                            throw std::invalid_argument(
                                "Invalid pulse"
                            );
                        }

                        std::cout
                            << "Type ARM to activate "
                            << "(first move may jump): "
                            << std::flush;

                        std::string confirm;

                        if (!std::getline(
                            std::cin, confirm
                        )) {
                            break;
                        }

                        if (confirm == "ARM") {
                            servo.pulse(channel, pulse);
                            position = pulse;

                            std::cout
                                << "Armed at "
                                << pulse << " us\n";
                        } else {
                            std::cout
                                << "Cancelled\n";
                        }
                    }
                } catch (const std::exception&) {
                    std::cout
                        << "Invalid pulse\n";
                }

                keyboard.enable();
                continue;
            }

            int* position = nullptr;
            int channel = -1;
            int direction = 0;

            if (key == 'a' || key == 'd') {
                position = &pan;
                channel = PAN_CHANNEL;
                direction = key == 'd' ? 1 : -1;
            } else if (key == 'w' || key == 's') {
                position = &tilt;
                channel = TILT_CHANNEL;
                direction = key == 'w' ? 1 : -1;
            }

            if (!position)
                continue;

            if (*position < 0) {
                std::cout
                    << "Arm the servo first\n";
                continue;
            }

            int next = std::clamp(
                *position + direction * step,
                MIN_US,
                MAX_US
            );

            if (next == *position) {
                std::cout
                    << "Test pulse envelope reached\n";
                continue;
            }

            servo.pulse(channel, next);
            *position = next;

            std::cout
                << (channel == PAN_CHANNEL
                    ? "PAN" : "TILT")
                << " channel " << channel
                << " commanded "
                << next << " us\n";
        }

        std::cout
            << "Exiting. PWM disabled.\n";

    } catch (const std::exception& error) {
        std::cerr
            << "ERROR: " << error.what() << "\n";
        return 1;
    }

    return 0;
}
