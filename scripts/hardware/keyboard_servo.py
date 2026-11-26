import math
import os
import select
import sys
import termios
import time
import tty

from smbus import SMBus
from luma.core.interface.serial import i2c
from luma.core.render import canvas
from luma.oled.device import ssd1306

PCA_BUS = 1
PCA_ADDRESS = 0x40
PAN_CHANNEL = 3
TILT_CHANNEL = 2

OLED_BUS = 3
OLED_ADDRESS = 0x3C
PWM_HZ = 50

# Electrical safeguards, not mechanical limits
MIN_PULSE_US = 1000
MAX_PULSE_US = 2000
ARM_PULSE_US = 1500

# Adjustable speed and movement steps
SPEED_CHOICES = (3, 5, 10, 20, 30)
STEP_CHOICES = (1, 2, 5, 10)

CHANNELS = {
    "pan": PAN_CHANNEL,
    "tilt": TILT_CHANNEL,
}


def configure_pca9685(bus):
    old = bus.read_byte_data(PCA_ADDRESS, 0x00)

    bus.write_byte_data(
        PCA_ADDRESS, 0x00, (old & 0x7F) | 0x10
    )

    prescale = round(
        25_000_000 / (4096 * PWM_HZ)
    ) - 1

    bus.write_byte_data(PCA_ADDRESS, 0xFE, prescale)

    wake = (old & 0x6F) | 0x20
    bus.write_byte_data(PCA_ADDRESS, 0x00, wake)

    time.sleep(0.01)

    bus.write_byte_data(
        PCA_ADDRESS, 0x00, wake | 0x80
    )


def angle_to_pulse(angle):
    return MIN_PULSE_US + (
        MAX_PULSE_US - MIN_PULSE_US
    ) * angle / 180.0


def pulse_ticks(pulse_us):
    return round(
        pulse_us * 4096 * PWM_HZ / 1_000_000
    )


def set_pwm(bus, channel, pulse_us):
    ticks = pulse_ticks(pulse_us)
    register = 0x06 + 4 * channel

    bus.write_i2c_block_data(
        PCA_ADDRESS,
        register,
        [0, 0, ticks & 0xFF, (ticks >> 8) & 0x0F]
    )

    return ticks


def disable_pwm(bus, channel):
    register = 0x06 + 4 * channel

    bus.write_i2c_block_data(
        PCA_ADDRESS,
        register,
        [0, 0, 0, 0x10]
    )


def read_key(fd):
    raw = os.read(fd, 1)

    if raw == b"\x1b":
        sequence = b""

        for _ in range(2):
            if select.select([fd], [], [], 0.08)[0]:
                sequence += os.read(fd, 1)

        return {
            b"[A": "w",
            b"[B": "s",
            b"[C": "d",
            b"[D": "a",
        }.get(sequence, "")

    return raw.decode(
        "utf-8", errors="ignore"
    ).lower()


def oled_screen(oled, current, speed, step):
    with canvas(oled) as draw:
        draw.text(
            (17, 0), "WHISPRYN", fill="white"
        )

        draw.line(
            (0, 14, 127, 14), fill="white"
        )

        for axis, y in (
            ("pan", 18),
            ("tilt", 30)
        ):
            angle = current[axis]

            value = (
                "--" if angle is None
                else f"{angle:.1f}"
            )

            draw.text(
                (2, y),
                f"{axis.upper()}: {value} deg*",
                fill="white"
            )

        draw.text(
            (2, 43),
            f"Speed: {speed} deg/s",
            fill="white"
        )

        draw.text(
            (2, 54),
            f"Step: {step} deg",
            fill="white"
        )


def main():
    if not sys.stdin.isatty():
        raise SystemExit(
            "Run from an interactive terminal."
        )

    fd = sys.stdin.fileno()
    original_terminal = termios.tcgetattr(fd)

    bus = None
    oled = None
    terminal_raw = False

    current = {
        "pan": None,
        "tilt": None
    }

    target = {
        "pan": None,
        "tilt": None
    }

    last_ticks = {
        "pan": None,
        "tilt": None
    }

    speed_idx = 1
    step_idx = 1

    def status_line():
        def label(axis):
            if current[axis] is None:
                return "OFF"

            return (
                f"{current[axis]:.1f}"
                f"->{target[axis]:.1f}"
            )

        print(
            f"\rPAN {label('pan')} | "
            f"TILT {label('tilt')} | "
            f"Speed {SPEED_CHOICES[speed_idx]} deg/s | "
            f"Step {STEP_CHOICES[step_idx]} deg   ",
            end="",
            flush=True
        )

    def enter_exact_target(axis):
        nonlocal terminal_raw

        if current[axis] is None:
            print(
                f"\nArm {axis.upper()} first."
            )
            return

        target[axis] = current[axis]

        termios.tcsetattr(
            fd,
            termios.TCSADRAIN,
            original_terminal
        )

        terminal_raw = False

        try:
            value = input(
                f"\n{axis.upper()} target "
                "(nominal 0-180 degrees): "
            ).strip()

            try:
                requested = float(value)
            except ValueError:
                print("Not a number.")
                return

            if (
                not math.isfinite(requested)
                or not 0 <= requested <= 180
            ):
                print(
                    "Outside electrical reference range."
                )
                return

            if abs(
                requested - current[axis]
            ) > 20:
                confirmation = input(
                    "Large move. Type YES to continue: "
                )

                if confirmation != "YES":
                    print("Cancelled.")
                    return

            target[axis] = requested

            print(
                f"{axis.upper()} target: "
                f"{requested:.1f} nominal degrees"
            )

        except EOFError:
            print("\nCommand cancelled.")

        finally:
            tty.setcbreak(fd)
            terminal_raw = True

    try:
        bus = SMBus(PCA_BUS)

        for channel in CHANNELS.values():
            disable_pwm(bus, channel)

        configure_pca9685(bus)

        oled = ssd1306(
            i2c(
                port=OLED_BUS,
                address=OLED_ADDRESS
            ),
            width=128,
            height=64
        )

        print("\nWHISPRYN MANUAL SERVO CONTROL")
        print("--------------------------------")
        print("W/S : TILT up/down")
        print("A/D : PAN left/right")
        print("Z/X : Speed slower/faster")
        print("N/M : Degree step smaller/larger")
        print("P/T : Enter exact target angle")
        print("1/2 : Arm PAN/TILT")
        print("H   : Halt and hold")
        print("SPACE : PWM off")
        print("Q   : Quit")
        print("--------------------------------")
        print("No physical centre assumed.")
        print("First arm command uses 1500 us.")
        print("It may cause sudden movement.")
        print("Clear mechanism before arming.\n")

        last_frame = 0.0
        last_time = time.monotonic()
        arm_pending = None

        tty.setcbreak(fd)
        terminal_raw = True

        while True:
            if select.select(
                [fd], [], [], 0.02
            )[0]:
                key = read_key(fd)

                if key == "q":
                    break

                elif key == " ":
                    for axis, channel in CHANNELS.items():
                        disable_pwm(bus, channel)

                        current[axis] = None
                        target[axis] = None
                        last_ticks[axis] = None

                    arm_pending = None
                    print("\nPWM disabled.")

                elif arm_pending is not None:
                    axis = arm_pending
                    arm_pending = None

                    if key == "y":
                        tick = set_pwm(
                            bus,
                            CHANNELS[axis],
                            ARM_PULSE_US
                        )

                        # Electrical reference only.
                        # NOT the physical centre.
                        current[axis] = 90.0
                        target[axis] = 90.0
                        last_ticks[axis] = tick

                        print(
                            f"\n{axis.upper()} armed "
                            f"at {ARM_PULSE_US} us."
                        )

                    else:
                        print("\nArming cancelled.")

                elif key in ("1", "2"):
                    axis = (
                        "pan" if key == "1"
                        else "tilt"
                    )

                    if current[axis] is None:
                        arm_pending = axis

                        print(
                            f"\nArm {axis.upper()} at "
                            "1500 us? Press Y."
                        )

                elif key == "h":
                    for axis in CHANNELS:
                        if current[axis] is not None:
                            target[axis] = current[axis]

                    print(
                        "\nMovement halted; holding."
                    )

                elif key in ("a", "d", "w", "s"):
                    axis = (
                        "pan"
                        if key in ("a", "d")
                        else "tilt"
                    )

                    if current[axis] is None:
                        print(
                            f"\nArm {axis.upper()} first."
                        )

                    else:
                        direction = (
                            1 if key in ("d", "w")
                            else -1
                        )

                        target[axis] = max(
                            0.0,
                            min(
                                180.0,
                                current[axis]
                                + direction
                                * STEP_CHOICES[step_idx]
                            )
                        )

                elif key == "z":
                    speed_idx = max(
                        0, speed_idx - 1
                    )

                elif key == "x":
                    speed_idx = min(
                        len(SPEED_CHOICES) - 1,
                        speed_idx + 1
                    )

                elif key == "n":
                    step_idx = max(
                        0, step_idx - 1
                    )

                elif key == "m":
                    step_idx = min(
                        len(STEP_CHOICES) - 1,
                        step_idx + 1
                    )

                elif key in ("p", "t"):
                    enter_exact_target(
                        "pan" if key == "p"
                        else "tilt"
                    )

                    last_time = time.monotonic()

                status_line()

            now = time.monotonic()

            dt = min(
                max(0.0, now - last_time),
                0.1
            )

            last_time = now

            # Move gradually at selected speed.
            for axis, channel in CHANNELS.items():
                if current[axis] is None:
                    continue

                distance = (
                    target[axis] - current[axis]
                )

                if abs(distance) < 0.0001:
                    continue

                step = min(
                    abs(distance),
                    SPEED_CHOICES[speed_idx] * dt
                )

                current[axis] += math.copysign(
                    step,
                    distance
                )

                pulse = angle_to_pulse(
                    current[axis]
                )

                ticks = pulse_ticks(pulse)

                if ticks != last_ticks[axis]:
                    last_ticks[axis] = set_pwm(
                        bus,
                        channel,
                        pulse
                    )

            if now - last_frame >= 0.2:
                oled_screen(
                    oled,
                    current,
                    SPEED_CHOICES[speed_idx],
                    STEP_CHOICES[step_idx]
                )

                last_frame = now

    except KeyboardInterrupt:
        print("\nInterrupted.")

    finally:
        if terminal_raw:
            termios.tcsetattr(
                fd,
                termios.TCSADRAIN,
                original_terminal
            )

        if bus is not None:
            for channel in CHANNELS.values():
                try:
                    disable_pwm(bus, channel)
                except OSError:
                    pass

            bus.close()

        if oled is not None:
            oled.clear()

        print(
            "\nWHISPRYN stopped. "
            "PWM disabled."
        )


if __name__ == "__main__":
    main()
