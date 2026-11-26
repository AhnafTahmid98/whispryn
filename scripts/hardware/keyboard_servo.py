
"""WHISPRYN keyboard calibration: PCA9685 + OLED.

Nominal degrees are reference values, not measured angles.
The user determines physical centres and safe travel limits.
"""

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


PCA_BUS, PCA_ADDR = 1, 0x40
OLED_BUS, OLED_ADDR = 3, 0x3C

PAN_CHANNEL = 3
TILT_CHANNEL = 0
CHANNELS = {"pan": PAN_CHANNEL, "tilt": TILT_CHANNEL}

PWM_HZ = 50

# Temporary electrical command safeguards.
# These are NOT calibrated mechanical limits.
MIN_US = 1000.0
MAX_US = 2000.0

SPEEDS = (3, 5, 10, 20, 30, 40, 60, 80)
STEPS = (1, 2, 5, 10)


def us_to_deg(pulse):
    return (pulse - MIN_US) * 180 / (MAX_US - MIN_US)


def deg_to_us(degrees):
    return MIN_US + degrees * (MAX_US - MIN_US) / 180


def set_output(bus, channel, pulse_us):
    ticks = round(pulse_us * 4096 * PWM_HZ / 1_000_000)
    reg = 0x06 + 4 * channel

    bus.write_i2c_block_data(
        PCA_ADDR,
        reg,
        [0, 0, ticks & 255, (ticks >> 8) & 15],
    )

    return ticks


def off(bus, channel):
    bus.write_i2c_block_data(
        PCA_ADDR,
        0x06 + 4 * channel,
        [0, 0, 0, 0x10],
    )


def setup(bus):
    old = bus.read_byte_data(PCA_ADDR, 0)

    bus.write_byte_data(
        PCA_ADDR, 0, (old & 0x7F) | 0x10
    )

    prescale = round(
        25_000_000 / (4096 * PWM_HZ)
    ) - 1

    bus.write_byte_data(PCA_ADDR, 0xFE, prescale)

    wake = (old & 0x6F) | 0x20
    bus.write_byte_data(PCA_ADDR, 0, wake)

    time.sleep(0.01)
    bus.write_byte_data(PCA_ADDR, 0, wake | 0x80)


def read_key(fd):
    key = os.read(fd, 1)

    if key == b"\x1b":
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

    return key.decode(errors="ignore").lower()


def main():
    if not sys.stdin.isatty():
        raise SystemExit("Run in a terminal or SSH session.")

    fd = sys.stdin.fileno()
    original = termios.tcgetattr(fd)

    bus = None
    oled = None
    raw = False

    # Commanded pulse widths, not physical angles.
    current = {"pan": None, "tilt": None}
    target = {"pan": None, "tilt": None}
    last_ticks = {"pan": None, "tilt": None}

    speed_i = 1
    step_i = 1

    def line_input(prompt):
        nonlocal raw

        termios.tcsetattr(
            fd, termios.TCSADRAIN, original
        )
        raw = False

        try:
            return input(prompt).strip()
        finally:
            tty.setcbreak(fd)
            raw = True

    def draw():
        with canvas(oled) as d:
            d.text((25, 0), "WHISPRYN", fill="white")
            d.line((0, 13, 127, 13), fill="white")

            for axis, y in (("pan", 17), ("tilt", 30)):
                pulse = current[axis]

                if pulse is None:
                    value = "OFF"
                else:
                    value = (
                        f"{us_to_deg(pulse):.0f}* "
                        f"{pulse:.0f}us"
                    )

                d.text(
                    (0, y),
                    f"{axis.upper()}: {value}",
                    fill="white",
                )

            d.text(
                (0, 44),
                f"Speed {SPEEDS[speed_i]} d/s",
                fill="white",
            )

            d.text(
                (0, 55),
                f"Step {STEPS[step_i]} deg",
                fill="white",
            )

    def status():
        values = []

        for axis in CHANNELS:
            pulse = current[axis]

            if pulse is None:
                values.append(f"{axis.upper()} OFF")
            else:
                values.append(
                    f"{axis.upper()} "
                    f"{us_to_deg(pulse):.1f}* "
                    f"({pulse:.0f}us)"
                )

        print(
            "\r" + " | ".join(values)
            + f" | Speed {SPEEDS[speed_i]} d/s"
            + f" | Step {STEPS[step_i]} d    ",
            end="",
            flush=True,
        )

    def arm(axis):
        if current[axis] is not None:
            print(f"\n{axis.upper()} already armed.")
            return

        print(
            f"\nFirst command may move "
            f"{axis.upper()} suddenly."
        )

        value = line_input(
            f"Initial {axis.upper()} pulse "
            f"({MIN_US:.0f}-{MAX_US:.0f} us, "
            "blank cancels): "
        )

        if not value:
            return

        try:
            pulse = float(value)
        except ValueError:
            print("Invalid pulse.")
            return

        if (
            not math.isfinite(pulse)
            or not MIN_US <= pulse <= MAX_US
        ):
            print("Outside electrical safeguards.")
            return

        confirmation = line_input(
            f"Type ARM to activate "
            f"{axis.upper()} at {pulse:.0f} us: "
        )

        if confirmation != "ARM":
            print("Cancelled.")
            return

        last_ticks[axis] = set_output(
            bus, CHANNELS[axis], pulse
        )

        current[axis] = pulse
        target[axis] = pulse

        print(
            f"\n{axis.upper()} armed at "
            f"{pulse:.0f} us."
        )

    def exact_target(axis):
        if current[axis] is None:
            print(f"\nArm {axis.upper()} first.")
            return

        target[axis] = current[axis]

        value = line_input(
            f"{axis.upper()} nominal target degrees "
            "(0-180, blank cancels): "
        )

        if not value:
            return

        try:
            degrees = float(value)
        except ValueError:
            print("Not a number.")
            return

        if (
            not math.isfinite(degrees)
            or not 0 <= degrees <= 180
        ):
            print("Outside current pulse reference.")
            return

        if abs(degrees - us_to_deg(current[axis])) > 20:
            if line_input(
                "Large command. Type YES: "
            ) != "YES":
                print("Cancelled.")
                return

        target[axis] = deg_to_us(degrees)

    try:
        bus = SMBus(PCA_BUS)

        for channel in CHANNELS.values():
            off(bus, channel)

        setup(bus)

        oled = ssd1306(
            i2c(port=OLED_BUS, address=OLED_ADDR),
            width=128,
            height=64,
        )

        draw()

        print("\nWHISPRYN KEYBOARD CALIBRATION")
        print("--------------------------------")
        print("1/2  : Arm PAN / TILT")
        print("A/D  : PAN left/right")
        print("W/S  : TILT up/down")
        print("Z/X  : Decrease/increase speed")
        print("N/M  : Decrease/increase degree step")
        print("P/T  : Exact nominal degree target")
        print("H    : Halt and hold")
        print("SPACE: Disable PWM")
        print("Q    : Quit")
        print("--------------------------------")
        print("No mechanical centre assumed.")

        tty.setcbreak(fd)
        raw = True

        last = time.monotonic()
        last_oled = 0

        while True:
            if select.select([fd], [], [], 0.02)[0]:
                key = read_key(fd)

                if key == "q":
                    break

                elif key == " ":
                    for axis, channel in CHANNELS.items():
                        off(bus, channel)

                        current[axis] = None
                        target[axis] = None
                        last_ticks[axis] = None

                    print("\nPWM disabled.")

                elif key in ("1", "2"):
                    arm("pan" if key == "1" else "tilt")
                    last = time.monotonic()

                elif key in ("p", "t"):
                    exact_target(
                        "pan" if key == "p" else "tilt"
                    )
                    last = time.monotonic()

                elif key == "h":
                    for axis in CHANNELS:
                        if current[axis] is not None:
                            target[axis] = current[axis]

                elif key in ("a", "d", "w", "s"):
                    axis = (
                        "pan" if key in ("a", "d")
                        else "tilt"
                    )

                    if target[axis] is None:
                        print(
                            f"\nArm {axis.upper()} first."
                        )
                    else:
                        direction = (
                            1 if key in ("d", "w") else -1
                        )

                        # Accumulate from TARGET, not current.
                        delta_us = (
                            deg_to_us(STEPS[step_i])
                            - MIN_US
                        )

                        target[axis] = max(
                            MIN_US,
                            min(
                                MAX_US,
                                target[axis]
                                + direction * delta_us,
                            ),
                        )

                elif key == "z":
                    speed_i = max(0, speed_i - 1)

                elif key == "x":
                    speed_i = min(
                        len(SPEEDS) - 1,
                        speed_i + 1,
                    )

                elif key == "n":
                    step_i = max(0, step_i - 1)

                elif key == "m":
                    step_i = min(
                        len(STEPS) - 1,
                        step_i + 1,
                    )

                status()

            now = time.monotonic()
            dt = min(max(now - last, 0), 0.1)
            last = now

            speed_us_per_second = (
                SPEEDS[speed_i]
                * (MAX_US - MIN_US) / 180
            )

            for axis, channel in CHANNELS.items():
                if current[axis] is None:
                    continue

                remaining = (
                    target[axis] - current[axis]
                )

                if abs(remaining) < 0.000001:
                    continue

                distance = min(
                    abs(remaining),
                    speed_us_per_second * dt,
                )

                current[axis] += math.copysign(
                    distance, remaining
                )

                ticks = round(
                    current[axis]
                    * 4096 * PWM_HZ / 1_000_000
                )

                if ticks != last_ticks[axis]:
                    last_ticks[axis] = set_output(
                        bus,
                        channel,
                        current[axis],
                    )

            if now - last_oled >= 0.2:
                draw()
                last_oled = now

    except KeyboardInterrupt:
        print("\nInterrupted.")

    finally:
        if raw:
            termios.tcsetattr(
                fd,
                termios.TCSADRAIN,
                original,
            )

        if bus is not None:
            for channel in CHANNELS.values():
                try:
                    off(bus, channel)
                except OSError:
                    pass

            bus.close()

        if oled is not None:
            oled.clear()

        print("\nStopped. PWM disabled.")


if __name__ == "__main__":
    main()
