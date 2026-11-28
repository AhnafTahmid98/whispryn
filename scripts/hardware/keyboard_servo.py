
"""WHISPRYN keyboard calibration: PCA9685 + OLED.

Nominal degrees are reference values, not measured angles.
The user determines physical centres and safe travel limits.
"""

import json
import math
import os
import select
import sys
import termios
import time
import tty
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Optional

from smbus import SMBus
from luma.core.interface.serial import i2c
from luma.core.render import canvas
from luma.oled.device import ssd1306


PCA_BUS, PCA_ADDR = 1, 0x40
OLED_BUS, OLED_ADDR = 3, 0x3C

PAN_CHANNEL = 0
TILT_CHANNEL = 2
CHANNELS = {"pan": PAN_CHANNEL, "tilt": TILT_CHANNEL}

PWM_HZ = 50

# Temporary electrical command safeguards.
# These are NOT calibrated mechanical limits.
MIN_US = 1000.0
MAX_US = 2000.0

SPEEDS = (3, 5, 10, 20, 30, 40, 60, 80, 100, 150, 200)
STEPS = (1, 2, 5, 10)
MARKS_FILE = Path(__file__).resolve().parents[2] / "runtime" / "keyboard_servo_marks.json"


@dataclass(frozen=True)
class AxisMarks:
    center: Optional[float] = None
    low: Optional[float] = None
    high: Optional[float] = None

    def __post_init__(self):
        for pulse in (self.center, self.low, self.high):
            if pulse is not None and (
                type(pulse) not in (int, float) or not MIN_US <= pulse <= MAX_US
            ):
                raise ValueError("Mark outside the electrical pulse range.")
        if self.low is not None and (self.center is None or self.low >= self.center):
            raise ValueError("Mark the lower limit below the center.")
        if self.high is not None and (self.center is None or self.high <= self.center):
            raise ValueError("Mark the upper limit above the center.")

    def bounds(self):
        return (MIN_US if self.low is None else self.low,
                MAX_US if self.high is None else self.high)

    def degrees(self, pulse):
        if self.center is None:
            return us_to_deg(pulse)
        return us_to_deg(pulse) - us_to_deg(self.center)

    def describe(self):
        parts = []
        for name, value in asdict(self).items():
            pulse = "unset" if value is None else f"{value:.1f} us"
            parts.append(f"{name} {pulse}")
        return ", ".join(parts)


def load_marks():
    try:
        data = json.loads(MARKS_FILE.read_text())
        if not isinstance(data, dict) or set(data) != set(CHANNELS):
            raise ValueError("Expected pan and tilt marks.")
        return {axis: AxisMarks(**data[axis]) for axis in CHANNELS}
    except FileNotFoundError:
        return {axis: AxisMarks() for axis in CHANNELS}
    except (OSError, TypeError, ValueError) as error:
        raise SystemExit(f"Cannot load {MARKS_FILE}: {error}") from error


def save_marks(marks):
    MARKS_FILE.parent.mkdir(parents=True, exist_ok=True)
    f = NamedTemporaryFile(mode="w", dir=MARKS_FILE.parent, delete=False)
    tmp = Path(f.name)
    try:
        with f:
            json.dump({axis: asdict(m) for axis, m in marks.items()}, f, indent=2)
            f.write("\n")
        tmp.replace(MARKS_FILE)
    finally:
        tmp.unlink(missing_ok=True)


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

    marks = load_marks()
    selected = "pan"
    speed_i = 1
    step_i = 0

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
                        f"{marks[axis].degrees(pulse):+.0f}* "
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
                    f"{marks[axis].degrees(pulse):+.1f}* "
                    f"({pulse:.0f}us)"
                )

        print(
            "\r" + " | ".join(values)
            + f" | Selected {selected.upper()}"
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

        low, high = marks[axis].bounds()
        if marks[axis].center is not None:
            print(f"Marked center: {marks[axis].center:.1f} us.")
        value = line_input(
            f"Initial {axis.upper()} pulse "
            f"({low:g}-{high:g} us, "
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
            or not low <= pulse <= high
        ):
            print("Outside allowed pulse range.")
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

        ref = marks[axis]
        low, high = ref.bounds()
        origin = "from marked center" if ref.center is not None else "absolute"
        value = line_input(
            f"{axis.upper()} nominal degrees {origin} "
            f"({ref.degrees(low):+.1f} to {ref.degrees(high):+.1f}, blank cancels): "
        )

        if not value:
            return

        try:
            degrees = float(value)
        except ValueError:
            print("Not a number.")
            return

        pulse = deg_to_us(degrees) if ref.center is None else (
            ref.center + degrees * (MAX_US - MIN_US) / 180
        )
        if not low <= pulse <= high:
            print("Outside allowed travel.")
            return

        if abs(degrees - ref.degrees(current[axis])) > 20:
            if line_input(
                "Large command. Type YES: "
            ) != "YES":
                print("Cancelled.")
                return

        target[axis] = pulse

    def calibration_key(axis, key):
        ref = marks[axis]
        pulse = current[axis]
        if key == "r":
            target[axis] = pulse
            if line_input(f"Type RESET to clear {axis.upper()} center and limits: ") != "RESET":
                return
            updated = AxisMarks()
        elif pulse is None:
            print(f"\nArm {axis.upper()} first.")
            return
        elif key == "0":
            if ref.center is None:
                print("\nJog to center and press C first.")
            else:
                target[axis] = ref.center
            return
        else:
            # Mark the current command and cancel any remaining jog.
            target[axis] = pulse
            field = {"c": "center", "[": "low", "]": "high"}[key]
            try:
                updated = replace(ref, **{field: pulse})
            except ValueError as error:
                print(f"\n{error}")
                return
        try:
            save_marks({**marks, axis: updated})
        except OSError as error:
            print(f"\nCould not save calibration: {error}")
            return
        marks[axis] = updated
        print(f"\n{axis.upper()} saved: {updated.describe()}")

    try:
        bus = SMBus(PCA_BUS)

        # Setup first: it enables auto-increment, which
        # the multi-byte off() writes depend on.
        setup(bus)

        for channel in CHANNELS.values():
            off(bus, channel)

        oled = ssd1306(
            i2c(port=OLED_BUS, address=OLED_ADDR),
            width=128,
            height=64,
        )

        draw()

        print("\nWHISPRYN KEYBOARD CALIBRATION")
        print("--------------------------------")
        print("1/2  : Select and arm PAN / TILT")
        print("A/D  : PAN left/right")
        print("W/S  : TILT up/down")
        print("Z/X  : Decrease/increase speed")
        print("N/M  : Decrease/increase degree step")
        print("Jog or press 1/2 to select the axis for the following keys:")
        print("C    : Mark current position as center (0 deg)")
        print("[ / ]: Mark lower / upper travel limit, short of the physical stop")
        print("0    : Return selected axis to center at selected speed")
        print("R    : Clear selected axis marks to recalibrate")
        print("P/T  : Exact PAN / TILT degrees, relative to center once marked")
        print("H    : Halt and hold")
        print("SPACE: Disable PWM")
        print("Q    : Quit")
        print("--------------------------------")
        print("First setup with unknown position: disconnect the linkage before ARM.")
        print("1500 us is the electrical midpoint. Jog to find the mounted center.")
        for axis, ref in marks.items():
            print(f"{axis.upper()} marks: {ref.describe()}")

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
                    selected = "pan" if key == "1" else "tilt"
                    arm(selected)
                    last = time.monotonic()

                elif key in ("c", "[", "]", "0", "r"):
                    calibration_key(selected, key)
                    last = time.monotonic()

                elif key in ("p", "t"):
                    selected = "pan" if key == "p" else "tilt"
                    exact_target(selected)
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
                    selected = axis

                    if target[axis] is None:
                        print(
                            f"\nArm {axis.upper()} first."
                        )
                    else:
                        direction = (
                            1 if key in ("d", "w") else -1
                        )

                        # Keep at most one jog ahead while exploring unknown travel.
                        delta_us = (
                            deg_to_us(STEPS[step_i])
                            - MIN_US
                        )

                        low, high = marks[axis].bounds()
                        target[axis] = max(
                            low,
                            min(
                                high,
                                current[axis]
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
                status()
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
