
import os
import sys
import time
import tty
import termios
import select

from smbus import SMBus
from luma.core.interface.serial import i2c
from luma.core.render import canvas
from luma.oled.device import ssd1306

# WHISPRYN confirmed connections
PCA_BUS = 1
PCA_ADDR = 0x40
PAN_CHANNEL = 3
TILT_CHANNEL = 2

OLED_BUS = 3
OLED_ADDR = 0x3C

PWM_HZ = 50
STEP_US = 10

# Temporary electrical safeguards,
# NOT calibrated mechanical travel limits.
PULSE_MIN = 1000
PULSE_MAX = 2000

# Electrical reference only, NOT physical centre.
START_PULSE = 1500

channels = {
    "pan": PAN_CHANNEL,
    "tilt": TILT_CHANNEL,
}

positions = {"pan": None, "tilt": None}


def configure_pwm(bus):
    old = bus.read_byte_data(PCA_ADDR, 0x00)

    bus.write_byte_data(
        PCA_ADDR, 0x00, (old & 0x7F) | 0x10
    )

    prescale = round(
        25_000_000 / (4096 * PWM_HZ)
    ) - 1

    bus.write_byte_data(PCA_ADDR, 0xFE, prescale)

    wake = (old & 0x6F) | 0x20
    bus.write_byte_data(PCA_ADDR, 0x00, wake)

    time.sleep(0.01)

    bus.write_byte_data(
        PCA_ADDR, 0x00, wake | 0x80
    )


def set_pulse(bus, channel, pulse):
    ticks = round(pulse * 4096 * PWM_HZ / 1_000_000)
    reg = 0x06 + 4 * channel

    bus.write_i2c_block_data(
        PCA_ADDR,
        reg,
        [0, 0, ticks & 255, (ticks >> 8) & 15],
    )


def disable(bus, channel):
    reg = 0x06 + 4 * channel
    bus.write_i2c_block_data(
        PCA_ADDR, reg, [0, 0, 0, 0x10]
    )


def screen(oled, status):
    with canvas(oled) as draw:
        draw.text((15, 1), "WHISPRYN", fill="white")
        draw.line((0, 17, 127, 17), fill="white")

        for axis, y in [("pan", 23), ("tilt", 37)]:
            pulse = positions[axis]
            value = "--" if pulse is None else str(pulse)
            draw.text(
                (5, y),
                f"{axis.upper()}: {value} us",
                fill="white"
            )

        draw.text((5, 52), status[:21], fill="white")


def read_key(fd):
    key = os.read(fd, 1)

    if key == b"\x1b":
        sequence = b""

        for _ in range(2):
            if select.select([fd], [], [], 0.1)[0]:
                sequence += os.read(fd, 1)

        return {
            b"[A": "w",
            b"[B": "s",
            b"[C": "d",
            b"[D": "a",
        }.get(sequence, "")

    return key.decode(errors="ignore").lower()


def move(bus, oled, axis, change):
    current = positions[axis]

    if current is None:
        print(f"\nArm {axis.upper()} first.")
        return

    target = max(
        PULSE_MIN,
        min(PULSE_MAX, current + change)
    )

    if target == current:
        print("\nElectrical command safeguard reached.")
        return

    set_pulse(bus, channels[axis], target)
    positions[axis] = target

    screen(oled, "MANUAL CONTROL")

    print(
        f"\rPAN: {positions['pan']} us  "
        f"TILT: {positions['tilt']} us    ",
        end="",
        flush=True
    )


def main():
    if not sys.stdin.isatty():
        raise SystemExit("Use an interactive terminal or SSH.")

    bus = SMBus(PCA_BUS)
    fd = sys.stdin.fileno()
    old_terminal = None
    oled = None

    try:
        for channel in channels.values():
            disable(bus, channel)

        configure_pwm(bus)

        oled = ssd1306(
            i2c(port=OLED_BUS, address=OLED_ADDR),
            width=128,
            height=64
        )

        screen(oled, "OUTPUTS OFF")

        print("\nWHISPRYN MANUAL SERVO TEST")
        print("1 = Arm PAN (channel 3)")
        print("2 = Arm TILT (channel 2)")
        print("A/D = PAN -/+")
        print("W/S = TILT +/-")
        print("Arrow keys also work")
        print("SPACE = Disable both outputs")
        print("Q = Quit")
        print("\nNo mechanical centre is assumed.")
        print("First arm command uses 1500 us.")
        print("It can cause a sudden movement.")
        print("Clear the mechanism before arming.\n")

        old_terminal = termios.tcgetattr(fd)
        tty.setcbreak(fd)

        pending = None

        while True:
            key = read_key(fd)

            # Confirm first activation of either servo.
            if pending is not None:
                if key == "y":
                    set_pulse(
                        bus,
                        channels[pending],
                        START_PULSE
                    )
                    positions[pending] = START_PULSE
                    screen(oled, f"{pending.upper()} ARMED")
                    print(
                        f"\n{pending.upper()} armed at "
                        f"{START_PULSE} us"
                    )
                else:
                    print("\nArming cancelled.")

                pending = None
                continue

            if key == "q":
                break

            elif key == " ":
                for channel in channels.values():
                    disable(bus, channel)

                positions["pan"] = None
                positions["tilt"] = None

                screen(oled, "OUTPUTS OFF")
                print("\nBoth outputs disabled.")

            elif key in ("1", "2"):
                axis = "pan" if key == "1" else "tilt"

                if positions[axis] is None:
                    pending = axis
                    print(
                        f"\nArm {axis.upper()} at "
                        f"{START_PULSE} us? "
                        "Press Y to confirm, "
                        "any other key to cancel."
                    )

            elif key == "a":
                move(bus, oled, "pan", -STEP_US)

            elif key == "d":
                move(bus, oled, "pan", STEP_US)

            elif key == "w":
                move(bus, oled, "tilt", STEP_US)

            elif key == "s":
                move(bus, oled, "tilt", -STEP_US)

            time.sleep(0.04)

    except KeyboardInterrupt:
        print("\nInterrupted.")

    finally:
        if old_terminal is not None:
            termios.tcsetattr(
                fd, termios.TCSADRAIN, old_terminal
            )

        for channel in channels.values():
            try:
                disable(bus, channel)
            except OSError:
                pass

        bus.close()

        if oled is not None:
            oled.clear()

        print("\nWHISPRYN stopped. PWM disabled.")


if __name__ == "__main__":
    main()
