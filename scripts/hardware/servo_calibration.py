
"""
WHISPRYN - Pan/Tilt Calibration with OLED

PAN:  PCA9685 channel 3
TILT: PCA9685 channel 2
OLED: I2C bus 3, address 0x3C

No automatic servo movement at startup.
"""

import json
import math
import time
from pathlib import Path

from smbus import SMBus
from luma.core.interface.serial import i2c
from luma.core.render import canvas
from luma.oled.device import ssd1306

PCA_ADDR = 0x40
PCA_BUS = 1
PWM_HZ = 50

PAN_CHANNEL = 3
TILT_CHANNEL = 2

OLED_BUS = 3
OLED_ADDR = 0x3C

# Provisional mechanical limits
LIMITS = {
    "pan": (45, 135),
    "tilt": (65, 115),
}

CHANNELS = {
    "pan": PAN_CHANNEL,
    "tilt": TILT_CHANNEL,
}

# Conservative pulse range for initial testing
PULSE_MIN_US = 1000
PULSE_MAX_US = 2000

ROOT = Path(__file__).resolve().parents[2]
SAVE_PATH = ROOT / "runtime" / "servo_calibration.json"

positions = {"pan": None, "tilt": None}


def configure_pca(bus):
    mode = bus.read_byte_data(PCA_ADDR, 0x00)

    # Sleep before setting the PWM frequency
    bus.write_byte_data(
        PCA_ADDR, 0x00, (mode & 0x7F) | 0x10
    )

    prescale = round(25_000_000 / (4096 * PWM_HZ)) - 1
    bus.write_byte_data(PCA_ADDR, 0xFE, prescale)

    # Wake, enable auto-increment, restart
    wake = (mode & 0x6F) | 0x20
    bus.write_byte_data(PCA_ADDR, 0x00, wake)
    time.sleep(0.01)
    bus.write_byte_data(PCA_ADDR, 0x00, wake | 0x80)


def set_servo(bus, channel, angle):
    pulse_us = (
        PULSE_MIN_US
        + (angle / 180.0)
        * (PULSE_MAX_US - PULSE_MIN_US)
    )

    ticks = round(pulse_us * 4096 * PWM_HZ / 1_000_000)
    reg = 0x06 + 4 * channel

    bus.write_i2c_block_data(
        PCA_ADDR,
        reg,
        [0, 0, ticks & 0xFF, (ticks >> 8) & 0x0F],
    )


def disable_servo(bus, channel):
    reg = 0x06 + 4 * channel
    bus.write_i2c_block_data(
        PCA_ADDR, reg, [0, 0, 0, 0x10]
    )


def show_oled(oled, status):
    pan = positions["pan"]
    tilt = positions["tilt"]

    with canvas(oled) as draw:
        draw.text((6, 2), "WHISPRYN", fill="white")
        draw.line((0, 17, 127, 17), fill="white")

        draw.text(
            (6, 23),
            f"PAN : {pan if pan is not None else '--'}",
            fill="white",
        )

        draw.text(
            (6, 36),
            f"TILT: {tilt if tilt is not None else '--'}",
            fill="white",
        )

        draw.text((6, 52), status[:20], fill="white")


def move(bus, oled, axis, target):
    minimum, maximum = LIMITS[axis]

    if not minimum <= target <= maximum:
        print(
            f"Rejected. {axis.upper()} range: "
            f"{minimum} to {maximum}"
        )
        return

    previous = positions[axis]

    if previous is None:
        print(
            f"First {axis.upper()} command may cause "
            "a sudden movement."
        )

        answer = input(
            f"Clear mechanism? Type MOVE to command {target}: "
        )

        if answer != "MOVE":
            print("Movement cancelled.")
            return

        set_servo(bus, CHANNELS[axis], target)

    else:
        # Gradual movement: maximum ~2 degrees per step
        steps = max(
            1, math.ceil(abs(target - previous) / 2)
        )

        for step in range(1, steps + 1):
            angle = previous + (
                (target - previous) * step / steps
            )
            set_servo(bus, CHANNELS[axis], angle)
            time.sleep(0.04)

    positions[axis] = target
    show_oled(oled, f"{axis.upper()} {target}")
    print(f"{axis.upper()} commanded to {target}")


def save_calibration():
    answer = input(
        "Verified both travel endpoints, screw clearance "
        "and cable slack physically? [yes/no]: "
    ).strip().lower()

    data = {
        "i2c_bus": PCA_BUS,
        "pca_address": "0x40",
        "channels": CHANNELS,
        "limits": LIMITS,
        "pulse_min_us": PULSE_MIN_US,
        "pulse_max_us": PULSE_MAX_US,
        "operator_verified": answer == "yes",
    }

    SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
    SAVE_PATH.write_text(json.dumps(data, indent=2))

    print("Calibration saved to:", SAVE_PATH)
    if answer != "yes":
        print("Saved as PROVISIONAL, not physically verified.")


def main():
    bus = SMBus(PCA_BUS)
    oled = None

    try:
        # Stop both outputs before configuration
        for channel in CHANNELS.values():
            disable_servo(bus, channel)

        configure_pca(bus)

        oled = ssd1306(
            i2c(port=OLED_BUS, address=OLED_ADDR),
            width=128,
            height=64,
        )

        show_oled(oled, "CALIBRATION")

        print("\nWHISPRYN SERVO CALIBRATION")
        print("PAN  = channel 3")
        print("TILT = channel 2")
        print("Commands:")
        print("  pan 90")
        print("  tilt 90")
        print("  off")
        print("  save")
        print("  quit")

        while True:
            command = input("\nWHISPRYN> ").strip().lower()

            if command == "quit":
                break

            elif command == "off":
                for channel in CHANNELS.values():
                    disable_servo(bus, channel)

                positions["pan"] = None
                positions["tilt"] = None
                show_oled(oled, "MOTORS OFF")

            elif command == "save":
                save_calibration()

            elif command.startswith(("pan ", "tilt ")):
                try:
                    axis, angle = command.split()
                    move(bus, oled, axis, int(angle))
                except ValueError:
                    print("Example: pan 100")

            else:
                print("Use pan N, tilt N, off, save or quit")

    except (KeyboardInterrupt, EOFError):
        print("\nCalibration interrupted.")

    finally:
        try:
            for channel in CHANNELS.values():
                disable_servo(bus, channel)
        finally:
            bus.close()

        if oled is not None:
            oled.clear()

        print("Servo outputs disabled.")


if __name__ == "__main__":
    main()
