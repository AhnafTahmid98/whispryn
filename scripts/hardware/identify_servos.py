import time
from smbus import SMBus

I2C_BUS = 1
PCA9685_ADDRESS = 0x40
PWM_FREQUENCY = 50

CHANNELS = [2, 3]

MODE1 = 0x00
PRESCALE = 0xFE
LED0_ON_L = 0x06


def configure_pca9685(bus):
    old_mode = bus.read_byte_data(PCA9685_ADDRESS, MODE1)

    # Sleep before updating PWM frequency
    sleep_mode = (old_mode & 0x7F) | 0x10
    bus.write_byte_data(PCA9685_ADDRESS, MODE1, sleep_mode)

    prescale = round(
        25_000_000 / (4096 * PWM_FREQUENCY)
    ) - 1

    bus.write_byte_data(
        PCA9685_ADDRESS, PRESCALE, prescale
    )

    # Wake and enable auto-increment
    wake_mode = (old_mode & 0x6F) | 0x20
    bus.write_byte_data(
        PCA9685_ADDRESS, MODE1, wake_mode
    )

    time.sleep(0.01)

    bus.write_byte_data(
        PCA9685_ADDRESS, MODE1, wake_mode | 0x80
    )


def set_pulse(bus, channel, pulse_us):
    ticks = round(
        pulse_us * 4096 * PWM_FREQUENCY / 1_000_000
    )

    register = LED0_ON_L + 4 * channel

    bus.write_i2c_block_data(
        PCA9685_ADDRESS,
        register,
        [0, 0, ticks & 0xFF, (ticks >> 8) & 0x0F]
    )


def disable_servo(bus, channel):
    register = LED0_ON_L + 4 * channel

    bus.write_i2c_block_data(
        PCA9685_ADDRESS,
        register,
        [0, 0, 0, 0x10]
    )


def main():
    print("=" * 42)
    print("WHISPRYN SERVO IDENTIFICATION")
    print("=" * 42)

    with SMBus(I2C_BUS) as bus:
        configure_pca9685(bus)

        # Disable outputs before testing
        for channel in CHANNELS:
            disable_servo(bus, channel)

        try:
            for channel in CHANNELS:
                print(f"\nReady to test CHANNEL {channel}")
                input("Press ENTER to start...")

                print("Moving to centre (1500 us)")
                set_pulse(bus, channel, 1500)
                time.sleep(1.5)

                print("Small movement (1560 us)")
                set_pulse(bus, channel, 1560)
                time.sleep(0.8)

                print("Returning to centre")
                set_pulse(bus, channel, 1500)
                time.sleep(1)

                disable_servo(bus, channel)

                print(f"\nCHANNEL {channel} test completed.")
                print("Observe which servo moved:")
                print("PAN  = left/right rotation")
                print("TILT = up/down rotation")

                input("Press ENTER to continue...")

        finally:
            for channel in CHANNELS:
                disable_servo(bus, channel)

    print("\nWHISPRYN identification finished!")


if __name__ == "__main__":
    main()
