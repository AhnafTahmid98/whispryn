import lgpio
import time

# GPIO pins
SERVO1 = 18
SERVO2 = 19

# Open GPIO chip
chip = lgpio.gpiochip_open(4)

# Set both GPIO pins as outputs
lgpio.gpio_claim_output(chip, SERVO1)
lgpio.gpio_claim_output(chip, SERVO2)

def set_servo(pin, angle):
    pulse = 500 + (angle / 180) * 2000
    lgpio.tx_servo(chip, pin, int(pulse))

try:
    print("Testing two servos together...")

    print("Both servos moving to 90 degrees")
    set_servo(SERVO1, 90)
    set_servo(SERVO2, 90)
    time.sleep(2)

    print("Both servos moving to 45 degrees")
    set_servo(SERVO1, 45)
    set_servo(SERVO2, 45)
    time.sleep(2)

    print("Both servos moving to 135 degrees")
    set_servo(SERVO1, 135)
    set_servo(SERVO2, 135)
    time.sleep(2)

    print("Returning both servos to 90 degrees")
    set_servo(SERVO1, 90)
    set_servo(SERVO2, 90)
    time.sleep(2)

    print("Test completed!")

finally:
    lgpio.tx_servo(chip, SERVO1, 0)
    lgpio.tx_servo(chip, SERVO2, 0)
    lgpio.gpiochip_close(chip)
