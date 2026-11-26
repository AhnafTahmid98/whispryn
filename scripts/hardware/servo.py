import lgpio
import time

SERVO_PIN = 18

chip = lgpio.gpiochip_open(4)

try:
    lgpio.gpio_claim_output(chip, SERVO_PIN)

    print("Testing SG90 servo...")

    for angle in [90, 45, 135, 90]:
        pulse = 1000 + (angle / 180) * 1000

        print(f"Moving to {angle} degrees")

        duty = pulse / 20000 * 100
        lgpio.tx_pwm(chip, SERVO_PIN, 50, duty)

        time.sleep(2)

    lgpio.tx_pwm(chip, SERVO_PIN, 0, 0)

    print("Servo test finished!")

finally:
    lgpio.gpiochip_close(chip)