# Data Plan — WHISPRYN

**Team:** Ahnaf Tahmid, Asikul Haque Rafsan, Farhana Faruk, Sayed Bhuiyan.

## Sensor

We will use an ESP32-S3 with two I²S MEMS microphones to recognize sounds and estimate their direction. Two servo motors will control horizontal (pan) and vertical (tilt) movement.

## Classes

Our TinyML model will recognize four classes:

- **Clap:** Activates an LED.
- **Whistle:** Triggers sound tracking and servo movement.
- **Hey:** Activates attention mode.
- **Background / Idle:** No action.

## Collection

Each team member will record 20 samples per class using the microphones.

We will collect recordings from different people, distances, and indoor environments. For whistle detection, we will include sounds from the left, centre, and right.

We will use **80% of recordings for training and 20% for testing**, keeping complete recordings separate.

## Risk

**Hey and Background / Idle** may be confused because ordinary conversations can resemble the wake word.

We will use the same two microphones to collect sound recordings from different people in various environments to improve recognition accuracy.

## Waiting for

We currently have the ESP32-S3. We plan to order two I²S microphones, two servo motors, and mounting accessories from DigiKey or Mouser. Data collection will begin after the microphones arrive.
