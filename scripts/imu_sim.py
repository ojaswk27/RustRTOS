#!/usr/bin/env python3
"""
imu_sim.py — Simulated IMU data stream for rtdrone demo (Track B).

Generates MPU6050-style raw accel+gyro binary packets at ~100Hz,
suitable for feeding into QEMU's UART1 via serial passthrough.

Packet format: 6 × int16_t big-endian = ax, ay, az, gx, gy, gz
  - Accelerometer: units = 1/16384 g (±2g range), 1.0g = 16384
  - Gyroscope: units = 1/131 deg/s (±250 dps range)

Usage:
  # Simulate level hovering (small perturbations):
  python3 scripts/imu_sim.py | <pipe to QEMU UART1>

  # QEMU passthrough for UART1:
  make qemu CPUS=1 QEMUEXTRA="-serial /dev/ttyUSB0"   # real hardware
  # or:
  mkfifo /tmp/imu_pipe
  python3 scripts/imu_sim.py > /tmp/imu_pipe &
  make qemu CPUS=1 QEMUEXTRA="-chardev pipe,id=imu,path=/tmp/imu_pipe -serial chardev:imu"
"""

import sys
import time
import math
import struct
import argparse


def simulate_imu(scenario: str = "hover", rate_hz: int = 100):
    """
    Generate realistic IMU data for different flight scenarios.

    Scenarios:
      hover    — small Gaussian noise around level attitude
      roll     — sinusoidal roll perturbation (±15 deg)
      turbulence — random high-frequency vibrations
    """
    t = 0.0
    dt = 1.0 / rate_hz

    # Noise level (raw ADC units)
    accel_noise = 50   # ~3mg
    gyro_noise  = 5    # ~0.04 deg/s

    # Gravity in accelerometer units (1g = 16384 LSB)
    GRAVITY = 16384

    while True:
        # Base: hovering level with gravity on Z axis
        ax = 0
        ay = 0
        az = GRAVITY

        gx = 0
        gy = 0
        gz = 0

        if scenario == "hover":
            # Small oscillations (PID control overshoot simulation)
            ax += int(200 * math.sin(t * 2.0))
            ay += int(150 * math.cos(t * 1.7))
            gx += int(50  * math.sin(t * 2.5))
            gy += int(40  * math.cos(t * 3.1))

        elif scenario == "roll":
            # Roll maneuver: ±15 degrees at 0.5Hz
            roll_angle = 15.0 * math.sin(t * math.pi)  # degrees
            roll_rad   = math.radians(roll_angle)
            ay += int(GRAVITY * math.sin(roll_rad))
            az  = int(GRAVITY * math.cos(roll_rad))
            gx += int(131 * 30.0 * math.cos(t * math.pi))  # 30 deg/s peak

        elif scenario == "turbulence":
            # High-frequency vibration (motor harmonics at ~100Hz)
            ax += int(500 * math.sin(t * 100.0 * 2 * math.pi))
            ay += int(400 * math.cos(t * 100.0 * 2 * math.pi))
            az += int(300 * math.sin(t *  67.0 * 2 * math.pi))

        # Add sensor noise (pseudo-random using sin/cos tricks)
        import random
        ax += int(random.gauss(0, accel_noise))
        ay += int(random.gauss(0, accel_noise))
        az += int(random.gauss(0, accel_noise))
        gx += int(random.gauss(0, gyro_noise))
        gy += int(random.gauss(0, gyro_noise))
        gz += int(random.gauss(0, gyro_noise))

        # Clamp to int16 range
        def clamp16(v):
            return max(-32768, min(32767, v))

        pkt = struct.pack(">6h",
                          clamp16(ax), clamp16(ay), clamp16(az),
                          clamp16(gx), clamp16(gy), clamp16(gz))

        sys.stdout.buffer.write(pkt)
        sys.stdout.buffer.flush()

        t += dt
        time.sleep(dt)


def main():
    parser = argparse.ArgumentParser(description="Simulated IMU data stream for rtdrone demo")
    parser.add_argument("--scenario", choices=["hover", "roll", "turbulence"],
                        default="hover", help="Flight scenario to simulate (default: hover)")
    parser.add_argument("--rate", type=int, default=100,
                        help="Output rate in Hz (default: 100)")
    args = parser.parse_args()

    print(f"[imu_sim] Streaming {args.scenario} scenario at {args.rate}Hz",
          file=sys.stderr)
    print(f"[imu_sim] Packet format: 6×int16 big-endian (ax,ay,az,gx,gy,gz)",
          file=sys.stderr)
    print(f"[imu_sim] Press Ctrl-C to stop", file=sys.stderr)

    try:
        simulate_imu(args.scenario, args.rate)
    except KeyboardInterrupt:
        print("\n[imu_sim] Stopped", file=sys.stderr)


if __name__ == "__main__":
    main()
