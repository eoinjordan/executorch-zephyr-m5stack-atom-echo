#!/usr/bin/env python3
# Copyright (c) 2026 Eoin Jordan
# SPDX-License-Identifier: Apache-2.0
"""Home Assistant bridge for the ATOM Echo wake-word device.

Reads the JSON lines the firmware prints on its USB serial port and publishes
them over MQTT with Home Assistant discovery, so the device shows up by itself
as:

  event.hey_zephyr          fires "hey_zephyr" on every wake event (automation trigger)
  binary_sensor.hey_zephyr_listening   on while the device is unmuted and alive
  sensor.hey_zephyr_p_wake  the latest p(wake) from the heartbeat

Run it on the machine the ATOM is plugged into (the HA host, a Pi, your PC):

  pip install paho-mqtt pyserial
  python ha_bridge.py --port COM3 --broker homeassistant.local --user mqtt --password ...

Wyoming-style audio streaming to HA Assist is the next step; this bridge only
carries events. See the README.
"""

import argparse
import json
import socket
import sys
import time

try:
    import paho.mqtt.client as mqtt
except ImportError:
    sys.exit("pip install paho-mqtt pyserial")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--port", required=True, help="serial port of the ATOM Echo, or - for stdin")
    p.add_argument("--baud", type=int, default=115200)
    p.add_argument("--broker", required=True)
    p.add_argument("--mqtt-port", type=int, default=1883)
    p.add_argument("--user")
    p.add_argument("--password")
    p.add_argument("--device-id", default="atom_echo_" + socket.gethostname().lower().replace("-", "_"))
    p.add_argument("--name", default="Hey Zephyr")
    args = p.parse_args()

    base = f"heyzephyr/{args.device_id}"
    avail = f"{base}/availability"
    device = {"identifiers": [args.device_id], "name": args.name, "manufacturer": "M5Stack",
              "model": "ATOM Echo, Zephyr + ExecuTorch wake word", "sw_version": "executorch-zephyr-m5stack-atom-echo"}

    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=args.device_id)
    if args.user:
        client.username_pw_set(args.user, args.password)
    client.will_set(avail, "offline", retain=True)
    client.connect(args.broker, args.mqtt_port, keepalive=60)
    client.loop_start()

    def discover(component, object_id, payload):
        payload.update({"availability_topic": avail, "device": device, "unique_id": f"{args.device_id}_{object_id}"})
        client.publish(f"homeassistant/{component}/{args.device_id}/{object_id}/config", json.dumps(payload), retain=True)

    discover("event", "wake", {"name": f"{args.name}", "state_topic": f"{base}/event", "event_types": ["hey_zephyr"],
                               "device_class": "button"})
    discover("binary_sensor", "listening", {"name": f"{args.name} listening", "state_topic": f"{base}/listening",
                                             "payload_on": "ON", "payload_off": "OFF", "device_class": "running"})
    discover("sensor", "p_wake", {"name": f"{args.name} p(wake)", "state_topic": f"{base}/p_wake",
                                  "unit_of_measurement": "%", "state_class": "measurement"})
    client.publish(avail, "online", retain=True)
    print(f"announced {args.device_id} to Home Assistant via {args.broker}")

    if args.port == "-":
        readline = sys.stdin.buffer.readline
    else:
        import serial

        readline = serial.Serial(args.port, args.baud, timeout=5).readline

    last_beat = time.time()
    try:
        while True:
            raw = readline()
            if not raw:
                if time.time() - last_beat > 30:
                    client.publish(f"{base}/listening", "OFF", retain=True)
                continue
            line = raw.decode(errors="replace").strip()
            if not line.startswith("{"):
                if line:
                    print(f"[mcu] {line}")
                continue
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                continue
            last_beat = time.time()
            if ev.get("wake") == 1:
                client.publish(f"{base}/event", json.dumps({"event_type": "hey_zephyr", "p": ev.get("p"),
                                                            "infer_us": ev.get("infer_us")}))
                print(f"*** HEY ZEPHYR  p={ev.get('p')}")
            if "p_wake" in ev:
                client.publish(f"{base}/p_wake", f"{100 * float(ev['p_wake']):.1f}", retain=True)
            if "muted" in ev:
                client.publish(f"{base}/listening", "OFF" if ev["muted"] else "ON", retain=True)
            elif "p_wake" in ev:
                client.publish(f"{base}/listening", "ON", retain=True)
    except KeyboardInterrupt:
        pass
    finally:
        client.publish(avail, "offline", retain=True)
        client.loop_stop()


if __name__ == "__main__":
    main()
