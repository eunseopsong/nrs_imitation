#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Ping-scan Dynamixel IDs without sending motion commands."""

import argparse

from dynamixel_sdk import PacketHandler, COMM_SUCCESS

from umi_ros2.dxl_port_handler import RobustPortHandler


COMMON_BAUDS = [
    57600,
    115200,
    1000000,
    2000000,
    3000000,
    4000000,
]


def parse_ids(value):
    if "-" in value:
        start, end = value.split("-", 1)
        return range(int(start), int(end) + 1)
    return [int(part) for part in value.split(",") if part.strip()]


def parse_bauds(value):
    if value == "common":
        return COMMON_BAUDS
    return [int(part) for part in value.split(",") if part.strip()]


def scan(port_name, bauds, ids, protocol):
    packet_handler = PacketHandler(protocol)
    found = []

    for baud in bauds:
        port = RobustPortHandler(port_name)
        if not port.setBaudRate(baud):
            print(f"[baud={baud}] open failed: {port_name}")
            continue

        print(f"[baud={baud}] scanning {ids.start if hasattr(ids, 'start') else ''}...", flush=True)

        try:
            for dxl_id in ids:
                model, result, error = packet_handler.ping(port, dxl_id)
                if result == COMM_SUCCESS:
                    print(
                        f"FOUND id={dxl_id} baud={baud} model={model} error={error}",
                        flush=True,
                    )
                    found.append((dxl_id, baud, model, error))
                elif result not in (-3001,):
                    result_text = packet_handler.getTxRxResult(result)
                    print(
                        f"id={dxl_id} baud={baud} result={result} ({result_text}) error={error}",
                        flush=True,
                    )
        finally:
            port.closePort()

    return found


def main(args=None):
    parser = argparse.ArgumentParser(
        description="Ping-scan Dynamixel IDs without moving the motor."
    )
    parser.add_argument(
        "--port",
        default="/dev/ttyUSB0",
        help="Serial port or /dev/serial/by-id path.",
    )
    parser.add_argument(
        "--baud",
        default="57600",
        help="Comma-separated baud list, or 'common'.",
    )
    parser.add_argument(
        "--ids",
        default="0-10",
        help="ID range like 0-10, or comma list like 0,1,2.",
    )
    parser.add_argument(
        "--protocol",
        default=2.0,
        type=float,
        help="Dynamixel protocol version.",
    )
    parsed = parser.parse_args(args=args)

    ids = parse_ids(parsed.ids)
    bauds = parse_bauds(parsed.baud)
    found = scan(parsed.port, bauds, ids, parsed.protocol)

    if found:
        print("\nSummary:")
        for dxl_id, baud, model, error in found:
            print(f"  id={dxl_id} baud={baud} model={model} error={error}")
        return 0

    print("\nNo Dynamixel responded.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
