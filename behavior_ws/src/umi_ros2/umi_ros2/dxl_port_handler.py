#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Dynamixel SDK port handler compatibility helpers."""

import errno

import serial
from dynamixel_sdk import PortHandler as DxlPortHandler


class IgnoreModemControlEIOSerial(serial.Serial):
    """Serial port that tolerates FTDI modem-control EIO during open."""

    def _update_dtr_state(self):
        try:
            super()._update_dtr_state()
        except OSError as exc:
            if exc.errno != errno.EIO:
                raise

    def _update_rts_state(self):
        try:
            super()._update_rts_state()
        except OSError as exc:
            if exc.errno != errno.EIO:
                raise


class RobustPortHandler(DxlPortHandler):
    """Dynamixel PortHandler variant for adapters that reject DTR/RTS ioctl."""

    def setupPort(self, cflag_baud):
        if self.is_open:
            self.closePort()

        self.ser = IgnoreModemControlEIOSerial(
            port=self.port_name,
            baudrate=self.baudrate,
            bytesize=serial.EIGHTBITS,
            timeout=0,
        )

        self.is_open = True
        self.ser.reset_input_buffer()
        self.tx_time_per_byte = (1000.0 / self.baudrate) * 10.0

        return True
