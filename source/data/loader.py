#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compatibility wrapper for nrs_imitation data loading."""

from __future__ import annotations

from .dataset import make_loaders


def load_data(*args, **kwargs):
    return make_loaders(*args, **kwargs)
