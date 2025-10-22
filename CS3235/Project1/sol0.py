#!/usr/bin/env python3

import sys

name = b"dlim70"
padding = b"\x00" * (10 - len(name))
grade = b"A+\x00"

combined = name + padding + grade

sys.stdout.buffer.write(combined)