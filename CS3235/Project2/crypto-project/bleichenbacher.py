#!/usr/bin/python3

# Run me like this:
# $ python3 bleichenbacher.py from_username+to_username+100.00
# or select "Bleichenbacher" from the VS Code debugger

from roots import *

import hashlib
import sys

# def forge_signature(message: str) -> int:
#     modulus = 0x00ba19093eb788b413b1a71981044ae3cd9cd9089ea9f45478e3832d9a6111d1715e3fa67e27c28e32edd4b720cdff40fda93d49610195103de04ba00b494bd457
#     e = 3
#     hashed = hashlib.sha256(message.encode()).digest()
#     ASN = bytes.fromhex("3031300d060960864801650304020105000420")

#     padding_length = (2048 // 8) - len(ASN) - len(hashed) - 3
#     forged_value = b"\x00\x01" + (b"\xff" * padding_length) + b"\x00" + ASN + hashed
#     forged_int = int.from_bytes(forged_value, byteorder="big")
#     forged_signature = integer_nthroot(forged_int, e)[0]
#     return forged_signature

def main():
    if len(sys.argv) < 2:
        print(f"usage: {sys.argv[0]} MESSAGE", file=sys.stderr)
        sys.exit(-1)
    message = sys.argv[1]

    temp = hex(0x0001FF003031300D060960864801650304020105000420)
    temp += hashlib.sha256(message.encode("latin-1")).hexdigest() + hex(0x00)[2:] * 402

    forged_signature, check = integer_nthroot(int(temp, 16), 3)
    forged_signature = forged_signature if check else forged_signature + 1
    
    print(bytes_to_base64(integer_to_bytes(forged_signature, 256)))


if __name__ == '__main__':
    main()
