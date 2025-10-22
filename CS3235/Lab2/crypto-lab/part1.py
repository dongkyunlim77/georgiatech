gtinfosec = b'\x03\x02\x03\x05'


def str_to_bytes(s: str) -> bytes:
    return s.encode('ascii')


def hex_to_bytes(s: str) -> bytes:
    return bytes.fromhex(s)


def int_to_bytes(n: int) -> bytes:
    return n.to_bytes(4, 'big')


def set_end_to_zero(b):
    b[-1] = 0


# Feel free to edit the main function however you like to help you debug, it won't be graded
#
# Run this script with the command: python3 part1.py
# or select "Part 1" from the VS Code debugger
def main():
    print(gtinfosec)

    bytes1 = str_to_bytes('Hello, world!')
    print(bytes1)

    bytes2 = hex_to_bytes('a2f295ac')
    print(bytes2)

    bytes3 = int_to_bytes(100599730)
    print(bytes3)

    b = bytearray(b'\x01\x02\x03\x04\x05')
    set_end_to_zero(b)
    print(b)


if __name__ == '__main__':
    main()
