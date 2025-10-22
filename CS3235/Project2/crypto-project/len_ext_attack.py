#!/usr/bin/python3

# Run me like this:
# $ python3 len_ext_attack.py "https://cryptoproject.gtinfosec.org/GTusername/lengthextension/api?token=....&command=...."
# or select "Length Extension" from the VS Code debugger

import sys
from urllib.parse import quote
from pysha256 import sha256, padding


class URL:
    def __init__(self, url: str):
        # prefix is the slice of the URL from "https://" to "token=", inclusive.
        self.prefix = url[:url.find('=') + 1].replace('\\','')
        self.token = url[url.find('=') + 1:url.find('&')].replace('\\','')
        # suffix starts at the first "command=" and goes to the end of the URL
        self.suffix = url[url.find('&') + 1:].replace('\\','')

    def __str__(self) -> str:
        return f'{self.prefix}{self.token}&{self.suffix}'

    def __repr__(self) -> str:
        return f'{type(self).__name__}({str(self).__repr__()})'


def main():
    if len(sys.argv) < 2:
        print(f"usage: {sys.argv[0]} URL_TO_EXTEND", file=sys.stderr)
        sys.exit(-1)

    url = URL(sys.argv[1])

    #
    # TODO: Modify the URL
    #
    
    length = 8 + len(url.suffix.encode("ascii"))
    pad_length = length + len(padding(length))

    new_command = "&command=UnlockSafes"
    new_hash = sha256(
        state=bytes.fromhex(url.token),
        count=pad_length,
    )

    new_hash.update(new_command.encode())


    url.token = new_hash.hexdigest()
    url.suffix += quote(padding(length)) + new_command

    print(url)


if __name__ == '__main__':
    main()
