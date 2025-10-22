# Some imports that we think may be useful to you.
from PIL import Image
from pyzbar.pyzbar import decode
from urllib.parse import parse_qs, urlparse
import base64
import hmac
import sys
import time

# Referred to https://www.rfc-editor.org/rfc/rfc4226#section-5.4
# for implementation example for HOTP.
def hotp(k, c, digits):
    counter_bytes = bytearray(8)
    for i in range(7, -1, -1):
        counter_bytes[i] = c & 0xFF
        c >>= 8
    
    mac = hmac.new(k, bytes(counter_bytes), "sha1").digest()

    offset = mac[-1] & 0x0F
    bin_code = ((mac[offset] & 0x7F) << 24) | \
                ((mac[offset + 1] & 0xFF) << 16) | \
                ((mac[offset + 2] & 0xFF) << 8) | \
                (mac[offset + 3] & 0xFF)

    return bin_code % (10 ** digits)


contents = decode(Image.open(sys.argv[1]))[0].data.decode()
# Referred to https://docs.python.org/3/library/urllib.parse.html#urllib.parse.urlparse
# for parsing URL
parsed = urlparse(contents)
params = parse_qs(parsed.query)

# Referred to https://github.com/google/google-authenticator/wiki/Key-Uri-Format
# for otpauth format
secret_b32 = params["secret"][0].upper()
period = int(params.get("period", ["30"])[0])
digits = int(params.get("digits", ["6"])[0])

key = base64.b32decode(secret_b32)

# Referred to https://www.rfc-editor.org/rfc/rfc6238#section-4
# for TOTP implementation
current_time = int(time.time())
counter = current_time // period

code = hotp(key, counter, digits)
print(code)