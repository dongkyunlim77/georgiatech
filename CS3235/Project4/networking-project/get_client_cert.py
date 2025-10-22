# Some imports that we think may be useful to you.
from cryptography.hazmat.primitives.asymmetric import ed25519
from cryptography.hazmat.primitives.serialization import (
    load_pem_private_key,
    Encoding,
    PublicFormat,
    PrivateFormat,
    NoEncryption,
)
import socket
import ssl
import sys


def generate_or_load_private_key(filename="private.pem"):
    try:
        return load_pem_private_key(open(filename, "rb").read(), password=None)
    except FileNotFoundError:
        pass
    private_key = ed25519.Ed25519PrivateKey.generate()
    open(filename, "wb").write(
        private_key.private_bytes(
            encoding=Encoding.PEM,
            format=PrivateFormat.PKCS8,
            encryption_algorithm=NoEncryption(),
        )
    )
    return private_key


def get_public_bytes(priv):
    pub = priv.public_key()
    return pub.public_bytes(
        encoding=Encoding.PEM, format=PublicFormat.SubjectPublicKeyInfo
    )


public_key = get_public_bytes(generate_or_load_private_key())
mdm_domain = sys.argv[1]
mdm_port = sys.argv[2]

username = b"linmcgee"
password = b"camille"

def encode(username, password, public_key):
    payload = b""
    payload += len(username).to_bytes(2, "big") + username
    payload += len(password).to_bytes(2, "big") + password
    payload += len(public_key).to_bytes(2, "big") + public_key
    
    return payload

payload = encode(username, password, public_key)
# Referred to https://docs.python.org/3/library/ssl.html on creating ssl context and ssl socket
context = ssl.create_default_context()
with socket.create_connection((mdm_domain, mdm_port)) as sock:
    with context.wrap_socket(sock, server_hostname=mdm_domain) as ssock:
        ssock.sendall(payload)
        length_bytes = ssock.recv(2)
        length = int.from_bytes(length_bytes, "big")
        data = ssock.recv(length)
        print(data.decode())
