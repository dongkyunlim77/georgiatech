# Some imports that we think may be useful to you.
import ipaddress
import socket
import ssl
import sys


def bytes_to_ip(b):
    return ipaddress.IPv4Address(b)


certificate_chain, private_key = sys.argv[1], sys.argv[2]
resolver_domain = sys.argv[3]
resolver_port = sys.argv[4]

context = ssl.create_default_context()
context.load_cert_chain(certificate_chain, private_key)
# TODO: add client certificate to context so that
#       is offered in the handshake.
#       context.load_cert_chain may be useful here.

# Referred to http://www.tcpipguide.com/free/t_DNSNameNotationandMessageCompressionTechnique.htm
# for DNS message encoding
def encode_domain(name):
    segments = name.strip(".").split(".")
    return b"".join(len(seg).to_bytes(1, "big") + seg.encode() for seg in segments) + b"\x00"

# Referred to https://www.geeksforgeeks.org/dns-message-format/ and f-society page
def decode_domain(packet, offset):
    domain = bytearray()
    while packet[offset] != 0:
        length = packet[offset]
        offset += 1
        domain += packet[offset : offset + length] + b"."
        offset += length
    return domain.decode(), offset + 1

# Referred to https://www.rfc-editor.org/rfc/rfc1035 and f-society website.
def parse_response(data, start, count):
    entries = []
    idx = start
    for _ in range(count):
        domain, idx = decode_domain(data, idx)
        type_code = int.from_bytes(data[idx:idx+2], "big")
        rdlen = int.from_bytes(data[idx+8:idx+10], "big")
        idx += 10
        rdata = data[idx:idx+rdlen]
        idx += rdlen
        if type_code == 1:
            entries.append(f"{domain} A {bytes_to_ip(rdata)}")
        elif type_code == 16:
            length = rdata[0]
            entries.append(f"{domain} TXT {rdata[1:length + 1].decode()}")
    return entries

def create_query(domain):
    # Referred to https://datatracker.ietf.org/doc/html/rfc1035#section-4.1.1 for 
    # DNS header and question format
    header = b"\xAB\xCD\x01\x00" + b"\x00\x01" + b"\x00\x00" * 3
    question = encode_domain(domain) + b"\x00\xff" + b"\x00\x01"
    full_msg = header + question
    return len(full_msg).to_bytes(2, "big") + full_msg

with socket.create_connection((resolver_domain, int(resolver_port))) as sock:
    with context.wrap_socket(sock, server_hostname=resolver_domain) as ssock:
        ssock.sendall(create_query("evil-corp.ink"))
        response = ssock.recv(4096)[2:]
        answer_count = int.from_bytes(response[6:8], "big")
        idx = 12
        _, idx = decode_domain(response, idx)
        idx += 4

        for answer in parse_response(response, idx, answer_count):
            print(answer)