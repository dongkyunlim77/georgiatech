#!/usr/bin/env python3

import sys
import socket

MESSAGE_SIZE = 9

try:
    HOST = sys.argv[1]
    PORT = int(sys.argv[2])
except (IndexError, ValueError):
    print(f'Usage: {sys.argv[0]} HOST PORT', file=sys.stderr)
    exit(1)

def read_message(sock, length):
    buffer = bytearray()
    # referred to below link to ensure we receive requested length of message
    # https://stackoverflow.com/questions/56813704/socket-programming-python-how-to-make-sure-entire-message-is-received?utm_source=chatgpt.com
    while len(buffer) < length:
        received = sock.recv(length - len(buffer))
        if not received:
            raise ConnectionError("connection lost")
        buffer.extend(received)
    return buffer

def send_sum(sock, data):
    # Referred to https://stackoverflow.com/questions/72195552/difference-between-struct-unpack-and-int-from-bytes
    num1 = int.from_bytes(data[:4], 'big')
    num2 = int.from_bytes(data[4:], 'big')
    result = (num1 + num2).to_bytes(4, 'big')
    sock.sendall(result)

def main():
    # Referred to echo server part from https://realpython.com/python-sockets/
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as connection:
        connection.connect((HOST, PORT))

        while True:
            try:
                message = read_message(connection, MESSAGE_SIZE)
            except ConnectionError as e:
                print(e, file=sys.stderr)
                break

            mode = message[0:1].decode()
            eight_bytes = message[1:]
            
            if mode == 'Q':
                send_sum(connection, eight_bytes)
            elif mode == 'S':
                secret = eight_bytes.decode('utf-8')
                print(secret)
                break


if __name__ == '__main__':
    main()
