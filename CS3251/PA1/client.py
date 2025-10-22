import socket
import threading
import sys
import argparse
import time

def println(s: str):
    try:
        sys.stdout.write(s + "\n")
        sys.stdout.flush()
    except Exception:
        pass

def _recv_line(sock: socket.socket) -> str | None:
    buf = bytearray()
    while True:
        b = sock.recv(1)
        if not b:
            return None if not buf else buf.decode(errors="ignore")
        if b == b"\n":
            return buf.decode(errors="ignore")
        buf += b

def recv_loop(sock: socket.socket, my_username: str, prebuffer: str | None = None):
    try:
        if prebuffer and prebuffer.strip() != f"{my_username} joined the chatroom":
            println(prebuffer)
        while True:
            line = _recv_line(sock)
            if line is None:
                return
            if line.strip() == f"{my_username} joined the chatroom":
                continue
            println(line)
    except Exception:
        return

def send_loop(sock: socket.socket):
    try:
        while True:
            line = sys.stdin.readline()
            if line == "":
                continue
            line = line.rstrip("\r\n")
            sock.sendall((line + "\n").encode())
            if line == ":Exit":
                return
    except Exception:
        return

def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("-join", action="store_true")
    parser.add_argument("-host", type=str)
    parser.add_argument("-port", type=int)
    parser.add_argument("-username", type=str)
    parser.add_argument("-passcode", type=str)
    args, _ = parser.parse_known_args()

    # Strict flag check
    if not (args.join and args.host and args.port and args.username and args.passcode):
        sys.exit(1)

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.connect((args.host, args.port))

        # Send AUTH first
        s.sendall(f"AUTH {args.username} {args.passcode}\n".encode())

        # Peek ONLY for "Incorrect passcode". Preserve anything else.
        pre = None
        s.settimeout(0.35)
        try:
            first = bytearray()
            got_any = False
            while True:
                b = s.recv(1)
                if not b:
                    break
                got_any = True
                if b == b"\n":
                    break
                first += b
            if got_any:
                text = first.decode(errors="ignore").strip()
                if text == "Incorrect passcode":
                    println("Incorrect passcode")
                    s.close()
                    return
                else:
                    # Preserve the first line (likely a broadcast) to print once
                    pre = text
        except socket.timeout:
            pass
        finally:
            s.settimeout(None)

        # Now it's safe to announce connection
        println(f"Connected to {args.host} on port {args.port}")

        # Start receiver (prints any preserved line exactly once)
        t = threading.Thread(target=recv_loop, args=(s, args.username, pre), daemon=True)
        t.start()

        # Handle user input
        send_loop(s)

    except Exception:
        # Best effort: print any immediate server text
        try:
            data = s.recv(1024)
            if data:
                println(data.decode(errors="ignore").strip())
        except Exception:
            pass
    finally:
        try:
            s.close()
        except Exception:
            pass

if __name__ == "__main__":
    main()