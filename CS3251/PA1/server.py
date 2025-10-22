import socket
import threading
import sys
import argparse
from datetime import datetime, timedelta

HOST = "127.0.0.1"

def println(s: str):
    try:
        sys.stdout.write(s + "\n")
        sys.stdout.flush()
    except BrokenPipeError:
        pass
    except Exception:
        pass

def valid_username(u: str) -> bool:
    return 0 < len(u) <= 8

def valid_passcode(p: str) -> bool:
    return len(p) <= 5 and p.isalnum()

class ChatServer:
    def __init__(self, port: int, passcode: str):
        self.port = port
        self.passcode = passcode
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind((HOST, port))
        self.server_socket.listen(5)
        self.lock = threading.Lock()
        self.clients = {}
        self.user_to_socket = {}
    
    def start(self):
        println(f"Server started on port {self.port}. Accepting connections")
        try:
            while True:
                connection, addr = self.server_socket.accept()
                threading.Thread(target=self.handle_client, args=(connection, addr), daemon=True).start()
        except KeyboardInterrupt:
            pass
        finally:
            self.server_socket.close()
    
    def broadcast(self, msg: str, exclude_sock=None, include_sender=False):
        with self.lock:
            targets = [s for s in self.clients.keys() if include_sender or s is not exclude_sock]
        for s in targets:
            try:
                s.sendall((msg + "\n").encode())
            except Exception:
                self.cleanup_socket(s)
    
    def send_to_user(self, username: str, msg: str):
        with self.lock:
            s = self.user_to_socket.get(username)
        if not s:
            return False
        try:
            s.sendall((msg + "\n").encode())
            return True
        except Exception:
            self.cleanup_socket(s)
            return False
    
    def list_active_users(self):
        with self.lock:
            return list(self.user_to_socket.keys())
    
    def cleanup_socket(self, s: socket.socket):
        with self.lock:
            username = self.clients.pop(s, None)
            if username:
                self.user_to_socket.pop(username, None)
        try:
            s.close()
        except Exception:
            pass
        if username:
            println(f"{username} left the chatroom")
            self.broadcast(f"{username} left the chatroom")
    
    def handle_client(self, connection: socket.socket, addr):
        try:
            connection.settimeout(60.0)
            auth_line = self._recv_line(connection)
        except Exception:
            connection.close()
            return
        
        ok, username_or_msg = self._process_auth_line(auth_line)
        if not ok:
            connection.sendall(b"Incorrect passcode\n")
            connection.close()
            return
        
        username = username_or_msg
        with self.lock:
            self.clients[connection] = username
            self.user_to_socket[username] = connection
        
        println(f"{username} joined the chatroom")
        self.broadcast(f"{username} joined the chatroom")

        try:
            connection.settimeout(None)
            while True:
                line = self._recv_line(connection)
                if line is None:
                    break
                line = line.rstrip("\r\n")
                
                if line == ":Exit":
                    self.cleanup_socket(connection)
                    return
                
                if line == ":)":
                    msg = f"{username}: [feeling happy]"
                    println(msg)
                    self.broadcast(msg, exclude_sock=connection)
                    continue

                if line == ":(":
                    msg = f"{username}: [feeling sad]"
                    println(msg)
                    self.broadcast(msg, exclude_sock=connection)
                    continue

                if line == ":mytime":
                    now_str = datetime.now().strftime("%a %b %d %H:%M:%S %Y")
                    msg = f"{username}: {now_str}"
                    println(msg)
                    self.broadcast(msg, include_sender=True)
                    continue

                if line == ":+1hr":
                    plus_one = (datetime.now() + timedelta(hours=1)).strftime("%a %b %d %H:%M:%S %Y")
                    msg = f"{username}: {plus_one}"
                    println(msg)
                    self.broadcast(msg, include_sender=True)
                    continue

                if line == ":Users":
                    users = ", ".join(self.list_active_users())
                    connection.sendall((f"Active Users: {users}\n").encode())
                    println(f"{username}: searched up active users")
                    continue

                if line.startswith(":Msg "):
                    parts = line.split(maxsplit=2)
                    if len(parts) >= 3:
                        target, message = parts[1], parts[2]
                        self.send_to_user(target, f"[Message from {username}]: {message}")
                        println(f"{username}: send message to {target}")
                        continue
                
                if len(line) > 100:
                    line = line[:100]
                msg = f"{username}: {line}"
                println(msg)
                self.broadcast(msg, exclude_sock=connection)
        except Exception:
            pass
        finally:
            self.cleanup_socket(connection)
    
    def _recv_line(self, connection: socket.socket):
        data = []
        while True:
            chunk = connection.recv(1)
            if not chunk:
                return None if not data else "".join(data)
            ch = chunk.decode(errors="ignore")
            if ch == "\n":
                return "".join(data)
            data.append(ch)
    
    def _process_auth_line(self, line: str):
        if not line:
            return (False, "")
        parts = line.strip().split()

        if len(parts) != 3 or parts[0] != "AUTH":
            return (False, "")
        _, username, passcode = parts

        if not valid_username(username) or not valid_passcode(passcode):
            return (False, "")
        
        if passcode != self.passcode:
            return (False, "")
        
        return (True, username)
    
def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("-start", action="store_true")
    parser.add_argument("-port", type=int)
    parser.add_argument("-passcode", type=str)
    args, _ = parser.parse_known_args()

    if not args.start or args.port is None or args.passcode is None:
        sys.exit(1)
    
    server = ChatServer(args.port, args.passcode)
    server.start()
    
if __name__ == "__main__":
    main()