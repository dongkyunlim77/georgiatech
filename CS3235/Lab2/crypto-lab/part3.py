from Crypto.Cipher import AES


class secret:
    def __repr__(self) -> str:
        encrypted = 'd0326fb8d0bfb1c5394991e7dcdf59b37889f2256772b8daad85dbeaff4483ed64cbb2b933d797bd530c8c92f57466d19033fe111d8ff904f1bc3d17d1fbfdd4'
        return AES.new(b'\x00'*16, AES.MODE_ECB).decrypt(bytes.fromhex(encrypted)).decode()

    def __str__(self) -> str:
        return 'secret'


s = secret()

# Put the secret string you found in the debugger here!
secret_string = 'GT is cool'