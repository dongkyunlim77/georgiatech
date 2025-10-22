#!/usr/bin/python3

# Run me like this:
# $ python3 padding_oracle.py "https://cryptoproject.gtinfosec.org/GTusername/paddingoracle/verify" "5a7793d3..."
# or select "Padding Oracle" from the VS Code debugger

import json
import sys
import time
from typing import Union, Dict, List

import requests

# Create one session for each oracle request to share. This allows the
# underlying connection to be re-used, which speeds up subsequent requests!
s = requests.session()


def oracle(url: str, messages: List[bytes]) -> List[Dict[str, str]]:
    while True:
        try:
            r = s.post(url, data={"message": [m.hex() for m in messages]})
            r.raise_for_status()
            return r.json()
        # Under heavy server load, your request might time out. If this happens,
        # the function will automatically retry in 10 seconds for you.
        except requests.exceptions.RequestException as e:
            sys.stderr.write(str(e))
            sys.stderr.write("\nRetrying in 10 seconds...\n")
            time.sleep(10)
            continue
        except json.JSONDecodeError as e:
            sys.stderr.write("It's possible that the oracle server is overloaded right now, or that provided URL is wrong.\n")
            sys.stderr.write("If this keeps happening, check the URL. Perhaps your GTusername is not set.\n")
            sys.stderr.write("Retrying in 10 seconds...\n\n")
            time.sleep(10)
            continue


def main():
    if len(sys.argv) != 3:
        print(f"usage: {sys.argv[0]} ORACLE_URL CIPHERTEXT_HEX", file=sys.stderr)
        sys.exit(-1)
    oracle_url, message = sys.argv[1], bytes.fromhex(sys.argv[2])

    if oracle(oracle_url, [message])[0]["status"] != "valid":
        print("Message invalid", file=sys.stderr)
    
    BLOCK_SIZE = 16
    candidates = []
    message_bytearray = bytearray(message)
    num_blocks = len(message_bytearray) // BLOCK_SIZE
    n= num_blocks - 2
    intermediate = [0] * BLOCK_SIZE
    temp = bytearray(message_bytearray)

    for i in range(256):
        temp[(n + 1) * BLOCK_SIZE - 1] = i
        status = oracle(oracle_url, [temp])[0]["status"]

        if status in ["valid", "invalid_mac"]:
            current = message_bytearray[(n + 1) * BLOCK_SIZE - 1]
            candidate_plain = 1 ^ current ^ i

            if candidate_plain == 1:
                penultimate_idx = (n + 1) * BLOCK_SIZE - 2
                backup_val = temp[penultimate_idx]
                found_bad_padding = False
                for toggle_candidate in range(1, 256):
                    temp[penultimate_idx] ^= toggle_candidate
                    verify_status = oracle(oracle_url, [temp])[0]["status"]
                    temp[penultimate_idx] = backup_val
                    if verify_status not in ["valid", "invalid_mac"]:
                        found_bad_padding = True
                        break
                if found_bad_padding:
                    candidates.append({"plain": candidate_plain, "status": status})
                else:
                    continue
            else:
                candidates.append({"plain": candidate_plain, "status": status})
        
    if len(candidates) == 2 and candidates[0]["status"] != "invalid_mac":
        plain = candidates[1]["plain"]
    else:
        plain = candidates[0]["plain"]
    current = message_bytearray[(n + 1) * BLOCK_SIZE - 1]
    
    padding = plain
    intermediate[BLOCK_SIZE - 1] = plain ^ current

    decoded = (plain.to_bytes((plain.bit_length() + 7) // 8, "big")).hex()
    temp = bytearray(message_bytearray)
    

    # Step 2:
    for i in range(BLOCK_SIZE - 2, -1, -1):
        for j in range(BLOCK_SIZE - 1, i, -1):
            temp[n * BLOCK_SIZE + j] = (BLOCK_SIZE - i) ^ intermediate[j]
        for j in range(256):
            temp[n * BLOCK_SIZE + i] = j
            status = oracle(oracle_url, [temp])[0]["status"]

            if status in ["valid", "invalid_mac"]:
                padding_size = BLOCK_SIZE - padding

                if (status == "valid" and i > padding_size) or (status == "invalid_mac" and i == padding_size):
                    continue

                current = message_bytearray[n * BLOCK_SIZE + i]
                plain = (BLOCK_SIZE - i) ^ current ^ j
                intermediate[i] = plain ^ current
                decoded = (plain.to_bytes((plain.bit_length() + 7) // 8, "big")).hex() + decoded

                break
        temp = bytearray(message_bytearray)
        
    message_bytearray = message_bytearray[0 : BLOCK_SIZE * (num_blocks - 1)]
    temp = bytearray(message_bytearray)

    # Step 3:
    for i in range(num_blocks - 3, -1, -1):
        for j in range(BLOCK_SIZE - 1, -1, -1):
            for k in range(BLOCK_SIZE - 1, j, -1):
                temp[i * BLOCK_SIZE + k] = (BLOCK_SIZE - j) ^ intermediate[k]
            for k in range(256):
                temp[i * BLOCK_SIZE + j] = k
                status = oracle(oracle_url, [temp])[0]["status"]

                if status in ["valid", "invalid_mac"]:
                    current = message_bytearray[i * BLOCK_SIZE + j]
                    plain = (BLOCK_SIZE - j) ^ current ^ k
                    intermediate[j] = plain ^ current
                    decoded = (plain.to_bytes((plain.bit_length()+ 7) // 8, "big")).hex() + decoded
                    break
            temp = bytearray(message_bytearray)
        message_bytearray = message_bytearray[: BLOCK_SIZE * (i + 1)]
        temp = bytearray(message_bytearray)
    
    #Step 4:
    decrypted = bytes.fromhex(decoded)
    decrypted = decrypted[:len(decrypted) - padding - 32]

    decrypted_text = decrypted.decode("ascii", errors="ignore")
    print(decrypted_text)
  


if __name__ == '__main__':
    main()
