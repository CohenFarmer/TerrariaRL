import socket
import json
import random

def main():
    host = "127.0.0.1"
    port = 7555

    print(f"Connecting to TerrariaRL mod at {host}:{port}...")
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((host, port))
    sock_file = sock.makefile("r")
    print("Connected!\n")

    step = 0
    try:
        while True:
            line = sock_file.readline()
            if not line:
                print("Mod disconnected.")
                break

            #strip thing
            line = line.lstrip('\ufeff')
            state = json.loads(line)

            if step % 15 == 0:
                print(f"Step {step:>5d} | "
                      f"HP: {state['hp']}/{state['max_hp']} | "
                      f"Pos: ({state['px']}, {state['py']}) | "
                      f"Ground: {state['ground']} | "
                      f"NPCs: {len(state['npcs'])} | "
                      f"Reward: {state['reward']}")

            #Random actions
            move_x = random.choice([0, 1, 2])
            jump = random.choice([0, 0, 0, 1])
            hotbar = 0
            mouse_x = random.randint(-200, 200)
            mouse_y = random.randint(-200, 200)
            use_item = random.choice([0, 0, 1])

            action = f"{move_x},{jump},{hotbar},{mouse_x},{mouse_y},{use_item}\n"
            sock.sendall(action.encode("utf-8"))
            step += 1

    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        sock.close()

if __name__ == "__main__":
    main()