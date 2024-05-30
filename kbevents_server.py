
import socket
import pyautogui
import time

PORT = 5000


def handle_command(command):
    """
        Trigger keyboard event with support with combined commands like: "shift+1"
    """
    if '+' in command:
        keys = command.split('+')

        # shift, ctrl, command key use case
        if keys[0] in ["shift", "ctrl", command]:
            with pyautogui.hold(keys[0]):
                pyautogui.press(keys[1])
             
        else:
        # use case of multiple keys activating (like you do for moving camera view)        
            with pyautogui.hold(keys[0]):
                time.sleep(0.2*len(keys))      
    else:
        pyautogui.press(command)  # Press a single key if no combination


server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.bind(('0.0.0.0', PORT))  # Listen on all network interfaces
server.listen(5)
print("Server is listening...")

while True:
    client_socket, addr = server.accept()
    print(f"Connection from {addr}")
    try:
        while True:
            command = client_socket.recv(1024).decode()
            if not command:
                break
            handle_command(command)  # Process the command
            print(f"Processed command: {command}")
            client_socket.send(f"Processed command: {command}".encode())
    finally:
        client_socket.close()