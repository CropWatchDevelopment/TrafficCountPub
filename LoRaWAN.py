import serial
import time

def read_response(serial_port):
    response = serial_port.read_all().decode('utf-8')
    return response

with serial.Serial('/dev/ttyUSB0', 115200, timeout=1) as ser:
    print(f'Opened serial port: {ser.name}')

    commands = [
        b'AT\r\n',
        b'AT\r\n',
        b'AT\r\n',
        b'AT+SEND=12:112233\r\n'
    ]

    for command in commands:
        ser.write(command)
        time.sleep(1)  # Sleep for 1 second to allow command processing
        response = read_response(ser)
        print(f'Response: {response}')
        
    ser.close()
    print('Serial port closed')
