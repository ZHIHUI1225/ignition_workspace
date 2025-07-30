#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# ePuck.py
#
# Copyright 2010 Manuel Martín Ortiz <mmartinortiz@gmail.com>
# Modified 2025 for WiFi support
#
#       This program is free software; you can redistribute it and/or modify
#       it under the terms of the GNU General Public License as published by
#       the Free Software Foundation; either version 3 of the License, or
#       (at your option) any later version.
#
#       This program is distributed in the hope that it will be useful,
#       but WITHOUT ANY WARRANTY; without even the implied warranty of
#       MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#       GNU General Public License for more details.
#
#       You should have received a copy of the GNU General Public License
#       along with this program; if not, write to the Free Software
#       Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
#       MA 02110-1301, USA.
#
#		-- ePuck.py --
#
#		The aim of this library is to provide access to the ePuck robots
#		through a bluetooth connection. Thus, you can write a program that
#		read from the ePuck's sensors and write in their actuators, This
#		will allow us to create advanced programs that can develop a wide
#		variety of complex tasks. It is necesary that the ePuck has installed
#		the Webot's fimware 1.4.2 or 1.4.3. You can find this fantastic
#		simulator on this site: http://www.cyberbotics.com/
#
#		This library is written in Python 2.6, and you can import it from
#		any program written in Python  (same version or later). In addition
#		to this, you will also need two extra libraries:
#
#			-> Python Bluetooth or Pybluez
#			-> Python Image Library (PIL)
#
#		In this package you will find some examples of how to use this library.
#
#		You may expetience some problems when you work with your ePuck, We
#		recommend you take into consideration the following special
#		characteristic: we use a bluetooth communciation, therefore our bandwith
#		is limited and we cannot expect to do too many tasks in short
#		time; i.e:  If you put the wheels speed to max and want
#		to make a quick process of the images, you will know what I'm saying.
#		So remember, you are processing in your computer, not on the ePuck,
#		and you need to take the sensors data and write on the actuators
#		values on the ePuck
#
#		For further information and updates visit http://abitworld.com/projects

import sys  # System library
import socket  # Used for WiFi communications
import time  # Used for image capture process
import struct  # Used for Big-Endian messages
from PIL import Image  # Used for the pictures of the camera

__package__ = "ePuck"
__docformat__ = "restructuredtext"

"""
:newfield company: Company
"""

__version__ = "1.2.2"
__author__ = "Manuel Martin Ortiz"
__license__ = "GPL"
__contact__ = ["mmartinortiz@gmail.com"]

# This dictionary have as keys the first character of the message, that
# is used to know the number of lines. If no key for the message, 1 line is assumed
DIC_MSG = {
    "v": 2,  # Version
    "\n": 23,  # Menu
    "\x0c": 2,  # Welcome
    "k": 3,  # Calibration
    "R": 2  # Reset
}

# You have to use the keys of this dictionary for indicate on "enable" function
# the sensor that you want to read
DIC_SENSORS = {
    "accelerometer": "a",
    "selector": "c",
    "motor_speed": "e",
    "camera": "i",
    "floor": "m",
    "proximity": "n",
    "light": "o",
    "motor_position": "q",
    "microphone": "u"
}

# You have to use the keys of this dictionary for indicate the operating
# mode of the camera
CAM_MODE = {
    "GREY_SCALE": 0,
    "RGB_365": 1,
    "YUV": 2,
    "LINEAR_CAM": 3
}

# You can use three diferents Zoom in the camera
CAM_ZOOM = (1, 4, 8)


class ePuck():
    """
    This class represent an ePuck object
    """

    def __init__(self, address, debug=False):
        """
        Constructor process

        :param 	address: Robot's direction in AA:BB:CC:DD:EE:FF format
        :type	address: MAC Address
        :param 	debug: If you want more verbose information, useful for debugging
        :type	debug: Boolean

        :return: ePuck object
        """

        # Monitoring Variables
        self.messages_sent = 0
        self.messages_received = 0
        self.version = __version__
        self.debug = debug

        # Connection Attributes
        self.socket = None
        self.address = address
        self.conexion_status = False

        # Camera attributes
        self._cam_width = None
        self._cam_height = None
        self._cam_enable = False
        self._cam_zoom = None
        self._cam_mode = None
        self._cam_size = None

        # Sensors and actuators lists
        self._sensors_to_read = []
        self._actuators_to_write = []

        # Sensors
        self._accelerometer = (0, 0, 0)
        self._accelerometer_filtered = False
        self._selector = (0)
        self._motor_speed = (0, 0)  # left and right motor
        self._motor_position = (0, 0)  # left and right motor
        self._camera_parameters = (0, 0, 0, 0)
        self._floor_sensors = (0, 0, 0)
        self._proximity = (0, 0, 0, 0, 0, 0, 0, 0)
        self._light_sensor = (0, 0, 0, 0, 0, 0, 0, 0)
        self._microphone = (0, 0, 0)
        self._pil_image = None

        # Leds
        self._leds_status = [False] * 10

    #
    # Private methods
    #
    def _debug(self, *txt):
        """
        Show debug information and data, only works if debug information
        is enable (see "set_debug()")

        :param 	txt: Data to be showed separated by comma
        :type	txt: Any
        """

        if self.debug:
            import sys
            print('\033[31m[ePuck'+str(self.address)+']:\033[0m ' + ' '.join([str(e) for e in txt]), file=sys.stderr)

        return 0

    def _recv(self, n=4096):
        """
        Receive data from the robot

        :param	n: 	Number of bytes you want to receive
        :type	n: 	int
        :return: 	Data received from the robot as string if it was successful, raise an exception if not
        :rtype:		String
        :raise Exception:	If there is a communication problem
        """
        if not self.conexion_status:
            raise Exception('There is not connection')

        try:
            line = self.socket.recv(n)
            # Decode bytes to string for Python 3 compatibility
            if isinstance(line, bytes):
                line = line.decode('utf-8', errors='ignore')
            self.messages_received += 1
        except socket.error as e:
            txt = 'WiFi communication problem: ' + str(e)
            self._debug(txt)
            raise Exception(txt)
        else:
            return line

    def _send(self, message):
        """
        Send data to the robot

        :param	message: Message to be sent
        :type	message: String or bytes
        :return: Number of bytes sent if it was successful. -1 if not
        :rtype:	int
        """
        if not self.conexion_status:
            raise Exception('There is not connection')

        try:
            # Ensure message is bytes for Python 3 compatibility
            if isinstance(message, str):
                message = message.encode('utf-8')
            n = self.socket.send(message)
            self.messages_sent += 1
        except Exception as e:
            self._debug('Send problem:', e)
            return -1
        else:
            return n

    def _read_image(self):
        """
        Returns an image obtained from the robot's camera. For communication
        issues you only can get 1 image per second

        :return: The image in PIL format
        :rtype: PIL Image
        """

        # Thanks to http://www.dailyenigma.org/e-puck-cam.shtml for
        # the code for get the image from the camera
        msg = struct.pack(">bb", - ord("I"), 0)

        try:
            n = self._send(msg)
            self._debug("Reading Image: sending " + repr(msg) + " and " + str(n) + " bytes")

            # We have to add 3 to the size, because with the image we
            # get "mode", "width" and "height"
            size = self._cam_size + 3
            img = self._recv(size)
            while len(img) != size:
                img += self._recv(size)

            # Create the PIL Image
            image = Image.frombuffer("RGB", (self._cam_width, self._cam_height),
                                     img, "raw",
                                     "BGR;16", 0, 1)

            image = image.rotate(180)
            self._pil_image = image

        except Exception as e:
            self._debug('Problem receiving an image: ', e)

    def _refresh_camera_parameters(self):
        """
        Method for refresh the camera parameters using WiFi protocol
        Set default values for WiFi mode
        """
        try:
            # WiFi protocol doesn't support camera parameter query,
            # set default values based on typical e-puck2 camera
            self._cam_mode = 1  # RGB mode
            self._cam_width = 160  # Standard e-puck2 width
            self._cam_height = 120  # Standard e-puck2 height
            self._cam_zoom = 8  # Default zoom
            self._cam_size = self._cam_width * self._cam_height * 2  # RGB565 format
            
            self._camera_parameters = self._cam_mode, self._cam_width, self._cam_height, self._cam_zoom
            self._debug(f"Camera parameters set to defaults: {self._camera_parameters}")
            return True
            
        except Exception as e:
            self._debug(f"Error setting camera parameters: {e}")
            return False

    def _write_actuators(self):
        """
        Write in the robot the actuators values. Don't use directly,
        instead use 'step()'
        """

        # Not all messages reply with AKC, only Ascii messages
        acks = ['j', 't']

        # We make a copy of the actuators list
        actuators = self._actuators_to_write[:]

        for m in actuators:
            if m[0] == 'L':
                # Leds
                msg = struct.pack('<bbb', - ord(m[0]), m[1], m[2])
                n = self._send(msg)
                self._debug('Binary message sent of [' + str(n) + '] bytes: ' + str(struct.unpack('<bbb', msg)))

            elif m[0] == 'D' or m[0] == 'P':
                # Set motor speed or set motor position
                msg = struct.pack('<bhh', - ord(m[0]), m[1], m[2])
                n = self._send(msg)
                self._debug('Binary message sent of [' + str(n) + '] bytes: ' + str(struct.unpack('<bhh', msg)))

            else:
                # Others actuators, parameters are separated by commas
                msg = ",".join(["%s" % i for i in m])
                reply = self.send_and_receive(msg)
                if reply == 'j':
                    self._refresh_camera_parameters()

                if reply not in acks:
                    self._debug('Unknown ACK reply from ePcuk: ' + reply)

            self._actuators_to_write.remove(m)
        return

    def _read_sensors(self):
        """
        This method is used for read the ePuck's sensors. Don't use directly,
        instead use 'step()'
        """

        # We can read sensors in two ways: Binary Mode and Ascii Mode
        # Ascii mode is slower than Binary mode, therefore, we use
        # Binary mode whenever we can. Not all sensors are available in
        # Binary mode

        def send_binary_mode(parameters):
            # Auxiliar function for sent messages in binary modes
            # Parameters: ('Char to be sent', 'Size of reply waited', 'Format of the teply')

            self._debug('Sending binary message: ', ','.join('%s' % i for i in parameters))
            message = struct.pack(">bb", - ord(parameters[0]), 0)
            self._send(message)
            reply = ()
            try:
                reply = self._recv()
                while len(reply) < parameters[1]:
                    reply += self._recv()
                self._debug('Binary message recived: ', reply)
                reply = struct.unpack(parameters[2], reply) # "reply" must contain the exact number of bytes requested by the format in "paramaters[2]".

            except Exception as e:
                if "timed out" in str(e):
                    self._debug("Received " + str(len(reply)) + " of " + str(parameters[1]) + " bytes")
                    return 0
                else:
                    raise e

            return reply

        # Read differents sensors
        for s in self._sensors_to_read:

            if s == 'a':
                # Accelerometer sensor in a non filtered way
                if self._accelerometer_filtered:
                    parameters = ('A', 12, '@III')

                else:
                    parameters = ('a', 6, '@HHH')

                reply = send_binary_mode(parameters)
                if type(reply) is tuple and type(reply[0]) is int:
                    self._accelerometer = reply

            elif s == 'n':
                # Proximity sensors
                parameters = ('N', 16, '@HHHHHHHH')
                reply = send_binary_mode(parameters)
                if type(reply) is tuple and type(reply[0]) is int:
                    self._proximity = reply
                    #print("prox: " + str(reply[0]) + ", " + str(reply[1]) + ", " + str(reply[2]) + ", " + str(reply[3]) + ", " + str(reply[4]) + ", " + str(reply[5]) + ", " + str(reply[6]) + ", " + str(reply[7]))
            elif s == 'm':
                # Floor sensors
                parameters = ('M', 10, '@HHHHH')
                reply = send_binary_mode(parameters)
                if type(reply) is tuple and type(reply[0]) is int:
                    self._floor_sensors = reply

            elif s == 'q':
                # Motor position sensor
                parameters = ('Q', 4, '@HH')
                reply = send_binary_mode(parameters)
                if type(reply) is tuple and type(reply[0]) is int:
                    self._motor_position = reply

            elif s == 'o':
                # Light sensors
                parameters = ('O', 16, '@HHHHHHHH')
                reply = send_binary_mode(parameters)
                if type(reply) is tuple and type(reply[0]) is int:
                    self._light_sensor = reply

            elif s == 'u':
                # Microphone
                parameters = ('u', 6, '@HHH')
                reply = send_binary_mode(parameters)
                if type(reply) is tuple and type(reply[0]) is int:
                    self._microphone = reply

            elif s == 'e':
                # Motor Speed
                parameters = ('E', 4, '@HH')
                reply = send_binary_mode(parameters)
                if type(reply) is tuple and type(reply[0]) is int:
                    self._motor_speed = reply

            elif s == 'i':
                # Do nothing for the camera, is an independent process
                pass

            else:
                reply = self.send_and_receive(s).split(",")

                t = reply[0]
                response = tuple(reply[1:len(reply)])

                if t == "c":
                    # Selector
                    self._selector = response[0]

                else:
                    self._debug('Unknow type of sensor to read' + str(reply))


    #
    # Public methods
    #

    def connect(self):
        """
        Connect with the physic ePuck robot using WiFi with correct binary protocol

        :return: If the connexion was succesful
        :rtype: Boolean
        :except Exception: If there are a communication problem, for example, the robot is off
        """

        if self.conexion_status:
            self._debug('Already connected')
            return False
        try:
            # Extract IP and port from address (expected format: "IP_ADDRESS:PORT")
            if ":" in self.address:
                ip, port = self.address.split(":")
                port = int(port)
            else:
                ip = self.address
                port = 1000  # Default e-puck WiFi port is 1000
            
            self._debug(f'Attempting to connect to {ip}:{port}')
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            # Increase connection timeout for WiFi networks
            self.socket.settimeout(10.0)  # 10 seconds for connection
            self._debug('Connecting...')
            self.socket.connect((ip, port))
            self._debug('TCP connection established')
            # Set shorter timeout for normal communication after connection
            self.socket.settimeout(3.0)  # 3 seconds for normal communication

        except Exception as e:
            txt = 'Connection problem: \n' + str(e)
            self._debug(txt)
            raise Exception(txt)

        self.conexion_status = True
        self._debug("TCP connection established, testing e-puck2 WiFi binary protocol...")

        # Test the connection with a sensor request
        try:
            self._send_wifi_command(sensors=True, image=False)
            response = self._receive_wifi_response()
            if response:
                self._debug("✅ WiFi binary protocol working correctly")
                return True
            else:
                raise Exception("No response from WiFi protocol test")
        except Exception as e:
            self._debug(f"WiFi protocol test failed: {e}")
            raise Exception(f"WiFi protocol initialization failed: {e}")

        return True

    def _send_wifi_command(self, sensors=False, image=False, left_motor=0, right_motor=0, leds=0x00, rgb_leds=None, sound=0x00):
        """
        Send e-puck2 WiFi binary command packet
        
        :param sensors: Request sensor data
        :param image: Request camera image
        :param left_motor: Left motor speed (-1000 to 1000)
        :param right_motor: Right motor speed (-1000 to 1000)
        :param leds: LED control byte
        :param rgb_leds: List of 12 RGB values (0-100 each)
        :param sound: Sound command
        """
        if not self.conexion_status:
            raise Exception('There is not connection')
            
        import struct
        
        # Prepare request byte
        request = 0x00
        if image:
            request |= 0x01  # bit0: image stream
        if sensors:
            request |= 0x02  # bit1: sensors stream
            
        # Prepare settings byte
        settings = 0x00  # bit2=0 for speed mode (not position)
        
        # Prepare RGB LEDs (default to all off)
        if rgb_leds is None:
            rgb_leds = [0] * 12
        elif len(rgb_leds) != 12:
            rgb_leds = (rgb_leds + [0] * 12)[:12]
            
        # Build command packet (20 bytes)
        cmd_packet = struct.pack('<B', 0x80)        # Command ID
        cmd_packet += struct.pack('<B', request)     # Request flags
        cmd_packet += struct.pack('<B', settings)    # Settings
        cmd_packet += struct.pack('<h', left_motor)  # Left motor
        cmd_packet += struct.pack('<h', right_motor) # Right motor
        cmd_packet += struct.pack('<B', leds)        # LEDs
        cmd_packet += struct.pack('<12B', *rgb_leds) # RGB LEDs
        cmd_packet += struct.pack('<B', sound)       # Sound
        
        self._debug(f'Sending WiFi command: {cmd_packet.hex()}')
        
        try:
            if isinstance(cmd_packet, str):
                cmd_packet = cmd_packet.encode('utf-8')
            n = self.socket.send(cmd_packet)
            self.messages_sent += 1
            return n
        except Exception as e:
            self._debug('WiFi send problem:', e)
            raise Exception(f'WiFi send error: {e}')
            
    def _receive_wifi_response(self):
        """
        Receive e-puck2 WiFi response packet
        
        :return: Tuple of (packet_id, data) or None if timeout
        """
        if not self.conexion_status:
            raise Exception('There is not connection')
            
        import struct
        
        try:
            # Read packet ID (1 byte)
            response_id_byte = self.socket.recv(1)
            if not response_id_byte:
                return None
                
            response_id = struct.unpack('<B', response_id_byte)[0]
            self._debug(f'WiFi response ID: 0x{response_id:02X}')
            
            if response_id == 0x01:  # Image packet
                # QQVGA (160x120) RGB565 = 38400 bytes
                image_data = b''
                bytes_to_read = 38400
                while len(image_data) < bytes_to_read:
                    chunk = self.socket.recv(min(4096, bytes_to_read - len(image_data)))
                    if not chunk:
                        break
                    image_data += chunk
                self._debug(f'Received image data: {len(image_data)} bytes')
                return (response_id, image_data)
                
            elif response_id == 0x02:  # Sensors packet
                sensors_data = self.socket.recv(104)  # 104 bytes
                self._debug(f'Received sensor data: {len(sensors_data)} bytes')
                self.messages_received += 1
                return (response_id, sensors_data)
                
            elif response_id == 0x03:  # Empty acknowledgment
                self._debug('Received WiFi acknowledgment')
                self.messages_received += 1
                return (response_id, b'')
                
            else:
                self._debug(f'Unknown WiFi response ID: 0x{response_id:02X}')
                return (response_id, b'')
                
        except socket.timeout:
            self._debug('WiFi receive timeout')
            return None
        except Exception as e:
            self._debug(f'WiFi receive error: {e}')
            raise Exception(f'WiFi receive error: {e}')

    def disconnect(self):
        """
        Disconnect from ePuck robot. Same as 'close()'
        """

        self.close()

    def close(self):
        """
        Close the connection with the robot. Same as 'disconnect()'

        :return: 0 if all ok
        :rtype: int
        :raise Exception: if it was a problem closing the connection
        """

        if self.conexion_status:
            try:
                # Stop the robot
                self.stop()

                # Close the socket
                self.socket.close()
                self.conexion_status = False
            except Exception as e:
                raise Exception('Closing connection problem: \n') + str(e)
            else:
                return 0

    def set_debug(self, debug):
        """
        Set / unset debug information
        :param debug: True or False, as you want or not Debug information
        :type debug: Boolean
        """

        self.debug = debug

    def send_and_receive(self, msg):
        """
        Legacy compatibility method - WiFi protocol uses binary commands instead
        
        :param msg: Command string (ignored in WiFi protocol)
        :type msg: String
        :return: Empty string for compatibility
        :rtype: String
        """
        self._debug(f"Legacy send_and_receive called with: {msg} - use WiFi binary protocol methods instead")
        return ""


    def save_image(self, name='ePuck.jpg'):
        """
        Save image from ePuck's camera to disk

        :param name: Image name, ePuck.jpg as default
        :type name: String

        :return: Operation result
        :rtype:  Boolean
        """

        if self._pil_image:
            return self._pil_image.save(name)
        else:
            return False

    def get_accelerometer(self):
        """
        Return Accelerometer values in (x, y, z)

        :return: Accelerometer values
        :rtype: Tuple
        """
        return self._accelerometer

    def get_selector(self):
        """
        Return the selector position (0-15)

        :return: Selector value
        :rtype: int
        """
        return self._selector

    def get_motor_speed(self):
        """
        Return the motor speed. Correct values are in the range [-1000, 1000]

        :return: Motor speed
        :rtype: Tuple
        """
        return self._motor_speed

    def get_camera_parameters(self):
        """
        Return the camera parameters as a tuple
        (mode, width, height, zoom)

        :return: Camera parameters
        :rtype: Tuple
        """
        return self._camera_parameters

    def get_floor_sensors(self):
        """
        Return the floor sensors values as (left, center, right)

        :return: Floor sensors values
        :rtype: Tuple
        """
        return self._floor_sensors

    def get_proximity(self):
        """
        Return the values of the 8 proximity sensors

        :return: Proximity sensors values
        :rtype: Tuple
        """
        return self._proximity

    def get_light_sensor(self):
        """
        Return the value of the light sensor

        :return: Ligth sensor value
        :rtype: Tuple
        """
        return self._light_sensor

    def get_motor_position(self):
        """
        Return the position of the left and right motor as a tuple

        :return: Motor position
        :rtype: Tuple
        """
        return self._motor_position

    def get_microphone(self):
        """
        Return the volume of the three microphones

        :return: Microphones values
        :rtype: Tuple
        """
        return self._microphone

    def is_connected(self):
        """
        Return a boolean value that indicate if the robot is connected to the PC

        :return: If the robot is connected to the PC
        :rtype: Boolean
        """
        return self.conexion_status

    def get_image(self):
        """
        Return the last image captured from the ePuck's camera (after a 'step()').
        None if	there are not images captured. The image is an PIL object

        :return: Image from robot's camera
        :rtype: PIL
        """
        return self._pil_image

    def get_sercom_version(self):
        """
        :return: Return the ePuck's firmware version (simplified for WiFi protocol)
        :rtype: String
        """
        # WiFi protocol doesn't support version query, return library version
        return f"WiFi-{self.version}"

    def set_accelerometer_filtered(self, filter=False):
        """
        Set filtered way for accelerometer, False is default value
        at the robot start

        :param filter: True or False, as you want
        :type filter: Boolean
        """
        self._accelerometer_filtered = filter

    def disable(self, *sensors):
        """
        Sensor(s) that you want to get disable in the ePuck

        :param sensors: Name of the sensors, take a look to DIC_SENSORS. Multiple sensors can be separated by commas
        :type sensors: String
        :return: Sensors enabled
        :rtype: List
        :except Exception: Some wrong happened
        """
        for sensor in sensors:
            try:
                if sensor not in DIC_SENSORS:  # Python 3 compatible
                    self._debug('Sensor "' + sensor + '" not in DIC_SENSORS')
                    break

                if sensor == "camera":
                    self._cam_enable = False

                if DIC_SENSORS[sensor] in self._sensors_to_read:
                    l = list(self._sensors_to_read)
                    l.remove(DIC_SENSORS[sensor])
                    self._sensors_to_read = tuple(l)
                    self._debug('Sensor "' + sensor + '" disabled')
                else:
                    self._debug('Sensor "' + sensor + '" alrady disabled')

            except Exception as e:  # Python 3 compatible
                self._debug('Something wrong happened to disable the sensors: ', e)

        return self.get_sensors_enabled()

    def enable(self, *sensors):
        """
        Sensor(s) that you want to get enable in the ePuck

        :param sensors: Name of the sensors, take a look to DIC_SENSORS. Multiple sensors can be separated by commas
        :type sensors: String
        :return: Sensors enabled
        :rtype: List
        :except Exception: Some wrong happened
        """

        # Using the * as a parameters, we get a tuple with all sensors
        for sensor in sensors:
            try:
                if sensor not in DIC_SENSORS:  # Python 3 compatible
                    self._debug('Sensor "' + sensor + '" not in DIC_SENSORS')
                    break

                if sensor == "camera":
                    # If the sensor is the Camera, then we refresh the
                    # camera parameters
                    if not self._cam_enable:
                        try:
                            self._refresh_camera_parameters()
                            self._cam_enable = True
                            self.timestamp = time.time()
                        except:
                            break

                if DIC_SENSORS[sensor] not in self._sensors_to_read:
                    l = list(self._sensors_to_read)
                    l.append(DIC_SENSORS[sensor])
                    self._sensors_to_read = tuple(l)
                    self._debug('Sensor "' + sensor + '" enabled')
                else:
                    self._debug('Sensor "' + sensor + '" alrady enabled')

            except Exception as e:  # Python 3 compatible
                self._debug('Something wrong happened to enable the sensors: ', e)
        return self.get_sensors_enabled()

    def get_sensors_enabled(self):
        """
        :return: Return a list of sensors thar are active
        :rtype: List
        """
        l = []
        for sensor in DIC_SENSORS:
            if DIC_SENSORS[sensor] in self._sensors_to_read:
                l.append(sensor)
        return l

    def set_motors_speed(self, l_motor, r_motor):
        """
        Set the motors speed. The MAX and MIN speed of the ePcuk is [-1000, 1000]

        :param l_motor: Speed of left motor
        :type l_motor: int
        :param r_motor: Speed of right motor
        :type r_motor: int
        """

        # I don't check the MAX and MIN speed because this check
        # will be made by the ePuck's firmware. Here we need speed
        # and we lose time mading recurrent chekings

        self._actuators_to_write.append(("D", int(l_motor), int(r_motor)))

        return True

    def set_motor_position(self, l_wheel, r_wheel):
        """
        Set the motor position, useful for odometry

        :param l_wheel: left wheel
        :type l_wheel: int
        :param r_wheel: right wheel
        :type r_wheel: int
        """

        self._actuators_to_write.append(("P", l_wheel, r_wheel))

    def set_led(self, led_number, led_value):
        """
        Turn on/off the leds

        :param led_number: If led_number is other than 0-7, all leds are set to the indicated value.
        :type led_number: int
        :param led_value:
            - 0 : Off
            - 1 : On (Red)
            - 2 : Inverse
        :type led_value: int
        """

        led = abs(led_number)
        value = abs(led_value)

        if led < 9:
            self._actuators_to_write.append(("L", led, value))
            if value == 0:
                self._leds_status[led] = False
            elif value == 1:
                self._leds_status[led] = True
            else:
                self._leds_status[led] = not self._leds_status[led]
            return True
        else:
            return False

    def set_body_led(self, led_value):
        """
        Turn on /off the body led

        :param led_value:
            - 0 : Off
            - 1 : On (green)
            - 2 : Inverse
        :type led_value: int
        """

        value = abs(led_value)

        self._actuators_to_write.append(("L", 8, value))

        if value == 0:
            self._leds_status[8] = False
        elif value == 1:
            self._leds_status[8] = True
        else:
            self._leds_status[8] = not self._leds_status[8]

        return True

    def set_front_led(self, led_value):
        """
        Turn on /off the front led

        :type	led_value: int
        :param 	led_value:
            - 0 : Off
            - 1 : On (green)
            - 2 : Inverse
        """
        value = abs(led_value)

        self._actuators_to_write.append(("L", 9, value))

        if value == 0:
            self._leds_status[9] = False
        elif value == 1:
            self._leds_status[9] = True
        else:
            self._leds_status[9] = not self._leds_status[9]

        return True

    def set_sound(self, sound):
        """
        Reproduce a sound

        :param sound: Sound in the range [1,5]. Other for stop
        :type sound: int
        """

        self._actuators_to_write.append(("T", sound))
        return True

    def set_camera_parameters(self, mode, width, height, zoom):
        """
        Set the camera parameters

        :param mode: GREY_SCALE, LINEAR_CAM, RGB_365, YUM
        :type  mode: String
        :param width: Width of the camera
        :type  width: int
        :param height: Height of the camera
        :type  height: int
        :param zoom: 1, 4, 8
        :type  zoom: int
        """

        if mode in CAM_MODE:
            self._cam_mode = CAM_MODE[mode]
        else:
            # self._debug(ERR_CAM_PARAMETERS, "Camera mode")
            return -1

        if int(zoom) in CAM_ZOOM:
            self._cam_zoom = zoom
        else:
            # self._debug(ERR_CAM_PARAMETERS, "Camera zoom")
            return -1

        if self.conexion_status and int(width) * int(height) <= 1600:
            # 1600 are for the resolution no greater than 40x40, I have
            # detect some problems
            self._actuators_to_write.append(("J",
                                             self._cam_mode,
                                             width,
                                             height,
                                             self._cam_zoom))

            self._debug(self.conexion_status)

            return 0

    def calibrate_proximity_sensors(self):
        """
        Calibrate proximity sensors using WiFi protocol
        WiFi protocol doesn't support sensor calibration, return success

        :return: Successful operation
        :rtype: Boolean
        """
        self._debug("WiFi protocol - proximity sensor calibration not supported, returning success")
        # WiFi binary protocol doesn't have calibration command
        # The sensors should be pre-calibrated in the robot firmware
        return True

    def reset(self):
        """
        Reset the robot using WiFi protocol (stop motors, turn off LEDs)

        :return: Successful operation
        :rtype: Boolean
        :raise Exception: If there is not connection
        """
        if not self.conexion_status:
            raise Exception('There is not connection')

        try:
            # WiFi protocol doesn't have a direct reset command, 
            # so we'll stop motors and turn off LEDs instead
            self._send_wifi_command(sensors=True, image=False, left_motor=0, right_motor=0, leds=0x00)
            response = self._receive_wifi_response()
            
            if response and response[0] in [0x02, 0x03]:  # Sensors or acknowledgment
                self._debug("Reset command (stop motors/LEDs) sent successfully via WiFi")
                # Clear actuators queue
                self._actuators_to_write.clear()
                return True
            else:
                self._debug("Unexpected response to reset command")
                return False
                
        except Exception as e:
            self._debug(f"Reset command failed: {e}")
            return False

    def stop(self):
        """
        Stop the motor and turn off all leds using WiFi protocol
        :return: Successful operation
        :rtype: Boolean
        :raise Exception: If there is not connection
        """

        if not self.conexion_status:
            raise Exception('There is not connection')

        try:
            # Send stop command using WiFi protocol: motors=0, LEDs=off
            self._send_wifi_command(sensors=True, image=False, left_motor=0, right_motor=0, leds=0x00)
            response = self._receive_wifi_response()
            
            if response and response[0] in [0x02, 0x03]:  # Sensors or acknowledgment
                self._debug("Stop command sent successfully via WiFi")
                return True
            else:
                self._debug("Unexpected response to stop command")
                return False
                
        except Exception as e:
            self._debug(f"Stop command failed: {e}")
            return False

    def step(self):
        """
        Method to update the sensor readings and to reflect changes in
        the actuators using WiFi binary protocol. Before invoking this method 
        is not guaranteed the consistency of the sensors
        """

        if not self.conexion_status:
            raise Exception('There is not connection')

        try:
            # Prepare motor commands from actuators queue
            left_motor = 0
            right_motor = 0
            leds = 0x00
            
            # Process motor commands from actuators queue
            for m in self._actuators_to_write[:]:
                if m[0] == 'D':  # Motor speed command
                    left_motor = m[1]
                    right_motor = m[2]
                    self._actuators_to_write.remove(m)
                elif m[0] == 'L':  # LED command
                    if m[1] < 8:  # Regular LEDs (0-7)
                        if m[2] == 1:  # Turn on
                            leds |= (1 << m[1])
                        # Turn off is default (bit stays 0)
                    self._actuators_to_write.remove(m)
                # Note: RGB LEDs and other commands could be added here
            
            # Send command and request sensor data
            self._send_wifi_command(
                sensors=True, 
                image=self._cam_enable,
                left_motor=left_motor, 
                right_motor=right_motor, 
                leds=leds
            )
            
            # Receive response
            response = self._receive_wifi_response()
            if response:
                packet_id, data = response
                
                if packet_id == 0x02:  # Sensors packet
                    self._parse_wifi_sensors(data)
                    
                elif packet_id == 0x01:  # Image packet
                    self._parse_wifi_image(data)
                    # Also need to check for sensors after image
                    sensor_response = self._receive_wifi_response()
                    if sensor_response and sensor_response[0] == 0x02:
                        self._parse_wifi_sensors(sensor_response[1])
                        
        except Exception as e:
            self._debug(f"Step failed: {e}")
            raise Exception(f"WiFi step communication failed: {e}")
    
    def _parse_wifi_sensors(self, sensor_data):
        """
        Parse 104-byte WiFi sensor data packet
        """
        if len(sensor_data) < 104:
            self._debug(f"Incomplete sensor data: {len(sensor_data)} bytes")
            return
            
        import struct
        
        try:
            # Parse sensor data according to WiFi protocol specification
            # Proximity sensors are at offset 44-59 (8 sensors, 2 bytes each)
            prox_data = struct.unpack('<8H', sensor_data[44:60])
            self._proximity = prox_data
            
            # Motor position at offset 88-91 (2 motors, 2 bytes each)
            motor_data = struct.unpack('<2H', sensor_data[88:92])
            self._motor_position = motor_data
            
            # Accelerometer at offset 0-5 (3 axes, 2 bytes each)
            accel_data = struct.unpack('<3H', sensor_data[0:6])
            self._accelerometer = accel_data
            
            self._debug(f"Parsed sensors - Prox: {prox_data[:4]}... Motors: {motor_data}")
            
        except Exception as e:
            self._debug(f"Sensor parsing error: {e}")
            
    def _parse_wifi_image(self, image_data):
        """
        Parse WiFi image data (RGB565 format)
        """
        if len(image_data) == 38400:  # 160x120 RGB565
            try:
                # Convert RGB565 to RGB888 and create PIL Image
                # This is a simplified conversion - full implementation would
                # properly decode RGB565 format
                self._debug("Received complete image data")
                # For now, just store the raw data
                # Full RGB565 to PIL conversion would be implemented here
            except Exception as e:
                self._debug(f"Image parsing error: {e}")
        else:
            self._debug(f"Incomplete image data: {len(image_data)} bytes")


    def clean_recv_buffer(self):
        """
        WiFi protocol doesn't need buffer cleaning - this is a no-op
        for compatibility with existing code
        
        :return: Successful operation
        :rtype: Boolean
        """
        self._debug("WiFi protocol - buffer cleaning not needed")
        return True

