from setuptools import setup, find_packages
import os
from glob import glob

package_name = 'epuck_driver_ros2'

setup(
    name=package_name,
    version='1.0.0',
    packages=find_packages(),
    package_data={
        package_name: [
            '*.py',  # 包含所有Python文件
            '__init__.py',
            'ePuck.py', 
            'epuck_driver_ros2.py',
            'simple_cmd_publisher.py',
            'keyboard_control.py'
        ]
    },
    package_dir={'': '.'},
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # 使用glob自动包含所有相关文件
        (os.path.join('lib', package_name), glob('epuck_driver_ros2/*.py')),  # 安装所有.py文件
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml'))
    ],
    include_package_data=True,
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='ramonmelo',
    maintainer_email='ramonmelo@todo.todo',
    description='ROS2 driver for e-puck robot with WiFi communication',
    license='GPLv3',
    tests_require=['pytest'],
)
