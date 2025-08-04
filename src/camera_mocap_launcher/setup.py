from setuptools import setup, find_packages
import os
from glob import glob

package_name = 'camera_mocap_launcher'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='ZHIHUI1225',
    maintainer_email='hui1225.zhi@connect.polyu.hk',
    description='Package for launching camera and VRPN mocap client',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'camera_node = camera_mocap_launcher.camera_node:main',
            'rectangle_detector = camera_mocap_launcher.rectangle_detector:main',
            'plot_path_on_video = camera_mocap_launcher.plot_path_on_video:main',
        ],
    },
)