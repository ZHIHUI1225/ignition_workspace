from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'mpc_test_package'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'rviz'), glob('rviz/*.rviz')),
        (os.path.join('lib', package_name), []),  # Create lib/mpc_test_package directory
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Test User',
    maintainer_email='test@example.com',
    description='Test package for MobileRobotMPC class',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'mpc_test_node = mpc_test_package.mpc_test_node:main',
            'cmd_vel_publisher = mpc_test_package.cmd_vel_publisher:main',
        ],
    },
)
