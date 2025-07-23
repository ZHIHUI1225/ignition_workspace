from setuptools import setup

package_name = 'visualization'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='user',
    maintainer_email='user@example.com',
    description='ROS2 package to visualize VRPN mocap poses',
    license='MIT',
    entry_points={
        'console_scripts': [
            'pose_plotter = visualization.pose_plotter:main',
        ],
    },
)
