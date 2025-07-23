from setuptools import setup

package_name = 'camera_mocap_launcher'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', [
            'launch/camera_mocap_with_vrpn.launch.py',
            'launch/camera_node.launch',
        ]),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='ZHIHUI1225',
    maintainer_email='hui1225.zhi@connect.polyu.hk',
    description='Package for launching camera and VRPN mocap client',
    license='Apache-2.0',
)