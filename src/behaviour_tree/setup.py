from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'behaviour_tree'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # Include launch files in the installation
        (os.path.join('share', package_name, 'launch'),
         glob(os.path.join('launch', '*launch.[pxy][yma]*')))
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='root',
    maintainer_email='hui1225.zhi@connect.polyu.hk',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'my_behaviour_tree = behaviour_tree.my_behaviour_tree:main',
            'my_behaviour_tree_modular = behaviour_tree.my_behaviour_tree_modular:main',
            'simple_tree = behaviour_tree.simple_tree:main',
            'snapshot_streams_demo = behaviour_tree.snapshot_streams_demo:main',
            'tutorial_four_reproduction = behaviour_tree.tutorial_four_reproduction:main',
            # 'demo_bt = behaviour_tree.demo_bt:main',
        ],
    },
)
