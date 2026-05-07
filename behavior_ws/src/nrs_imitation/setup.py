import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'nrs_imitation'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        (
            'share/ament_index/resource_index/packages',
            ['resource/' + package_name],
        ),
        (
            'share/' + package_name,
            ['package.xml'],
        ),
        (
            os.path.join('share', package_name, 'launch'),
            glob('launch/*.launch.py'),
        ),
    ],
    install_requires=[
        'setuptools',
    ],
    zip_safe=True,
    maintainer='eunseop',
    maintainer_email='lexondms1@g.skku.edu',
    description='ROS 2 nodes for force-aware imitation learning recording and policy inference',
    license='MIT',
    extras_require={
        'test': ['pytest'],
    },
    entry_points={
        'console_scripts': [
            # Inference / debug
            'node_cmdmotion_infer = nrs_imitation.node_cmdmotion_infer:main',
            'node_check_inference = nrs_imitation.node_check_inference:main',

            # Demonstration recording
            'vr_demo_hdf5_recorder = nrs_imitation.vr_demo_hdf5_recorder:main',
            'vr_demo_txt_recorder = nrs_imitation.vr_demo_txt_recorder:main',

            # Joystick controller
            'vr_demo_joy_controller = nrs_imitation.vr_demo_joy_controller:main',
        ],
    },
)