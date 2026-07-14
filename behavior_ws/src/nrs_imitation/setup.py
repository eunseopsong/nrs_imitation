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
            'inference_single_cam = nrs_imitation.inference_single_cam:main',
            'inference_dual_cam = nrs_imitation.inference_dual_cam:main',
            'inference_gripper_single_cam = nrs_imitation.inference_gripper_single_cam:main',
            'stain_mask_publisher = nrs_imitation.stain_mask_publisher:main',

            # Demonstration recording
            'hdf5_recorder_single_cam = nrs_imitation.hdf5_recorder_single_cam:main',
            'hdf5_recorder_single_cam_stain_mask = nrs_imitation.hdf5_recorder_single_cam_stain_mask:main',
            'hdf5_recorder_dual_cam = nrs_imitation.hdf5_recorder_dual_cam:main',
            'gripper_hdf5_recorder_single_cam = nrs_imitation.gripper_hdf5_recorder_single_cam:main',
            'gripper_hdf5_recorder_dual_cam = nrs_imitation.gripper_hdf5_recorder_dual_cam:main',
            'vr_demo_txt_recorder = nrs_imitation.vr_demo_txt_recorder:main',

            # Stage-1 / Stage-2 separated workflow
            'vr_stage1_hdf5_recorder = nrs_imitation.vr_stage1_hdf5_recorder:main',
            'vr_stage1_episode_pusher = nrs_imitation.vr_stage1_episode_pusher:main',

            # Joystick controller
            'vr_demo_joy_controller = nrs_imitation.vr_demo_joy_controller:main',
        ],
    },
)
