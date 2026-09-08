import os
from glob import glob

from setuptools import find_packages, setup

package_name = 'stain_relative_frame'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml', 'README.md']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='eunseop',
    maintainer_email='lexondms1@g.skku.edu',
    description=(
        'Stain-centred TCP frame: home-pose repeatability, pixel->robot homography, '
        'stain-origin detection, dataset relativisation and the pre-training '
        'shortcut-leakage gate.'
    ),
    license='MIT',
    extras_require={'test': ['pytest']},
    entry_points={
        'console_scripts': [
            # [0] preconditions
            'home_pose_repeatability = stain_relative_frame.home_pose_repeatability:main',
            'clean_reference_capture = stain_relative_frame.clean_reference_capture:main',

            # [1] homography calibration (offline, once)
            'homography_collect = stain_relative_frame.homography_collect:main',
            'homography_pick_pixels = stain_relative_frame.homography_pick_pixels:main',
            'homography_jog_calibrate = stain_relative_frame.homography_jog_calibrate:main',
            'homography_depth_calibrate = stain_relative_frame.homography_depth_calibrate:main',
            'homography_fit = stain_relative_frame.homography_fit:main',

            # [2] stain origin
            'stain_origin_offline = stain_relative_frame.stain_origin_offline:main',
            'stain_origin_node = stain_relative_frame.stain_origin_node:main',

            # [3] dataset conversion
            'dataset_relativize = stain_relative_frame.dataset_relativize:main',

            # [4] pre-training gate
            'validate_shortcut = stain_relative_frame.validate_shortcut:main',

            # [5] path audit
            'audit_inference_path = stain_relative_frame.audit_inference_path:main',

            # driver
            'run_pipeline = stain_relative_frame.run_pipeline:main',
        ],
    },
)
