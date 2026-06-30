import os
from glob import glob

from setuptools import find_packages, setup


package_name = "dynamixel_joy_controller"


setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        (
            "share/ament_index/resource_index/packages",
            ["resource/" + package_name],
        ),
        (
            "share/" + package_name,
            ["package.xml", "README.md"],
        ),
        (
            os.path.join("share", package_name, "launch"),
            glob("launch/*.launch.py"),
        ),
        (
            os.path.join("share", package_name, "config"),
            glob("config/*.yaml"),
        ),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="eunseop",
    maintainer_email="lexondms1@g.skku.edu",
    description="Joystick command nodes for Dynamixel-based gripper controllers.",
    license="MIT",
    extras_require={
        "test": ["pytest"],
    },
    entry_points={
        "console_scripts": [
            (
                "f710_gripper_joy_controller = "
                "dynamixel_joy_controller.f710_gripper_joy_controller:main"
            ),
        ],
    },
)
