from glob import glob
from setuptools import find_packages, setup

package_name = "umi_ros2"

setup(
    name=package_name,
    version="0.0.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", glob("launch/*.launch.py")),
        ("share/" + package_name + "/config", glob("config/*.yaml")),
    ],
    install_requires=["setuptools", "dynamixel-sdk", "opencv-python", "pyserial"],
    zip_safe=True,
    maintainer="son_rb",
    maintainer_email="syoungk20@naver.com",
    description="ROS 2 nodes for controlling a Dynamixel-based UMI gripper.",
    license="TODO: License declaration",
    extras_require={
        "test": [
            "pytest",
        ],
    },
    entry_points={
        "console_scripts": [
            "umi_gripper_pico = umi_ros2.umi_grp:main",
            "umi_gripper_sub = umi_ros2.umi_grp_sub:main",
            "umi_gripper_sub_pwm = umi_ros2.umi_grp_sub_pwm:main",
        ],
    },
)
