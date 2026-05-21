#!/usr/bin/env python3
import os
import csv
from pathlib import Path
import numpy as np
import rclpy
from rclpy.node import Node

from vive_tracker_interfaces.srv import ViveCalibration
from vive_tracker_ros2.utils import matrix_to_pose


class CalibrationTestNode(Node):
    def __init__(self):
        super().__init__("calibration_test_node")
        self.client = self.create_client(ViveCalibration, "vive_tracker_ros/calibrate")
        self.wait_for_service()

    def wait_for_service(self):
        while not self.client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("서비스를 기다리는 중입니다...")

    def load_matrices_from_csv(self, file_path):
        """CSV 파일에서 4x4 행렬을 읽어옵니다."""
        matrices = []
        with open(file_path, "r") as f:
            reader = csv.reader(f)
            lines = list(reader)

            # 4줄마다 하나의 행렬 생성
            for i in range(0, len(lines), 4):
                if i + 4 <= len(lines):
                    matrix = np.zeros((4, 4))
                    for j in range(4):
                        row = [float(val) for val in lines[i + j]]
                        matrix[j] = row
                    matrices.append(matrix)

        return matrices

    def run_calibration_test(self, robot_file, vive_file):
        """로봇 포즈와 바이브 트래커 포즈를 읽어서 캘리브레이션 서비스를 호출합니다."""
        # 파일 경로 설정vive_tracker_ros2/config/calibration_matrix_temp.json
        package_dir = Path(__file__).resolve().parents[1]
        robot_path = os.path.join(package_dir, "data", robot_file)
        vive_path = os.path.join(package_dir, "data", vive_file)

        # 행렬 데이터 로드
        robot_matrices = self.load_matrices_from_csv(robot_path)
        vive_matrices = self.load_matrices_from_csv(vive_path)

        if len(robot_matrices) != len(vive_matrices):
            self.get_logger().error(
                f"로봇 포즈({len(robot_matrices)})와 트래커 포즈({len(vive_matrices)})의 개수가 다릅니다."
            )
            return False

        # 행렬을 Pose 메시지로 변환
        robot_poses = [matrix_to_pose(matrix) for matrix in robot_matrices]
        vive_poses = [matrix_to_pose(matrix) for matrix in vive_matrices]

        # 서비스 요청 생성
        request = ViveCalibration.Request()
        request.robot_poses = robot_poses
        request.tracker_poses = vive_poses

        # 서비스 호출
        future = self.client.call_async(request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=3)

        if future.done():
            self.get_logger().info(f"캘리브레이션 결과: {future.result().success}")
            return future.result().success
        else:
            self.get_logger().error("서비스 호출 실패")
            return False


def main():
    rclpy.init()
    node = CalibrationTestNode()
    node.run_calibration_test("robot_tf.csv", "vive_tf.csv")
    node.destroy_node()
    rclpy.shutdown()
