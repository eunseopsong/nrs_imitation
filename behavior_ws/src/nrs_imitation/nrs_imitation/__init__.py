import os
from datetime import datetime  # 파일 상단 import 부분에 추가


REPO_ROOT = os.path.expanduser("~/nrs_imitation")

class ActPolicyInfer:
    def __init__(self):
        # ... (기존 코드 그대로)

        self.policy.to(device).eval()
        self.node.get_logger().info("[READY] ACT model ready for inference")

        # ======================
        # 🔴 Inference 로그 파일 준비
        # ======================
        log_dir = os.path.join(REPO_ROOT, "analysis_logs")
        os.makedirs(log_dir, exist_ok=True)
        ts = datetime.now().strftime("%m%d_%H%M%S")
        self.log_path = os.path.join(log_dir, f"act_infer_{ts}.csv")
        self.log_file = open(self.log_path, "w", buffering=1)

        # CSV 헤더: step, ros_time, joint0..5
        self.log_file.write("step,ros_time,j0,j1,j2,j3,j4,j5\n")
        self.node.get_logger().info(f"[INFO] Inference log -> {self.log_path}")


        # --------------------------------------------------
        # 🔹 Interpolation (upsampling) 설정
        #   - policy: 20 Hz (모델 호출)
        #   - publish: 20 Hz 에서 한 step을 interp_factor 개로 쪼갬
        # --------------------------------------------------
        self.interp_factor = 100       # 예: 5면 한 policy step을 5개로 쪼갬
        self.prev_joints_cmd = None  # 이전 step의 (6,) joint action
        self.prev_force_cmd = None   # 이전 step의 (3,) force action
