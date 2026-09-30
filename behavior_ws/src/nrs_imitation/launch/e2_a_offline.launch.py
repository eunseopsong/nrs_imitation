"""Run the registered A policy and candidate F0 offline check; no ROS nodes."""
import os
from pathlib import Path

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, LogInfo, OpaqueFunction, EmitEvent
from launch.events import Shutdown
from launch.substitutions import LaunchConfiguration

from nrs_imitation.e2_ablation import ROOT


def configure(context):
    config = Path(LaunchConfiguration('config').perform(context)).expanduser().resolve()
    python = Path('/home/eunseop/miniconda3/envs/nrs_imitation/bin/python3')
    script = ROOT/'scripts/e2_A_pilot_offline.py'
    for path in (config, python, script):
        if not path.is_file():
            raise FileNotFoundError(str(path))
    pythonpath = os.pathsep.join(filter(None, (
        str(ROOT/'behavior_ws/src/nrs_imitation'), str(ROOT/'source'),
        os.environ.get('PYTHONPATH', ''))))

    def finished(event, _context):
        if event.returncode != 0:
            raise RuntimeError('E2 A offline check failed; see the process output and saved attempt.')
        return [LogInfo(msg='E2 A offline check completed; result path is printed above. No hardware approval or F0 freezing.'),
                EmitEvent(event=Shutdown(reason='A offline check completed'))]

    return [
        LogInfo(msg='E2 A offline policy/F0 check only. Robot and spindle receive no commands.'),
        ExecuteProcess(cmd=[str(python), str(script), '--config', str(config)],
            cwd=str(ROOT), output='screen', on_exit=finished,
            additional_env=dict(PYTHONPATH=pythonpath, HF_HUB_OFFLINE='1',
                OPENBLAS_NUM_THREADS='2', OMP_NUM_THREADS='2', PYTHONUNBUFFERED='1')),
    ]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('config',
            default_value=str(ROOT/'experiments/e2_A_pilot_20260927/config.json'),
            description='Candidate A pilot config used only by the offline check'),
        OpaqueFunction(function=configure),
    ])
