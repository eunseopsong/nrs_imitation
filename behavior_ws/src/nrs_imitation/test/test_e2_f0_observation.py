"""F0 observation correctness and read-only launch routing; no ROS nodes."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from nrs_imitation.e2_f0_observation import TOPICS, summarize_records, analyze_bag

ROOT = Path(__file__).resolve().parents[4]


def test_base_to_actual_tcp_rotation_and_stale_pair_handling():
    p='/ur10skku/currentP';f='/ur10skku/currentF'
    summary, rows = summarize_records([
        (f,0,[1.,0.,0.]), (p,100_000_000,[0.,0.,200.,0.,0.,np.pi/2]),
        (f,110_000_000,[1.,0.,0.]), (f,200_000_000,[1.,0.,0.]),
        (f,400_000_000,[1.,0.,0.]), (f,500_000_000,[float('nan'),0.,0.])])
    assert summary['fresh_force_pose_pairs']==2
    assert summary['valid_force_messages']==4
    np.testing.assert_allclose(summary['force_tcp_N']['mean'],[0.,-1.,0.],atol=1e-10)
    assert rows[0]['fresh_pose'] is False and rows[-1]['fresh_pose'] is False
    assert rows[-1]['Fz_tcp_N'] is None
    assert not summary['calibration_verified'] and not summary['F0_frozen']
    assert summary['force_axis_sign_validated'] is False


def test_no_feedback_is_reported_without_guessing_a_zero():
    summary, rows = summarize_records([])
    assert summary['capture_status']=='insufficient_feedback'
    assert summary['force_tcp_N'] is None and not rows


def test_real_rosbag_serialization_and_analysis_without_ros_context(tmp_path):
    import rosbag2_py
    from rclpy.serialization import serialize_message
    from std_msgs.msg import Float64MultiArray, String
    writer=rosbag2_py.SequentialWriter()
    writer.open(rosbag2_py.StorageOptions(uri=str(tmp_path/'bag'),storage_id='sqlite3'),
                rosbag2_py.ConverterOptions(input_serialization_format='cdr',output_serialization_format='cdr'))
    for name,kind in TOPICS.items():
        writer.create_topic(rosbag2_py.TopicMetadata(name=name,type=kind,serialization_format='cdr'))
    for i in range(4):
        ns=1_000_000_000+i*100_000_000
        writer.write('/ur10skku/currentP',serialize_message(Float64MultiArray(data=[0.,0.,200.,0.,0.,0.])),ns)
        writer.write('/ur10skku/currentF',serialize_message(Float64MultiArray(data=[.1,-.2,.3,0.,0.,0.])),ns+1_000_000)
        writer.write('/ur10skku/ctlMode',serialize_message(String(data='Position')),ns)
    del writer
    summary=analyze_bag(tmp_path)
    assert summary['capture_status']=='recorded' and summary['fresh_force_pose_pairs']==4
    np.testing.assert_allclose(summary['force_tcp_N']['mean'],[.1,-.2,.3])
    assert (tmp_path/'force_pose.csv').is_file() and (tmp_path/'manifest.json').is_file()
    assert json.loads((tmp_path/'summary.json').read_text())['F0_frozen'] is False


@pytest.mark.parametrize('mode',['check','record'])
def test_launch_only_records_allowlisted_feedback_and_never_moves(mode,tmp_path):
    from launch import LaunchContext
    from launch.actions import ExecuteProcess, LogInfo
    from launch.utilities import perform_substitutions
    spec=importlib.util.spec_from_file_location('f0_observe_launch',ROOT/'behavior_ws/src/nrs_imitation/launch/e2_f0_observe.launch.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    context=LaunchContext();context.launch_configurations.update(mode=mode,duration_s='20',results_root=str(tmp_path))
    actions=module.configure(context)
    processes=[a for a in actions if isinstance(a,ExecuteProcess)]
    if mode=='check':
        assert all(isinstance(a,LogInfo) for a in actions)
        assert list(tmp_path.iterdir())==[]
    else:
        assert len(processes)==1
        command=[perform_substitutions(context,arg) for arg in processes[0].cmd]
        assert command[:3]==['ros2','bag','record']
        assert command[-3:]==list(TOPICS)
        assert not any('/cmdMotion' in arg or 'service'==arg or 'pub'==arg for arg in command)
        request=json.loads(next(tmp_path.rglob('request.json')).read_text())
        assert request['force_commands'] is False and request['F0_candidate_applied'] is False
