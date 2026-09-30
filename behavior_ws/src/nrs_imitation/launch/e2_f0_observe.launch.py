"""Record unloaded robot feedback only. No motion/force services or publishers."""
from datetime import datetime
import json
from pathlib import Path
import signal
import tempfile

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, LogInfo, OpaqueFunction, RegisterEventHandler, TimerAction, EmitEvent
from launch.event_handlers import OnProcessStart, OnProcessExit
from launch.events import matches_action
from launch.events.process import SignalProcess
from launch.substitutions import LaunchConfiguration

from nrs_imitation.e2_f0_observation import TOPICS, analyze_bag


def configure(context):
    mode = LaunchConfiguration('mode').perform(context)
    duration = float(LaunchConfiguration('duration_s').perform(context))
    if not 5. <= duration <= 120.:
        raise ValueError('duration_s must be between 5 and 120 seconds')
    if mode == 'check':
        return [LogInfo(msg='F0 unloaded observation: rosbag only; topics='+', '.join(TOPICS)+
            '; duration='+str(duration)+' s; no nodes/processes started by check.')]
    if mode != 'record':
        raise ValueError('mode must be check or record')
    today = datetime.now()
    base = Path(LaunchConfiguration('results_root').perform(context)).expanduser()/today.strftime('%Y%m%d')/'E2/F0_validation'
    base.mkdir(parents=True,exist_ok=True)
    folder = Path(tempfile.mkdtemp(prefix=today.strftime('unloaded_%H%M%S_'),dir=base))
    request = dict(created_at=today.astimezone().isoformat(),phase='unloaded_home_baseline',
        duration_s=duration,topics=TOPICS,operator_preparation='Robot stationary at HOME, tool clear of workpiece, spindle OFF; robot feedback drivers running.',
        motion_commands=False,force_commands=False,F0_candidate_applied=False,
        purpose='First F0 validation step: capture baseline; not a 23 N application test.',
        results_directory=str(folder))
    (folder/'request.json').write_text(json.dumps(request,ensure_ascii=False,indent=2)+'\n')
    recorder = ExecuteProcess(cmd=['ros2','bag','record','--node-name','e2_f0_unloaded_recorder',
        '-s','sqlite3','-o',str(folder/'bag'),*TOPICS],output='screen',sigterm_timeout='10',sigkill_timeout='5')
    def finished(event, ctx):
        try:
            summary=analyze_bag(folder)
            message='F0 observation saved: '+str(folder)+'; status='+summary['capture_status']
            if summary['force_tcp_N']:
                message+='; TCP mean Fz='+format(summary['force_tcp_N']['mean'][2],'.3f')+' N'
            message+='; calibration/F0 are NOT approved by this capture.'
        except Exception as exc:
            (folder/'analysis_error.json').write_text(json.dumps(dict(error=str(exc),recorder_returncode=event.returncode),indent=2)+'\n')
            message='F0 observation incomplete: '+str(folder)+'; '+str(exc)
        return [LogInfo(msg=message)]
    return [LogInfo(msg='Prepare stationary unloaded HOME with spindle OFF. Feedback capture only; robot receives no motion/force commands. '+str(folder)),
        RegisterEventHandler(OnProcessStart(target_action=recorder,on_start=[TimerAction(period=duration,actions=[
            EmitEvent(event=SignalProcess(signal_number=signal.SIGINT,process_matcher=matches_action(recorder)))])])),
        RegisterEventHandler(OnProcessExit(target_action=recorder,on_exit=finished)),recorder]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('mode',default_value='check',choices=['check','record']),
        DeclareLaunchArgument('duration_s',default_value='20'),
        DeclareLaunchArgument('results_root',default_value='/home/eunseop/nrs_imitation/results'),
        OpaqueFunction(function=configure)])
