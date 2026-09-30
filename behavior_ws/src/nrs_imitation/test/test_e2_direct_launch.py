"""Direct-launch bookkeeping tests; no LaunchService or ROS nodes are started."""
import ast
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from launch import LaunchContext
from launch.actions import DeclareLaunchArgument, GroupAction, IncludeLaunchDescription
from launch.utilities import perform_substitutions, normalize_to_list_of_substitutions
from launch_ros.actions import Node
from launch_ros.utilities import evaluate_parameters
from rclpy.parameter import Parameter

from nrs_imitation.e2_ablation import EXPERIMENT, ROOT
from nrs_imitation import e2_run_context as runs


def config():
    return json.loads((EXPERIMENT/'config.json').read_text())


def resolved_inference_parameters(include,context):
    """Resolve both nested launch layers/YAML without starting any ROS process."""
    for name,value in include.launch_arguments:
        context.launch_configurations[name]=perform_substitutions(context,normalize_to_list_of_substitutions(value))
    description=include.launch_description_source.get_launch_description(context)
    for action in description.entities:
        if isinstance(action,DeclareLaunchArgument):action.execute(context)
        elif isinstance(action,IncludeLaunchDescription):
            found=resolved_inference_parameters(action,context)
            if found is not None:return found
        elif isinstance(action,Node):
            executable=perform_substitutions(context,normalize_to_list_of_substitutions(action.node_executable))
            if executable=='inference_single_cam':
                values={}
                for params in evaluate_parameters(context,action._Node__parameters):
                    if isinstance(params,dict):values.update(params)
                return values
    return None


def assert_inference_parameter_types(parameters):
    """Use the consumer's declarations, not a second manually copied schema."""
    tree=ast.parse((ROOT/'behavior_ws/src/nrs_imitation/nrs_imitation/inference_core.py').read_text())
    checked=set()
    for node in ast.walk(tree):
        if not isinstance(node,ast.Call) or not isinstance(node.func,ast.Attribute):continue
        if node.func.attr!='declare_parameter' or len(node.args)<2:continue
        try:name,default=map(ast.literal_eval,node.args[:2])
        except (ValueError,TypeError):continue
        if name not in parameters or default is None or default==[]:continue
        actual=Parameter(name,value=parameters[name]).type_
        expected=Parameter(name,value=default).type_
        assert actual==expected,(name,parameters[name],actual,expected)
        checked.add(name)
    assert 'ckpt_auto_subdir' in checked and len(checked)>100
    return checked


def launch_fixture(tmp_path, monkeypatch, condition='B'):
    spec=importlib.util.spec_from_file_location('e2_direct_launch_test',ROOT/'behavior_ws/src/nrs_imitation/launch/e2_abc.launch.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    monkeypatch.setattr(runs,'session_root',lambda session:tmp_path/session)
    parent_logs=tmp_path/'parent_logs';parent_logs.mkdir()
    (parent_logs/'launch.log').write_text('offline test log\n')
    monkeypatch.setattr(module,'launch_config',SimpleNamespace(log_dir=str(parent_logs)))
    callbacks=[];monkeypatch.setattr(module.atexit,'register',callbacks.append)
    monkeypatch.setattr(module,'get_package_share_directory',lambda name:str(ROOT/'behavior_ws/src/nrs_imitation'))
    context=LaunchContext()
    context.launch_configurations.update(config=str(EXPERIMENT/'config.json'),mode='run',condition=condition,session='test_session')
    return module,context,callbacks


def test_common_state_trials_need_no_block_or_specimen_and_keep_unique_ids(tmp_path):
    cfg=config()
    a,af=runs.create_attempt(cfg,'B','test',session_path=tmp_path)
    b,bf=runs.create_attempt(cfg,'B','test',session_path=tmp_path)
    assert af!=bf and a['run']['repeat_id']!=b['run']['repeat_id']
    assert a['run']['block_id'] is None and a['common']['block_id'] is None
    assert a['run']['specimen_id']==a['run']['region_id']=='unspecified'
    assert a['run']['surface_reuse']=='assumed_same_state'
    assert a['run']['specimen_state_assumption']['physically_verified'] is False
    assert a['run']['specimen_state_assumption']['mode']=='same_state_each_run'
    assert runs.read(af/'attempt.json')['executed'] is False


def test_check_mode_does_not_create_attempt_or_register_lifecycle(tmp_path,monkeypatch):
    module,context,callbacks=launch_fixture(tmp_path,monkeypatch)
    context.launch_configurations['mode']='check'
    monkeypatch.setattr(module,'readiness',lambda cfg:{'test':'offline'})
    assert len(module.configure(context))==1
    assert not callbacks and not list(tmp_path.rglob('attempt.json'))


def test_blocked_direct_launch_preserves_each_attempt_and_parent_logs(tmp_path,monkeypatch):
    module,context,callbacks=launch_fixture(tmp_path,monkeypatch)
    monkeypatch.setattr(module,'hardware_blockers',lambda *args,**kwargs:['unverified test protocol'])
    for _ in range(2):
        with pytest.raises(RuntimeError,match='preflight blocked'):module.configure(context)
    for close in callbacks:close()
    attempts=list(tmp_path.glob('test_session/B/*/attempt.json'))
    assert len(attempts)==2
    for attempt in attempts:
        saved=runs.read(attempt)
        assert saved['status']=='preflight_blocked' and saved['executed'] is False
        assert saved['block_id'] is None
        assert (attempt.parent/'archive_manifest.json').is_file()
        assert (attempt.parent/'ros_logs/launch_parent/launch.log').read_text()=='offline test log\n'
        assert not (attempt.parent/'launch_claim.json').exists()


@pytest.mark.parametrize('condition',['B','C'])
def test_direct_launch_builds_existing_ros_include_and_records_lifecycle(tmp_path,monkeypatch,condition):
    module,context,callbacks=launch_fixture(tmp_path,monkeypatch,condition)
    monkeypatch.setattr(module,'hardware_blockers',lambda *args,**kwargs:[])
    actions=module.configure(context)
    group=next(a for a in actions if isinstance(a,GroupAction))
    include=next(a for a in group.get_sub_entities() if isinstance(a,IncludeLaunchDescription))
    args=dict(include.launch_arguments)
    folder=Path(args['e2_config']).parent
    run=runs.read(folder/'config.json')['run']
    assert args['use_force_observation']==str(condition=='C').lower()
    assert args['metrics_repeat_id']==run['run_id']
    assert args['overlay_record_output_dir']==str(folder/'video')
    parameters=resolved_inference_parameters(include,context)
    assert parameters is not None
    assert_inference_parameter_types(parameters)
    assert parameters['ckpt_auto_subdir']=='polishing/single_cam'
    assert parameters['ckpt_dir']==str(Path(config()['models'][condition]['checkpoint']).parent)
    from nrs_imitation import inference_core
    def prohibit_fallback(*args):raise AssertionError('Pinned E2 checkpoint must not use latest-directory search')
    monkeypatch.setattr(inference_core,'_find_latest_checkpoint_dir',prohibit_fallback)
    assert inference_core._resolve_checkpoint_dir(parameters['ckpt_dir'],parameters['act_root'],
        parameters['policy_class'],parameters['ckpt_auto_subdir'])==parameters['ckpt_dir']
    lifecycle=callbacks[0].__self__
    lifecycle.process_started(SimpleNamespace(pid=123,process_name='offline_mock_executor'),None)
    lifecycle.shutdown_requested(SimpleNamespace(reason='offline test shutdown'),None)
    lifecycle.process_exited(SimpleNamespace(pid=123,returncode=0),None)
    callbacks[0]()
    record=runs.read(folder/'attempt.json')
    assert record['launch_processes'][0]['exit_code']==0
    assert record['executed'] is None and record['completed'] is None
    assert record['status']=='launch_exited' and not record['archive_errors']
    assert (folder/'archive_manifest.json').is_file()


def test_prepared_cli_attempt_does_not_create_duplicate_and_cannot_be_reused(tmp_path,monkeypatch):
    module,context,callbacks=launch_fixture(tmp_path,monkeypatch)
    selected,folder=runs.create_attempt(config(),'B','test_session')
    record=runs.read(folder/'attempt.json');record['status']='launch_requested'
    runs.write(folder/'attempt.json',record)
    context.launch_configurations['config']=str(folder/'config.json')
    monkeypatch.setattr(module,'hardware_blockers',lambda *args,**kwargs:[])
    module.configure(context)
    assert len(list(tmp_path.rglob('attempt.json')))==1 and not callbacks
    with pytest.raises(FileExistsError):module.configure(context)


def test_checkpoint_subdirectory_remains_a_string_after_ros_yaml_evaluation():
    from launch.launch_description_sources import PythonLaunchDescriptionSource
    source=ROOT/'behavior_ws/src/nrs_imitation/launch/inference_clean_single_cam.launch.py'
    include=IncludeLaunchDescription(PythonLaunchDescriptionSource(str(source)),launch_arguments={
        'ckpt_auto_subdir':'false','gradcam_enable':'false','removal_viz_enable':'false'}.items())
    parameters=resolved_inference_parameters(include,LaunchContext())
    assert parameters['ckpt_auto_subdir']=='false'
    assert_inference_parameter_types(parameters)
