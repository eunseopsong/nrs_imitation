"""Validate launch routing without ROS nodes, policy inference, or robot I/O."""
import copy
import importlib.util
import json
from pathlib import Path

import pytest
from launch import LaunchContext, LaunchDescription
from launch.actions import (DeclareLaunchArgument, ExecuteProcess, GroupAction,
                            IncludeLaunchDescription, LogInfo, OpaqueFunction,
                            PopEnvironment, PopLaunchConfigurations,
                            PushEnvironment, PushLaunchConfigurations,
                            RegisterEventHandler, SetEnvironmentVariable, SetLaunchConfiguration,
                            UnsetLaunchConfiguration)
from launch.utilities import normalize_to_list_of_substitutions, perform_substitutions
from launch_ros.actions import Node
from launch_ros.utilities import evaluate_parameters


ROOT = Path(__file__).resolve().parents[4]
CONFIG = ROOT / 'experiments/e2_rule_replay_20260920/config.json'
SPEC = importlib.util.spec_from_file_location('rtc_launch',
    ROOT / 'behavior_ws/src/nrs_imitation/launch/rtc_timed.launch.py')
rtc = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(rtc)


def context(method, mode='check'):
    c = LaunchContext()
    c.launch_configurations.update(dict(method=method, config=str(CONFIG), mode=mode,
        checkpoint='', episode='', run_tag=''))
    return c


def test_missing_selection_rejected_before_hardware_preflight(monkeypatch):
    def unexpected(*args, **kwargs):
        pytest.fail('Should reject selection before hardware preflight')
    monkeypatch.setattr(rtc, 'hardware_blockers', unexpected)
    for method, message in [('C', 'requires checkpoint'), ('T', 'requires episode')]:
        with pytest.raises(ValueError, match=message):
            rtc.configure(context(method))


@pytest.mark.parametrize('method', ['R', 'T', 'C'])
def test_check_has_no_node_include_or_files(method, monkeypatch, tmp_path):
    cfg = json.loads(CONFIG.read_text())
    c = context(method)
    c.launch_configurations.update(checkpoint=cfg['il']['checkpoint'], episode=cfg['replay']['episode_id'])
    monkeypatch.setattr(rtc, 'ROOT', tmp_path)
    actions = rtc.configure(c)
    assert all(isinstance(a, LogInfo) for a in actions)
    assert list(tmp_path.iterdir()) == []


def test_reject_wrong_selection_and_policy_preset():
    with pytest.raises(ValueError, match='differs from config.il'):
        rtc.prepare('C', CONFIG, '/tmp/nonselected/policy_best.ckpt')
    with pytest.raises(ValueError, match='differs from config.replay'):
        rtc.prepare('T', CONFIG, episode='episode_999')


def test_different_preset_and_off_stats_rejected(monkeypatch):
    original = json.loads(CONFIG.read_text())
    cfg = copy.deepcopy(original)
    cfg['il']['flow_infer_steps'] = 11
    monkeypatch.setattr(rtc, 'load_config', lambda path: cfg)
    with pytest.raises(ValueError, match='flow_infer_steps=10'):
        rtc.prepare('R', CONFIG)
    monkeypatch.setattr(rtc, 'load_config', lambda path: original)
    monkeypatch.setattr(rtc.StatsUnpickler, 'load', lambda self:
                        {'policy_config': {'use_force_observation': False}})
    with pytest.raises(ValueError, match='explicitly record'):
        rtc.prepare('C', CONFIG, original['il']['checkpoint'])


@pytest.mark.parametrize('method', ['R', 'T', 'C'])
def test_run_routes_to_existing_launch_and_preserves_config(method, monkeypatch, tmp_path):
    original = CONFIG.read_bytes()
    cfg = json.loads(original)
    monkeypatch.setattr(rtc, 'ROOT', tmp_path)
    monkeypatch.setattr(rtc, 'get_package_share_directory', lambda name: str(tmp_path))
    c = context(method, 'run')
    c.launch_configurations.update(checkpoint=cfg['il']['checkpoint'], episode=cfg['replay']['episode_id'])
    # Construct actions only. Never execute the included robot launch.
    actions = rtc.configure(c)
    assert isinstance(actions[-1], GroupAction)
    # Inspect the included launch via actions; GroupAction also adds scope actions.
    from launch.actions import IncludeLaunchDescription
    include = next(a for a in actions[-1].get_sub_entities() if isinstance(a, IncludeLaunchDescription))
    forwarded = dict(include.launch_arguments)
    assert forwarded['execution_method'] == rtc.METHODS[method]
    assert forwarded['inference_mode'] == 'timed_topic'
    assert forwarded['ckpt_dir'] == str(Path(cfg['il']['checkpoint']).parent)
    snapshot = json.loads(Path(forwarded['e2_config']).read_text())
    assert snapshot['paper_experiment'] == 'E1'
    assert snapshot['e1_operator_automation'] is True
    assert snapshot['execution_equivalence'] == 'unverified'
    assert snapshot['recipe'] == cfg['recipe']
    assert snapshot['executor'] == cfg['executor']
    assert snapshot['il'] == cfg['il']
    assert CONFIG.read_bytes() == original


@pytest.mark.parametrize('method', ['R', 'T', 'C'])
def test_nested_launch_uses_stain_config_and_keeps_experiment_context(method, monkeypatch, tmp_path):
    """Expand real nested launches, stopping before every node/process action."""
    from ament_index_python.packages import get_package_share_directory
    from stain_relative_frame.config import load_config as load_stain_config

    def forbidden_process(*args, **kwargs):
        pytest.fail('Offline launch inspection must never execute a process')

    monkeypatch.setattr(Node, 'execute', forbidden_process)
    monkeypatch.setattr(ExecuteProcess, 'execute', forbidden_process)
    monkeypatch.setattr(rtc, 'ROOT', tmp_path)
    cfg = json.loads(CONFIG.read_text())
    c = context(method, 'run')
    c.launch_configurations.update(checkpoint=cfg['il']['checkpoint'], episode=cfg['replay']['episode_id'])
    parent_config = dict(c.launch_configurations)
    c.environment['RTC_TEST_INHERITED_ENV'] = 'preserved'
    parent_environment = dict(c.environment)
    nodes = {}
    allowed_actions = (DeclareLaunchArgument, GroupAction, IncludeLaunchDescription,
                       OpaqueFunction, PopEnvironment, PopLaunchConfigurations,
                       PushEnvironment, PushLaunchConfigurations,
                       RegisterEventHandler, SetEnvironmentVariable, SetLaunchConfiguration,
                       UnsetLaunchConfiguration)

    def inspect(entities):
        for entity in entities:
            if isinstance(entity, LaunchDescription):
                inspect(entity.entities)
                continue
            if entity.condition is not None and not entity.condition.evaluate(c):
                continue
            if isinstance(entity, Node):
                executable = perform_substitutions(c, normalize_to_list_of_substitutions(entity.node_executable))
                # Humble has no public getter for normalized node parameters.
                # Evaluate them directly without Node.execute or parameter files.
                params = evaluate_parameters(c, entity._Node__parameters)
                nodes[executable] = (params, dict(c.launch_configurations))
                assert c.environment['RTC_TEST_INHERITED_ENV'] == 'preserved'
                continue
            if isinstance(entity, LogInfo):
                continue
            assert isinstance(entity, allowed_actions), type(entity).__name__
            inspect(entity.execute(c) or [])

    inspect(rtc.configure(c))
    origin_params = nodes['stain_origin_node'][0][0]
    expected_config = Path(get_package_share_directory('stain_relative_frame')) / 'config/stain_relative_frame.yaml'
    assert Path(origin_params['config']).resolve() == expected_config.resolve()
    load_stain_config(origin_params['config'])  # Exercise the loader that failed in the user's run.
    assert origin_params['method'] == 'auto'
    assert origin_params['detect_params'] == cfg['common']['frozen_roi']['detect_params_file']
    executor_context = nodes['e2_executor'][1]
    inference_context = nodes['inference_single_cam'][1]
    snapshot = Path(executor_context['e2_config'])
    assert snapshot == Path(inference_context['e2_config'])
    assert executor_context['execution_method'] == rtc.METHODS[method]
    assert inference_context['inference_mode'] == 'timed_topic'
    assert json.loads(snapshot.read_text())['launch_context']['legacy_config_path'] == str(CONFIG)
    assert c.launch_configurations == parent_config
    assert dict(c.environment) == parent_environment
