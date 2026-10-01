"""Dedicated low-load E2 execution process. Never imports torch or inference_core.

One process owns timed cmdMotion after the existing verified start alignment.
Shutdown keeps ROS alive for queue cancellation and observed controller hold.
"""
import fcntl
import json
import signal
import time
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation
import rclpy
from rclpy.node import Node
from rclpy.signals import SignalHandlerOptions
from rclpy.qos import QoSProfile, ReliabilityPolicy
from std_msgs.msg import Float64MultiArray, String
from std_srvs.srv import Trigger
from y2_rob_motion_interfaces.srv import SingleArmCommand

from .e2_providers import load_config, hardware_blockers, file_hash
from .e2_timed_execution import (TRANSPORT_VERSION, TimedPlan, TimedExecution,
                                ExecutionFault, StopVerifier, executor_settings, clock_id)
from .execution_metrics import ExecutionRecorder, stamp, pose_fields
from .e2_return_home import ReturnHome
from .e2_ablation import is_ablation, prepare_A_force
from .e2_direct_abc import PARAMETERS as DIRECT_PARAMETERS, from_node, object_hash
from .e2_protection import (ProtectionMonitor, FeedbackProtectionMonitor, ProtectionFault, record_from_config,
                            ACQUISITION_TOPIC, PROVENANCE_TOPIC, VERSION as PROTECTION_VERSION)


class E2Executor(Node):
    def __init__(self):
        super().__init__('e2_executor')
        for name, default in [('e2_config',''),('execution_method','il'),('e2_enable_hardware',False),
                              ('e2_session_id',''),('metrics_run_tag','E2'),('act_root','/home/eunseop/nrs_imitation')]:
            self.declare_parameter(name,default)
        for name, default in DIRECT_PARAMETERS.items():
            self.declare_parameter(name, default)
        self.config_path = self.get_parameter('e2_config').value
        self.method = self.get_parameter('execution_method').value
        self.direct_a = bool(self.get_parameter('e2_direct_a').value)
        self.direct_abc = bool(self.get_parameter('e2_direct_abc').value)
        if self.is_direct_run() and (self.config_path or self.method != 'il'):
            raise ValueError('Direct E2 requires IL and an empty e2_config')
        self.config = from_node(self) if self.is_direct_run() else load_config(self.config_path)
        self.ablation = is_ablation(self.config)
        self.e1_operator_automation = (not self.ablation and
            self.config.get('paper_experiment') == 'E1' and
            self.config.get('e1_operator_automation') is True)
        blockers = ([] if self.is_direct_run() else
            hardware_blockers(self.config,self.method,self.get_parameter('e2_enable_hardware').value))
        if blockers: raise RuntimeError('E2 executor preflight: '+ '; '.join(blockers))
        self.a_processing_force = (self.config['external_force_N'] if self.is_direct_run() else
            prepare_A_force(self.config) if self.ablation and self.config['condition'] == 'A' else None)
        self.session = self.get_parameter('e2_session_id').value
        if not self.session: raise ValueError('A unique e2_session_id is required')
        self.lock_file = open('/tmp/nrs_e2_executor_ur10skku.lock','a')
        fcntl.flock(self.lock_file,fcntl.LOCK_EX | fcntl.LOCK_NB)
        self.identity = object_hash(self.config) if self.is_direct_run() else file_hash(self.config_path)
        self.clock_identity = clock_id()
        self.settings = executor_settings(self.config, self.method)
        self.protection = (FeedbackProtectionMonitor(self.config['protection'], self.settings) if self.is_direct_run() else
            ProtectionMonitor(record_from_config(self.config), self.settings)
            if self.ablation and self.config['condition'] == 'A' else None)
        self.protection_wait_reason = None
        self.engine = TimedExecution(self.settings)
        self.state = 'waiting_feedback'
        self.pose = self.force = None
        self.pose_at = self.force_at = self.mode_at = -float('inf')
        self.mode = ''
        self.pending = None
        self.stop_failed_latched = False
        self.stop_ack = False
        self.stopping_initial = False
        self.stop_reason = None
        self.stop_started = None
        self.last_mode_send = -float('inf')
        self.started_at = None
        self.arm_at = None
        self.final_target = None
        self.return_home = None
        self.return_spec = self.config.get('recipe', {}).get('return_home', {}) if self.method == 'rule' else {}
        self.shutdown_requested = False
        self.shutdown_at = None
        self.processing = False
        self.processing_completed = False
        self.auto_processing_interval = 0
        self.last_console_status = None
        self.last_operator_event = None
        self.phase_index = -1
        self.sample_last = {}
        self.feedback_received = {'pose':0,'force':0}
        self.command_id = 0
        self.reference = None
        self.ready_reference = None
        self.last_phase = None
        self.provider_at = -float('inf')
        root = Path(self.get_parameter('act_root').value)
        metadata = dict(method={'il':'C','rule':'R','replay':'T'}[self.method],
            execution_method=self.method,transport=TRANSPORT_VERSION,session_id=self.session,
            c_pose_conditioning=self.settings.get('c_pose_conditioning'),
            rtc_pose_conditioning=self.config.get('rtc_pose_conditioning'),
            code_root=str(root),config=self.config,config_sha256=self.identity,
            controller_applied_force=None,raw_sensor_available=False,rpm_measured=None,
            rpm_setpoint=self.config["common"].get("rpm_setpoint"),
            rpm_status=self.config["common"].get("rpm_status", "unspecified"),
            rpm_unknown_reason=self.config["common"].get("rpm_unknown_reason"),
            force_frame='robot_base measurement; target frame depends on deployed force controller',
            physical_completion_available='verified stationary feedback/mode; optional R Position lift, measured release and demo-start return',
            sampling=dict(feedback_max_hz=20.,commands_hz=125.,source_timestamp='headerless; acquisition time unknown'),
            clock_id=self.clock_identity,host_monotonic_only=True,
            artifact_paths=[__file__,str(Path(__file__).with_name('e2_timed_execution.py')),
                str(Path(__file__).with_name('e2_return_home.py')),self.config_path],
            warnings=['E2 timed transport differs from E1 legacy service stream',
                      'Actual TCP response can differ from requested timed path due to common filtering, limits, and force control'])
        if self.e1_operator_automation:
            metadata['processing_markers'] = dict(mode='automatic_contact_proxy',
                source='existing executor contact gate',
                contact_on_N=self.settings['contact_on_N'],
                contact_off_N=self.settings['contact_off_N'],
                physical_processing_verified=False, evaluation_interval_verified=False,
                no_contact_interval='NA',
                note='Contact intervals are estimates, not operator/teacher processing labels. '
                     'Stop requests close an open interval without asserting contact release.')
            metadata['automatic_close'] = 'after existing queue-cancel and stationary-TCP verification'
        if self.ablation:
            metadata.update(experiment_id='E2', method=self.config['condition'],
                execution_contract_hash=self.config['execution_contract_hash'],
                run=self.config.get('run'), predicted_force='not_predicted' if self.config['condition']=='A' else 'policy',
                force_sign_provenance=self.config['calibration'],
                trial_stage=self.config.get('trial_stage', 'main'),
                external_F0=self.config['f0'], commissioning=self.config.get('pilot'),
                protection_version=PROTECTION_VERSION if self.protection is not None else None,
                warnings=['Filtered base wrench is not automatically calibrated contact normal force',
                          'sent target is not controller-applied target'])
        if self.is_direct_run():
            condition = self.config['condition']
            metadata.update(experiment_id='E2', method=condition,
                predicted_force='not_predicted' if condition == 'A' else 'policy',
                run=self.config['run'], comparison=self.config.get('comparison'),
                external_force_N=self.a_processing_force,
                runtime_profile='matched_ABC' if self.direct_abc else 'config_free_A',
                protection_version=self.protection.version, physical_validation=None,
                external_force_profile=(dict(shape='linear' if self.config['force_ramp_sec'] else 'constant',
                    duration_s=self.config['force_ramp_sec'],
                    start='first contact' if self.config['force_ramp_sec'] else 'execution_start',
                    shared_contact_gate_and_slew=True) if condition == 'A' else None),
                processing_markers=dict(mode='automatic_execution_start',
                    source='fresh Force mode acknowledgement after start alignment',
                    physical_processing_verified=False))
            metadata['artifact_paths'].extend([str(Path(__file__).with_name('e2_direct_abc.py')),
                str(Path(__file__).with_name('e2_protection.py'))])
            if self.config['run']['directory']:
                runtime_path = Path(self.config['run']['directory'])/'runtime.json'
                if runtime_path.is_file():
                    metadata['artifact_paths'].append(str(runtime_path))
        if self.protection is not None:
            source_required = self.protection.requires_source
            metadata.update(raw_sensor_available=None if source_required else False,
                raw_sensor_required_before_ready=source_required,
                force_feedback_topic='/ur10skku/currentF',
                sensor_acquisition_freshness_available=source_required,
                sampling=dict(feedback_max_hz=20.,commands_hz=125.,
                    source_timestamp=('host UDP reception via protection_feedback; sensor ADC time unavailable'
                        if source_required else 'headerless currentF; local ROS receipt freshness only')),
                safety_comparability=('shared A/B/C measured-wrench protection; see comparison hash'
                    if self.direct_abc else 'additional A protection revision; not identical to archived B/C'))
        log_root = (Path(self.config['run']['directory'])/'executor' if self.ablation else root/'logs/inference_metrics')
        if self.is_direct_run() and self.config['run']['directory']:
            log_root = Path(self.config['run']['directory'])/'executor'
        self.recorder = ExecutionRecorder(log_root,
            self.get_parameter('metrics_run_tag').value+'_executor',metadata,8192,warn=self.get_logger().warn)
        qos = QoSProfile(depth=1,reliability=ReliabilityPolicy.RELIABLE)
        self.pub_command = self.create_publisher(Float64MultiArray,'/ur10skku/cmdMotion',qos)
        self.pub_mode = self.create_publisher(String,'/ur10skku/cmdMode',10)
        self.pub_status = self.create_publisher(String,'/e2/executor_status',10)
        self.client = self.create_client(SingleArmCommand,'/singleArm_cmd/single_arm_command')
        self.create_subscription(Float64MultiArray,'/ur10skku/currentP',self.on_pose,qos)
        self.create_subscription(Float64MultiArray,'/ur10skku/currentF',self.on_force,qos)
        self.create_subscription(String,'/ur10skku/ctlMode',self.on_mode,qos)
        if self.protection is not None and self.protection.requires_source:
            self.create_subscription(String,ACQUISITION_TOPIC,self.on_acquisition,qos)
            self.create_subscription(String,PROVENANCE_TOPIC,self.on_wrench_provenance,qos)
        self.create_subscription(String,'/e2/plan',self.on_plan,10)
        self.create_subscription(String,'/e2/provider_control',self.on_provider_control,10)
        for name in ('abort','finish','processing_start','processing_end','phase_next'):
            self.create_service(Trigger,'~/'+name,lambda req,res,n=name:self.operator_event(n,res))
        self.create_timer(self.settings['control_period_s'],self.tick)
        self.create_timer(.1,self.status)
        self.event('executor_started',hardware_motion_started=False,transport=TRANSPORT_VERSION)
        if self.ablation or self.is_direct_run():
            self.event('run_start', run=self.config['run'])
        self.get_logger().info('E2 executor waiting for fresh robot feedback; R/T/C use the same timed transport.')

    def timing(self, source='e2_executor'):
        return stamp(self.get_clock().now().nanoseconds,source=source)

    def event(self, name, **details):
        context = dict(state=self.state, **self.force_selection_status())
        context.update(details)
        self.recorder.event(name,self.timing(),session_id=self.session,**context)

    def is_direct_run(self):
        return getattr(self, 'direct_a', False) or getattr(self, 'direct_abc', False)

    def uses_external_force(self):
        return (self.is_direct_run() or self.ablation) and self.config['condition'] == 'A'

    def force_selection_status(self):
        """Selected target before gate/slew; never a claim of applied force."""
        external = self.uses_external_force()
        return dict(processing=self.processing,
            force_source='external_F0' if external else 'provider',
            external_force_fz_N=(float(self.a_processing_force) if self.processing else 0.) if external else None,
            external_force_ramp_sec=(self.config['force_ramp_sec'] if external and self.is_direct_run() else None),
            external_force_ramped_fz_N=(self.engine.external_force_ramped_fz
                if external and self.is_direct_run() and self.processing else None),
            command_supply_active=self.state == 'running' and not self.engine.closed,
            controller_applied_force=None)

    def sampled(self, name, now):
        if now-self.sample_last.get(name,-float('inf')) < .05: return False
        self.sample_last[name]=now;return True

    def on_pose(self,msg):
        self.feedback_received['pose'] += 1
        values=np.asarray(msg.data,float)
        if len(values)<6 or not np.isfinite(values).all():
            if getattr(self,'protection',None) is not None:
                self.protection_failure('invalid pose feedback')
            elif self.state in ('running','returning'):self.request_stop('safety_stop',details='invalid pose')
            return
        self.pose,self.pose_at=values[:6].copy(),time.monotonic()
        if self.sampled('pose',self.pose_at):
            row=dict(self.timing('/ur10skku/currentP'),**pose_fields(values.tolist(),verified=True))
            self.recorder.emit('tcp_pose',row)

    def on_force(self,msg):
        self.feedback_received['force'] += 1
        values=np.asarray(msg.data,float)
        if len(values)<(6 if getattr(self,'protection',None) is not None else 3) or not np.isfinite(values).all():
            if getattr(self,'protection',None) is not None:
                self.protection_failure('invalid six-axis force feedback')
            elif self.state in ('running','returning'):self.request_stop('safety_stop',details='invalid force')
            return
        self.force,self.force_at=values.copy(),time.monotonic()
        if self.sampled('force',self.force_at):
            row=dict(self.timing('/ur10skku/currentF'),representation='filtered_republished_feedback',
                semantic_frame='robot_base',force_unit='N',torque_unit='Nm',raw_values=values.tolist(),
                correction_status=('source receipt provenance in protection_feedback events'
                    if getattr(self,'protection',None) is not None and self.protection.requires_source else
                    'deployed configuration; source acquisition timestamp unavailable'))
            row.update(zip(['fx','fy','fz','tx','ty','tz'],values.tolist()))
            self.recorder.emit('wrench',row)
        if (getattr(self,'protection',None) is not None and not self.protection.requires_source
                and self.state not in ('waiting_feedback','stopped','stop_failed')):
            self.protection_check(self.force_at)

    def on_mode(self,msg):
        self.mode,self.mode_at=msg.data,time.monotonic()

    def protection_failure(self, reason):
        self.protection.fault = self.protection.fault or str(reason)
        if self.protection_wait_reason != str(reason):
            self.event('protection_fault',reason=str(reason),state=self.state,
                       source_acquisition=('host UDP receive; sensor ADC timestamp unavailable'
                           if self.protection.requires_source else
                           'currentF ROS receipt; sensor acquisition timestamp unavailable'),
                       actual_pose=None if self.pose is None else self.pose.tolist(),
                       raw_sensor=None if self.protection.raw is None else
                           {k:v for k,v in self.protection.raw.items() if k!='wrench'},
                       controller_used=None if self.protection.base is None else
                           {k:v for k,v in self.protection.base.items() if k!='wrench'})
        self.protection_wait_reason = str(reason)
        if self.state not in ('waiting_feedback','stopped'):
            self.stop_failed_latched = True
            self.engine.stop()
            if self.state in ('stopping','stop_failed'):
                self.stop_reason = 'safety_stop'
            else:
                self.request_stop('safety_stop',details='protection: '+str(reason))

    def protection_check(self, now):
        protection = getattr(self,'protection',None)
        if protection is None:
            return True
        try:
            kwargs = {} if protection.requires_source else {'force': self.force}
            protection.check(now,self.get_clock().now().nanoseconds,self.pose,
                now-self.pose_at,now-self.force_at,now-self.mode_at,**kwargs)
            self.protection_wait_reason = None
            return True
        except ProtectionFault as exc:
            if self.state == 'waiting_feedback' and not protection.fault:
                if self.protection_wait_reason != str(exc):
                    self.event('protection_waiting',reason=str(exc),motion_allowed=False)
                self.protection_wait_reason = str(exc)
            else:
                self.protection_failure(str(exc))
            return False

    def protection_feedback(self, msg, raw):
        if self.protection is None or not self.protection.requires_source:
            return
        now = time.monotonic()
        try:
            obj = json.loads(msg.data)
            method = self.protection.ingest_raw if raw else self.protection.ingest_base
            method(obj,now,self.get_clock().now().nanoseconds)
            if self.sampled('raw_acquisition' if raw else 'wrench_provenance',now):
                self.event('protection_feedback',kind='raw_sensor' if raw else 'controller_used',
                           packet=obj,receipt_monotonic_s=now)
            if self.state not in ('waiting_feedback','stopped','stop_failed'):
                self.protection_check(now)
        except (ValueError,TypeError,KeyError) as exc:
            self.event('protection_invalid_packet',raw=raw,payload=msg.data)
            self.protection_failure(str(exc))

    def on_acquisition(self,msg):
        self.protection_feedback(msg,True)

    def on_wrench_provenance(self,msg):
        self.protection_feedback(msg,False)

    def status(self):
        body=dict(session_id=self.session,config_sha256=self.identity,clock_id=self.clock_identity,
            transport=TRANSPORT_VERSION,state=self.state,started_at=self.started_at,
            last_plan_id=self.engine.last_plan_id,stop_reason=self.stop_reason,
            log_dir=str(self.recorder.path) if self.recorder.path else None,
            sent_at=time.monotonic())
        body.update(self.force_selection_status(),
            last_operator_event=getattr(self,'last_operator_event',None),
            queue_cancel_ack=bool(getattr(self,'stop_ack',False)),
            physical_stop_verified=self.state=='stopped' and bool(getattr(self,'stop_ack',False)))
        if getattr(self,'protection',None) is not None:
            body.update(protection_version=self.protection.version,
                        protection_wait_reason=self.protection_wait_reason)
        if getattr(self,'e1_operator_automation',False):
            body.update(processing_marker_source='automatic_contact_proxy',
                        physical_processing_verified=False)
            key=(self.state,self.processing,self.stop_reason,body['queue_cancel_ack'])
            if key != self.last_console_status:
                self.last_console_status=key
                self.get_logger().info('[E1] state=%s processing_contact_proxy=%s '
                    'stop_reason=%s queue_cancel_ack=%s physical_stop_verified=%s '
                    'session=%s log=%s' % (self.state,self.processing,self.stop_reason,
                    body['queue_cancel_ack'],body['physical_stop_verified'],self.session,body['log_dir']))
        elif self.is_direct_run():
            key = (self.state, self.processing, self.stop_reason, self.protection_wait_reason)
            if key != self.last_console_status:
                self.last_console_status = key
                self.get_logger().info('[E2 %s] state=%s processing=%s force_source=%s F0=%s N protection=%s' %
                    (self.config['condition'], self.state, self.processing, body['force_source'],
                     body['external_force_fz_N'], self.protection_wait_reason))
        self.pub_status.publish(String(data=json.dumps(body)))

    def automatic_processing_marker(self, active, result=None, termination=None):
        """Observe the existing gate; never select targets or change the engine.

        E2 A uses self.processing to select external force, so this automation
        is restricted to the explicit E1 launch workflow.
        """
        if (not getattr(self,'e1_operator_automation',False) or self.ablation or
                bool(active) == self.processing):
            return
        self.processing=bool(active)
        if active:
            self.auto_processing_interval += 1
        else:
            self.processing_completed=True
        self.event('processing_start' if active else 'processing_end',
            source='automatic_contact_proxy', interval_id=self.auto_processing_interval,
            force_threshold_used=True, contact_on_N=self.settings['contact_on_N'],
            contact_off_N=self.settings['contact_off_N'],
            measured_fz_N=None if self.force is None else float(self.force[2]),
            provider_phase=None if result is None else result['phase'],
            termination=termination, interval_truncated=termination is not None,
            physical_processing_verified=False, evaluation_interval_verified=False,
            physical_contact_release='unknown')

    def on_provider_control(self,msg):
        try:
            d=json.loads(msg.data)
            if d.get('session_id')!=self.session:return
            if d.get('event')=='heartbeat':
                self.provider_at=time.monotonic();return
            if d.get('event')=='abort':self.request_stop(d.get('reason','manual_abort'))
        except (ValueError,TypeError) as exc:self.event('invalid_provider_control',error=str(exc))

    def on_plan(self,msg):
        try:
            d=json.loads(msg.data)
            if d.get('session_id')!=self.session:return
            if self.state not in ('ready','arming','running'):
                self.event('plan_rejected',reason='executor not accepting plans',state=self.state);return
            if not self.protection_check(time.monotonic()):return
            if (d.get('clock_id')!=self.clock_identity or d.get('config_sha256')!=self.identity or
                    d.get('method')!=self.method or d.get('frame')!='robot_base'):
                raise ExecutionFault('plan identity/frame/clock mismatch')
            reference=d['reference_xy_mm']
            if self.reference is not None and reference!=self.reference:
                raise ExecutionFault('frozen task reference changed')
            self.reference=reference
            p=TimedPlan(d['plan_id'],d['generated_at'],d['time'],d['action'],d['phase'],d['final'])
            preparation_started=time.monotonic()
            self.engine.accept(p,time.monotonic())
            preparation_ms=1000.*(time.monotonic()-preparation_started)
            self.provider_at=time.monotonic()
            self.event('plan_received',plan_id=p.plan_id,elapsed_duration_s=float(p.time[-1]),
                source_plan_time=p.generated_at,reference_xy_mm=reference,
                c_pose_conditioning=self.settings.get('c_pose_conditioning'),preparation_ms=preparation_ms,
                first_requested_pose=p.action[0,:6].tolist(),last_requested_pose=p.action[-1,:6].tolist())
            if self.state=='ready':
                if self.ablation:
                    self.event('reference_ready',reference_xy_mm=reference)
                    self.event('first_command_ready',plan_id=p.plan_id)
                self.arm_at=time.monotonic();self.state='arming';self.send_mode('Force')
        except Exception as exc:
            self.request_stop('error',details='invalid plan: '+str(exc))

    def send_mode(self,mode):
        self.pub_mode.publish(String(data=mode));self.last_mode_send=time.monotonic()
        self.event('controller_mode_requested',mode=mode,physical_mode_verified=False)

    def request_stop(self,reason,details=None,initial=False):
        if self.state in ('stopping','stopped','stop_failed'):return
        self.stop_reason=reason;self.stopping_initial=initial
        self.stop_started=time.monotonic();self.stop_ack=False
        self.state='stopping'
        if not initial:self.engine.stop()
        # Idling ignores cmdMotion in the inspected controller callback. This
        # also inhibits an outstanding PTP generator while its service is busy.
        # Position alone would still accept that generator's queued publishes.
        self.send_mode('Idling')
        self.verifier=StopVerifier(self.stop_started,
            window_s=self.settings['stop_window_s'],max_age_s=self.settings['feedback_max_age_s'])
        self.event('stop_requested',reason=reason,details=details,initial_reset=initial,
            physical_stop_verified=False,spindle_stop_commanded=False)
        if self.client.service_is_ready():
            req=SingleArmCommand.Request();req.command_mode='PTP9D_STREAM_STOP'
            self.pending=self.client.call_async(req)
            self.pending.add_done_callback(self.stop_response)
        else:
            self.event('stop_service_unavailable',controller_hold_mode_requested=True)
        if not initial:
            self.automatic_processing_marker(False,termination=reason)

    def stop_response(self,future):
        try:
            response=future.result()
            if not response.success:raise ExecutionFault(response.message)
            self.stop_ack=True
            self.event('queue_cancel_ack',physical_stop_verified=False)
            if self.ablation:self.event('stop_acknowledged',scope='queue cancellation only',physical_stop_verified=False)
            # Old queue/thread has joined. Select Position only after ack,
            # preventing outstanding old queued poses from defeating the hold.
            self.send_mode('Position')
            self.verifier=StopVerifier(time.monotonic(),window_s=self.settings['stop_window_s'],
                max_age_s=self.settings['feedback_max_age_s'])
        except Exception as exc:
            self.event('queue_cancel_error',error=str(exc));self.stop_ack=False

    def operator_event(self,name,response):
        def complete(accepted, message):
            self.last_operator_event=dict(name=name,accepted=accepted,message=message,
                                          monotonic_s=time.monotonic())
            self.event('operator_event_accepted' if accepted else 'operator_event_rejected',
                       operator_event=name,reason=message,**self.force_selection_status())
            self.status()
            response.success=accepted;response.message=message
            return response
        if name not in ('phase_next','processing_start','processing_end','finish','abort'):
            return complete(False,'Unknown operator event')
        if name == 'phase_next':
            phases = self.config.get('protocol', {}).get('phase_ids', [])
            if self.state!='running' or not self.ablation or not self.processing or self.phase_index+1 >= len(phases):
                return complete(False,'No next reviewed processing phase in running state')
            self.phase_index += 1
            self.event('phase_or_pass_marker',phase_id=phases[self.phase_index],source='operator')
            return complete(True,'Phase marker recorded')
        if name in ('processing_start','processing_end'):
            if getattr(self, 'direct_abc', False):
                return complete(False,'Matched A/B/C starts automatically; use finish or abort to end the attempt')
            if getattr(self,'e1_operator_automation',False):
                return complete(False,'E1 contact-proxy markers are automatic; use finish to end the run')
            if self.state!='running':
                return complete(False,'Processing events require running state')
            if (name=='processing_start')==self.processing:
                return complete(False,'Duplicate/out-of-order processing event')
            if name=='processing_start' and (self.ablation or self.is_direct_run()) and getattr(self,'processing_completed',False):
                return complete(False,'Processing interval already ended; no restart within this attempt')
            self.processing=name=='processing_start'
            if not self.processing:self.processing_completed=True
            self.event(name,source='operator',force_threshold_used=False)
            if (self.ablation or self.is_direct_run()) and self.processing:
                self.phase_index = 0
                self.event('approach_end',source='operator')
                self.event('phase_or_pass_marker',phase_id=self.config['protocol']['phase_ids'][0],source='operator')
        else:
            if name=='finish' and self.state!='running':
                return complete(False,'finish requires running state')
            if name=='abort' and self.state in ('stopping','stopped','stop_failed'):
                return complete(False,'Stop already requested; check physical_stop_verified and stop state')
            if self.ablation or self.is_direct_run():
                self.event('finish_requested' if name=='finish' else 'abort_requested',source='operator',
                           processing_end_recorded=not self.processing,quality_verified=None)
            self.request_stop('manual_abort' if name=='abort' else 'operator_finish')
        return complete(True,'Event accepted; target selection/stop request only. Check executor status for physical stop.')

    def log_command(self,stage,values,result,now):
        position_mode = result.get('controller_mode') == 'Position'
        row=dict(self.timing('/ur10skku/cmdMotion' if stage=='node_sent' else 'e2_timed_sampler'),
            command_stage=stage,command_mode='cmdMotion',command_id=self.command_id,
            inference_id=result['plan_id'],plan_id=result['plan_id'],action_index=result['action_index'],
            semantic_frame='pose:robot_base; force:controller_config_dependent',
            position_unit='mm',orientation_unit='rotvec_rad',
            force_unit=None if position_mode and stage=='node_sent' else 'N',raw_values=values.tolist(),
            execution_status='sent_not_verified_executed' if stage=='node_sent' else 'requested_not_sent',
            limited=result['limited'],limiting_reason=('common contact gate + legacy gain/rate limits' +
                (' + pose smoothing/handover/acceleration' if result.get('conditioning_profile') else '')),
            details=dict(elapsed_s=result['elapsed_s'],contact=result['contact'],phase=result['phase'],dt_s=result['dt_s'],
                endpoint_lateness_s=result['endpoint_lateness_s'],
                conditioning_profile=result.get('conditioning_profile'),
                force_source=result.get('force_source'),processing=self.processing,
                phase_id=(self.config['protocol']['phase_ids'][self.phase_index]
                          if (self.ablation or self.is_direct_run()) and self.processing and self.phase_index>=0 else None),
                pose_age_s=result.get('pose_age_s'),force_age_s=result.get('force_age_s'),
                gate_transition=result.get('contact_transition'),gate_reason=result.get('gate_reason'),
                force_ramp_start=result.get('force_ramp_start'),
                force_ramp_sec=result.get('force_ramp_sec'),
                force_ramp_fraction=result.get('force_ramp_fraction'),
                controller_applied_force=None,
                controller_mode=result.get('controller_mode','Force'),
                force_semantics='Position mode sends pose6 only' if position_mode else 'signed controller force target'))
        row.update(zip(['x','y','z','rx','ry','rz','fx','fy','fz'],values.tolist()))
        self.recorder.emit('commands',row)

    def start_return_home(self, now):
        # The work pose's base Z can differ under force control. Work arrival
        # checks XY/orientation; retraction starts at the actual held TCP Z.
        if (self.final_target is None or self.reference is None or
                np.linalg.norm(self.pose[:2]-self.final_target[:2]) > self.settings['completion_position_tolerance_mm'] or
                (Rotation.from_rotvec(self.pose[3:])*Rotation.from_rotvec(self.final_target[3:]).inv()).magnitude()
                    > self.settings['completion_rotation_tolerance_rad']):
            raise ExecutionFault('Cannot return automatically: work endpoint XY/orientation not reached')
        home = np.asarray(self.config['common']['demo_start_pose6'], float).copy()
        home[:2] += np.asarray(self.reference, float)
        self.return_home = ReturnHome(self.settings, self.return_spec, self.pose, home, now)
        self.state = 'returning'
        if self.processing:
            self.processing=False
            self.event('processing_end',source='work_trajectory_end')
        self.event('home_return_started',home_pose_base=home.tolist(),
            source='common.demo_start_pose6 plus frozen stain XY',
            queue_cancel_verified=True,controller_hold_verified=True)
        self.status()

    def tick_return_home(self, now):
        try:
            if self.recorder.write_errors:raise ExecutionFault('execution logger write failed')
            if now-self.provider_at>self.settings['provider_timeout_s']:
                raise ExecutionFault('provider heartbeat lost during home return')
            result=self.return_home.tick(now,self.pose,self.force,self.pose_at,self.force_at,self.mode,self.mode_at)
            while self.return_home.events:
                event=self.return_home.events.pop(0);self.event(event.pop('event'),**event)
            if result['done']:
                self.final_target=self.return_home.home.copy()
                self.request_stop('home_return_complete');return
            self.command_id+=1
            # Position callback consumes pose6; never label unused zero force
            # columns as a force command sent to the controller.
            self.pub_command.publish(Float64MultiArray(data=result['sent'][:6].tolist()))
            for stage,key in [('time_sampled','requested'),('contact_gated','gated'),('node_sent','sent')]:
                values=result[key][:6] if stage=='node_sent' else result[key]
                self.log_command(stage,values,result,now)
        except Exception as exc:self.request_stop('safety_stop',details=str(exc))

    def tick(self):
        now=time.monotonic()
        if self.state=='waiting_feedback':
            if (self.pose is not None and self.force is not None and self.client.service_is_ready() and
                    max(now-self.pose_at,now-self.force_at,now-self.mode_at)<=self.settings['feedback_max_age_s'] and
                    self.protection_check(now)):
                self.request_stop('startup_reset',initial=True)
            return
        if self.state in ('ready','arming','running','returning','stopping','stop_failed'):
            if not self.protection_check(now) and self.state not in ('stopping','stop_failed'):
                return
        if self.state in ('stopping','stop_failed'):
            if not self.stop_ack and now-self.last_mode_send>=.1:self.send_mode('Idling')
            if self.stop_ack and self.pose is not None and self.verifier.observe(now,self.pose,self.mode,self.mode_at,self.pose_at):
                if self.ablation and not self.stopping_initial:
                    self.event('physical_hold_verified',fresh_feedback=True,mode=self.mode)
                if self.stopping_initial and not self.shutdown_requested and not self.stop_failed_latched:
                    self.state='ready';self.event('executor_ready',stale_queue_cleared=True,stationary_feedback_verified=True)
                else:
                    if (getattr(self,'return_spec',{}).get('enabled') and self.stop_reason=='trajectory_end' and
                            not self.shutdown_requested and not self.stop_failed_latched):
                        try:
                            self.start_return_home(now)
                            return
                        except Exception as exc:
                            self.stop_reason='home_return_failed'
                            self.event('home_return_not_started',reason=str(exc))
                    self.state='stopped'
                    verified_target = (self.final_target is not None and
                        np.linalg.norm(self.pose[:3]-self.final_target[:3])<=self.settings['completion_position_tolerance_mm'] and
                        (Rotation.from_rotvec(self.pose[3:])*Rotation.from_rotvec(self.final_target[3:]).inv()).magnitude()<=self.settings['completion_rotation_tolerance_rad'])
                    returned=getattr(self,'return_home',None)
                    released=(returned is not None and returned.complete and returned.released and
                        0 <= now-self.force_at <= self.settings['feedback_max_age_s'] and
                        abs(self.force[2]) <= self.return_spec['release_abs_fz_N'])
                    normal=(not self.stop_failed_latched and self.started_at is not None and
                        (self.stop_reason=='operator_finish' or (self.stop_reason=='trajectory_end' and verified_target) or
                         (self.stop_reason=='home_return_complete' and verified_target and released)))
                    if self.processing:
                        self.processing=False
                        self.event('processing_end',source='executor_stop',termination=self.stop_reason)
                    self.event('normal_completion' if normal else self.stop_reason,
                        controller_hold_verified=True,queue_cancel_verified=True,
                        final_target_reached=bool(verified_target),retract_complete=bool(released and verified_target),
                        home_pose_reached=bool(released and verified_target),
                        physical_contact_release='verified_clearance_and_abs_Fz' if released else 'unknown',
                        spindle_state='operator controlled; not changed')
                self.status()
            elif now-self.stop_started > self.settings['stop_timeout_s'] and self.state!='stop_failed':
                self.state='stop_failed';self.stop_failed_latched=True
                self.event('stop_not_verified',queue_cancel_ack=self.stop_ack,mode=self.mode,
                    feedback_age_s=now-self.pose_at,physical_stop_verified=False)
                self.get_logger().error('E2 stop NOT verified. Check robot/driver and use the physical stop if needed. No automatic resume.')
            return
        if self.state=='returning':
            self.tick_return_home(now);return
        if self.state in ('ready','arming') and np.isfinite(self.provider_at) and now-self.provider_at>self.settings['provider_timeout_s']:
            self.request_stop('error',details='provider heartbeat lost during start alignment');return
        if self.state=='arming':
            if now-self.arm_at > self.settings['mode_timeout_s']:
                self.request_stop('error',details='Force mode acknowledgement timeout');return
            if (self.mode=='Force' and self.mode_at>self.arm_at and
                    max(now-self.pose_at,now-self.force_at,now-self.mode_at)<=self.settings['feedback_max_age_s']):
                self.engine.start(now,self.pose);self.started_at=now;self.state='running'
                self.event('execution_start',processing_started=False,monotonic_start=now)
                if self.ablation:self.event('approach_start',source='execution_state')
                if (self.is_direct_run() and not self.processing and
                        not getattr(self, 'processing_completed', False)):
                    self.processing = True
                    self.phase_index = 0
                    self.event('processing_start', source='execution_start', automatic=True,
                        force_threshold_used=False, physical_processing_verified=False)
                    self.event('phase_or_pass_marker',
                        phase_id=self.config['protocol']['phase_ids'][0], source='execution_start')
                self.status()
            return
        if self.state!='running':return
        try:
            if self.recorder.write_errors:
                raise ExecutionFault('execution logger write failed')
            if now-self.provider_at>self.settings['provider_timeout_s']:
                raise ExecutionFault('provider heartbeat lost')
            if self.mode!='Force' or now-self.mode_at>self.settings['feedback_max_age_s']:
                raise ExecutionFault('controller Force mode lost or stale')
            external = ((self.a_processing_force if self.processing else 0.)
                        if self.uses_external_force() else None)
            result=self.engine.tick(now,self.pose,self.force,now-self.pose_at,now-self.force_at,
                external_force_fz=external,
                external_force_ramp_sec=(self.config['force_ramp_sec'] if self.is_direct_run() else 0.))
            if result is None:return
            if 'end' in result:
                self.final_target=result.get('last_target')
                self.request_stop(result['end']);return
            if result.get('force_ramp_started_now'):
                self.event('force_ramp_started', source='first_contact',
                    measured_fz_N=float(self.force[2]), ramp_sec=self.config['force_ramp_sec'],
                    target_force_N=self.a_processing_force,
                    physical_contact_verified=False)
            self.automatic_processing_marker(result['contact'],result=result)
            if self.last_phase!=result['phase']:
                self.event('provider_phase',phase=result['phase'],processing_interval=(
                    'automatic_execution_start' if self.is_direct_run() else
                    'automatic_contact_proxy' if getattr(self,'e1_operator_automation',False)
                    else 'operator events'));self.last_phase=result['phase']
            self.command_id+=1
            self.pub_command.publish(Float64MultiArray(data=result['sent'].tolist()))
            if self.command_id==1:self.event('first_command_sent',processing_started=(
                self.processing if self.is_direct_run() else False))
            stages=[('time_sampled','requested'),('contact_gated','gated'),('node_sent','sent')]
            if result.get('conditioning_profile'):
                stages.insert(1,('pose_conditioned','conditioned'))
            for stage,key in stages:
                self.log_command(stage,result[key],result,now)
        except Exception as exc:self.request_stop('safety_stop',details=str(exc))

    def request_shutdown(self):
        if not self.shutdown_requested:
            self.shutdown_requested=True;self.shutdown_at=time.monotonic()
            self.request_stop('manual_abort')

    def may_exit(self):
        if (getattr(self,'e1_operator_automation',False) and self.state=='stopped' and
                self.stop_ack):
            return True
        if not self.shutdown_requested:return False
        if self.state=='stopped':return True
        # Never represent shutdown deadline as verified robot stop. Keep Idling
        # requested; report a failure if communication/feedback cannot verify it.
        if time.monotonic()-self.shutdown_at>self.settings['stop_timeout_s']+2.:
            self.event('shutdown_without_verified_stop',state=self.state)
            return True
        return False

    def destroy_node(self):
        if self.ablation:
            self.event('run_closed',state=self.state,feedback_callbacks_received=self.feedback_received,
                record_limit_hz=20.,source_acquisition_rate=None,DDS_loss_count=None,
                processing_complete=None,quality_verified=None)
        self.recorder.close()
        self.lock_file.close()
        return super().destroy_node()


def main(args=None):
    rclpy.init(args=args,signal_handler_options=SignalHandlerOptions.NO)
    node=None
    try:
        node=E2Executor()
        signal.signal(signal.SIGINT,lambda *_:node.request_shutdown())
        signal.signal(signal.SIGTERM,lambda *_:node.request_shutdown())
        while rclpy.ok() and not node.may_exit():rclpy.spin_once(node,timeout_sec=.02)
    except Exception as exc:
        if node is not None and rclpy.ok():
            node.get_logger().error('Executor exception; requesting controlled stop: '+str(exc))
            node.request_stop('error',details=str(exc))
            node.request_shutdown()
            while rclpy.ok() and not node.may_exit():
                rclpy.spin_once(node,timeout_sec=.02)
        raise
    finally:
        if node is not None:node.destroy_node()
        rclpy.try_shutdown()


if __name__=='__main__':main()
