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


class E2Executor(Node):
    def __init__(self):
        super().__init__('e2_executor')
        for name, default in [('e2_config',''),('execution_method','il'),('e2_enable_hardware',False),
                              ('e2_session_id',''),('metrics_run_tag','E2'),('act_root','/home/eunseop/nrs_imitation')]:
            self.declare_parameter(name,default)
        self.config_path = self.get_parameter('e2_config').value
        self.method = self.get_parameter('execution_method').value
        self.config = load_config(self.config_path)
        self.ablation = is_ablation(self.config)
        blockers = hardware_blockers(self.config,self.method,self.get_parameter('e2_enable_hardware').value)
        if blockers: raise RuntimeError('E2 executor preflight: '+ '; '.join(blockers))
        self.a_processing_force = (prepare_A_force(self.config)
            if self.ablation and self.config['condition'] == 'A' else None)
        self.session = self.get_parameter('e2_session_id').value
        if not self.session: raise ValueError('A unique e2_session_id is required')
        self.lock_file = open('/tmp/nrs_e2_executor_ur10skku.lock','a')
        fcntl.flock(self.lock_file,fcntl.LOCK_EX | fcntl.LOCK_NB)
        self.identity = file_hash(self.config_path)
        self.clock_identity = clock_id()
        self.settings = executor_settings(self.config, self.method)
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
        if self.ablation:
            metadata.update(experiment_id='E2', method=self.config['condition'],
                execution_contract_hash=self.config['execution_contract_hash'],
                run=self.config.get('run'), predicted_force='not_predicted' if self.config['condition']=='A' else 'policy',
                force_sign_provenance=self.config['calibration'],
                trial_stage=self.config.get('trial_stage', 'main'),
                external_F0=self.config['f0'], commissioning=self.config.get('pilot'),
                warnings=['Filtered base wrench is not automatically calibrated contact normal force',
                          'sent target is not controller-applied target'])
        log_root = (Path(self.config['run']['directory'])/'executor' if self.ablation else root/'logs/inference_metrics')
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
        self.create_subscription(String,'/e2/plan',self.on_plan,10)
        self.create_subscription(String,'/e2/provider_control',self.on_provider_control,10)
        for name in ('abort','finish','processing_start','processing_end','phase_next'):
            self.create_service(Trigger,'~/'+name,lambda req,res,n=name:self.operator_event(n,res))
        self.create_timer(self.settings['control_period_s'],self.tick)
        self.create_timer(.1,self.status)
        self.event('executor_started',hardware_motion_started=False,transport=TRANSPORT_VERSION)
        if self.ablation:
            self.event('run_start', run=self.config['run'])
        self.get_logger().info('E2 executor waiting for fresh robot feedback; R/T/C use the same timed transport.')

    def timing(self, source='e2_executor'):
        return stamp(self.get_clock().now().nanoseconds,source=source)

    def event(self, name, **details):
        self.recorder.event(name,self.timing(),session_id=self.session,**details)

    def sampled(self, name, now):
        if now-self.sample_last.get(name,-float('inf')) < .05: return False
        self.sample_last[name]=now;return True

    def on_pose(self,msg):
        self.feedback_received['pose'] += 1
        values=np.asarray(msg.data,float)
        if len(values)<6 or not np.isfinite(values).all():
            if self.state in ('running','returning'):self.request_stop('safety_stop',details='invalid pose')
            return
        self.pose,self.pose_at=values[:6].copy(),time.monotonic()
        if self.sampled('pose',self.pose_at):
            row=dict(self.timing('/ur10skku/currentP'),**pose_fields(values.tolist(),verified=True))
            self.recorder.emit('tcp_pose',row)

    def on_force(self,msg):
        self.feedback_received['force'] += 1
        values=np.asarray(msg.data,float)
        if len(values)<3 or not np.isfinite(values).all():
            if self.state in ('running','returning'):self.request_stop('safety_stop',details='invalid force')
            return
        self.force,self.force_at=values.copy(),time.monotonic()
        if self.sampled('force',self.force_at):
            row=dict(self.timing('/ur10skku/currentF'),representation='filtered_republished_feedback',
                semantic_frame='robot_base',force_unit='N',torque_unit='Nm',raw_values=values.tolist(),
                correction_status='deployed configuration; source acquisition timestamp unavailable')
            row.update(zip(['fx','fy','fz','tx','ty','tz'],values.tolist()))
            self.recorder.emit('wrench',row)

    def on_mode(self,msg):
        self.mode,self.mode_at=msg.data,time.monotonic()

    def status(self):
        body=dict(session_id=self.session,config_sha256=self.identity,clock_id=self.clock_identity,
            transport=TRANSPORT_VERSION,state=self.state,started_at=self.started_at,
            last_plan_id=self.engine.last_plan_id,stop_reason=self.stop_reason,
            log_dir=str(self.recorder.path) if self.recorder.path else None,
            sent_at=time.monotonic())
        self.pub_status.publish(String(data=json.dumps(body)))

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
        if name == 'phase_next':
            phases = self.config.get('protocol', {}).get('phase_ids', [])
            if not self.ablation or not self.processing or self.phase_index+1 >= len(phases):
                response.success=False;response.message='No next reviewed processing phase';return response
            self.phase_index += 1
            self.event('phase_or_pass_marker',phase_id=phases[self.phase_index],source='operator')
            response.success=True;response.message='Phase marker recorded';return response
        if name in ('processing_start','processing_end'):
            if self.state!='running':
                response.success=False;response.message='Processing events require running state';return response
            if (name=='processing_start')==self.processing:
                response.success=False;response.message='Duplicate/out-of-order processing event';return response
            self.processing=name=='processing_start'
            self.event(name,source='operator',force_threshold_used=False)
            if self.ablation and self.processing:
                self.phase_index = 0
                self.event('approach_end',source='operator')
                self.event('phase_or_pass_marker',phase_id=self.config['protocol']['phase_ids'][0],source='operator')
        else:
            if name=='finish' and self.state!='running':
                response.success=False;response.message='finish requires running state';return response
            if self.ablation:
                self.event('finish_requested' if name=='finish' else 'abort_requested',source='operator',
                           processing_end_recorded=not self.processing,quality_verified=None)
            self.request_stop('manual_abort' if name=='abort' else 'operator_finish')
        response.success=True;response.message='Event accepted; stop completion is reported separately in executor status/log'
        return response

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
                          if self.ablation and self.processing and self.phase_index>=0 else None),
                pose_age_s=result.get('pose_age_s'),force_age_s=result.get('force_age_s'),
                gate_transition=result.get('contact_transition'),gate_reason=result.get('gate_reason'),
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
            self.event('processing_end',source='work_trajectory_end');self.processing=False
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
                    max(now-self.pose_at,now-self.force_at,now-self.mode_at)<=self.settings['feedback_max_age_s']):
                self.request_stop('startup_reset',initial=True)
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
                        self.event('processing_end',source='executor_stop',termination=self.stop_reason);self.processing=False
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
                        if self.ablation and self.config['condition']=='A' else None)
            result=self.engine.tick(now,self.pose,self.force,now-self.pose_at,now-self.force_at,
                                    external_force_fz=external)
            if result is None:return
            if 'end' in result:
                self.final_target=result.get('last_target')
                self.request_stop(result['end']);return
            if self.last_phase!=result['phase']:
                self.event('provider_phase',phase=result['phase'],processing_interval='operator events');self.last_phase=result['phase']
            self.command_id+=1
            self.pub_command.publish(Float64MultiArray(data=result['sent'].tolist()))
            if self.command_id==1:self.event('first_command_sent',processing_started=False)
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
