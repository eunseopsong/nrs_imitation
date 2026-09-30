"""A runtime protection using explicit limits and source packet provenance.

No default force/torque limits, F0-derived limits, ROS I/O, or approval writes.
This is software monitoring, not a certified independent robot safety system.
"""
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

VERSION = 'e2_source_wrench_workspace_v1'
ACQUISITION_TOPIC = '/ur10skku/ft_acquisition'
PROVENANCE_TOPIC = '/ur10skku/currentF_provenance'
RAW_SCHEMA = 'aft_ether_acquisition_v1'
BASE_SCHEMA = 'y2_wrench_provenance_v1'
LIMIT_KEYS = ('measured_force_abs_limits_N', 'measured_torque_abs_limits_Nm',
              'sensor_raw_force_abs_limits_N', 'sensor_raw_torque_abs_limits_Nm')
CAPABILITIES = frozenset(('measured_wrench', 'source_freshness', 'raw_sensor_envelope',
                          'actual_tcp_workspace', 'startup_alignment_coverage'))


class ProtectionFault(ValueError):
    pass


def record_from_config(config):
    return json.loads(Path(config['pilot']['commissioning_record']['path']).read_text())


class ProtectionMonitor:
    def __init__(self, record, settings):
        self.frame = record['measured_wrench_frame']
        if self.frame not in ('sensor', 'robot_base', 'controller_tcp'):
            raise ProtectionFault('unsupported protection wrench frame')
        self.limits = {}
        for key in LIMIT_KEYS:
            value = record.get(key)
            if (not isinstance(value, list) or len(value) != 3 or
                    any(isinstance(v, bool) or not isinstance(v, (int, float)) for v in value)):
                raise ProtectionFault(key + ': explicit per-axis limits required')
            arr = np.asarray(value, float)
            if not np.isfinite(arr).all() or np.any(arr <= 0):
                raise ProtectionFault(key + ': positive finite limits required')
            self.limits[key] = arr
        self.max_age = record['acquisition_max_age_s']
        if (isinstance(self.max_age, bool) or not isinstance(self.max_age, (int, float)) or
                not np.isfinite(self.max_age) or self.max_age <= 0):
            raise ProtectionFault('explicit acquisition_max_age_s required')
        self.feedback_age = settings['feedback_max_age_s']
        self.workspace = settings['workspace_limits']
        self.raw = self.base = None
        self.raw_at = self.base_at = -float('inf')
        self.anchor = None
        self.armed = False
        self.fault = None

    def fail(self, reason):
        self.fault = self.fault or reason
        raise ProtectionFault(self.fault)

    def _wrench(self, value):
        try:
            a = np.asarray(value, float)
        except (ValueError, TypeError):
            self.fail('malformed wrench')
        if a.shape != (6,) or not np.isfinite(a).all():
            self.fail('invalid six-axis wrench')
        return a

    def _stamp(self, obj, now_ros_ns):
        stamp = obj.get('source_ros_ns')
        if isinstance(stamp, bool) or not isinstance(stamp, int) or stamp <= 0:
            self.fail('invalid source receive timestamp')
        age = (now_ros_ns-stamp)/1e9
        if age < 0 or age > self.max_age:
            self.fail('source timestamp stale or clocks incompatible')
        return stamp

    def _metadata(self, obj, frame):
        if (obj.get('wrench_frame') != frame or obj.get('force_unit') != 'N' or
                obj.get('torque_unit') != 'Nm' or
                obj.get('timestamp_origin') != 'host_udp_receive_estimate'):
            self.fail('producer frame/unit/source-clock contract mismatch')

    def ingest_raw(self, obj, now, now_ros_ns):
        if self.fault:
            self.fail(self.fault)
        if not isinstance(obj, dict) or obj.get('schema') != RAW_SCHEMA:
            self.fail('raw sensor producer contract missing')
        self._metadata(obj, 'sensor')
        if obj.get('valid') is not True:
            self.fail('invalid Ethernet sensor packet')
        sequence = obj.get('sequence'); boot = obj.get('boot_id')
        if (isinstance(sequence, bool) or not isinstance(sequence, int) or sequence <= 0 or
                not isinstance(boot, str) or not boot):
            self.fail('invalid sensor sequence/boot identity')
        stamp = self._stamp(obj, now_ros_ns)
        if self.raw is not None:
            if boot != self.raw['boot_id'] or sequence < self.raw['sequence']:
                self.fail('sensor producer restarted or sequence regressed')
            if sequence == self.raw['sequence']:
                if stamp != self.raw['source_ros_ns']:
                    self.fail('cached sensor packet was restamped')
                return False  # never refresh freshness from duplicate packets
            if stamp <= self.raw['source_ros_ns']:
                self.fail('sensor source timestamp failed to advance')
        wrench = self._wrench(obj.get('sensor_wrench'))
        if (np.any(np.abs(wrench[:3]) >= self.limits['sensor_raw_force_abs_limits_N']) or
                np.any(np.abs(wrench[3:]) >= self.limits['sensor_raw_torque_abs_limits_Nm'])):
            self.fail('raw sensor envelope exceeded (before zero/gravity/filter)')
        if self.armed and obj.get('initialized') is not True:
            self.fail('sensor initialization lost during execution')
        self.raw = dict(obj, wrench=wrench)
        self.raw_at = now
        return True

    def ingest_base(self, obj, now, now_ros_ns):
        if self.fault:
            self.fail(self.fault)
        if (not isinstance(obj, dict) or obj.get('schema') != BASE_SCHEMA or
                obj.get('source_contract') is not True):
            self.fail('controller feedback lacks source acquisition provenance')
        self._metadata(obj, 'robot_base')
        if obj.get('valid') is not True:
            self.fail('invalid controller wrench')
        stamp = self._stamp(obj, now_ros_ns)
        wrench = self._wrench(obj.get('base_wrench'))
        if self.base is not None and stamp < self.base['source_ros_ns']:
            self.fail('controller source timestamp regressed')
        if self.base is None or stamp != self.base['source_ros_ns']:
            self.base_at = now
        self.base = dict(obj, wrench=wrench)

    def check(self, now, now_ros_ns, pose, pose_age, force_age, mode_age):
        if self.fault:
            self.fail(self.fault)
        if self.raw is None or self.base is None:
            raise ProtectionFault('waiting for source sensor/controller provenance')
        if self.raw.get('initialized') is not True:
            raise ProtectionFault('waiting for existing sensor initialization')
        for obj, at, name in ((self.raw, self.raw_at, 'sensor'), (self.base, self.base_at, 'controller')):
            if now-at < 0 or now-at > self.max_age:
                self.fail(name + ' acquisition stopped advancing')
            self._stamp(obj, now_ros_ns)
        ages = np.asarray([pose_age, force_age, mode_age], float)
        if not np.isfinite(ages).all() or np.any(ages < 0) or np.any(ages > self.feedback_age):
            raise ProtectionFault('pose/force/mode feedback stale during protected phase')
        p = np.asarray(pose, float)
        if p.shape != (6,) or not np.isfinite(p).all():
            self.fail('invalid actual TCP pose')
        wrench = self.base['wrench'].copy()
        if self.frame == 'sensor':
            wrench = self.raw['wrench']
        elif self.frame == 'controller_tcp':
            rot = Rotation.from_rotvec(p[3:]).inv()
            wrench = np.r_[rot.apply(wrench[:3]), rot.apply(wrench[3:])]
        # Rotation of axes only: torque remains at the source sensor origin.
        if (np.any(np.abs(wrench[:3]) >= self.limits['measured_force_abs_limits_N']) or
                np.any(np.abs(wrench[3:]) >= self.limits['measured_torque_abs_limits_Nm'])):
            self.fail('measured force/torque abort threshold reached')
        if self.anchor is None:
            self.anchor = p[:3].copy()  # one anchor covers HOME alignment through termination
        delta = p[:3]-self.anchor
        if (np.linalg.norm(delta[:2]) > self.workspace['max_xy_from_start_mm'] or
                -delta[2] > self.workspace['max_z_down_from_start_mm'] or
                delta[2] > self.workspace['max_z_up_from_start_mm']):
            self.fail('actual TCP outside common workspace from startup anchor')
        self.armed = True
        return dict(version=VERSION, raw_sequence=self.raw['sequence'],
                    source_ros_ns=self.raw['source_ros_ns'], anchor_mm=self.anchor.tolist())
