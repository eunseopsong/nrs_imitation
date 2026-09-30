"""Offline analysis of an unloaded F0 observation bag. Never controls a robot."""
from bisect import bisect_right
from collections import Counter
from datetime import datetime
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

TOPICS = {'/ur10skku/currentP': 'std_msgs/msg/Float64MultiArray',
          '/ur10skku/currentF': 'std_msgs/msg/Float64MultiArray',
          '/ur10skku/ctlMode': 'std_msgs/msg/String'}


def statistics(values):
    a = np.asarray(values, dtype=float)
    if len(a) == 0:
        return None
    return dict(count=len(a), mean=a.mean(axis=0).tolist(), std=a.std(axis=0).tolist(),
                min=a.min(axis=0).tolist(), max=a.max(axis=0).tolist())


def summarize_records(records, max_pose_age_s=.2):
    """Use bag receipt time and the most recent preceding actual TCP pose."""
    poses, forces, modes = [], [], []
    malformed = Counter()
    for topic, ns, values in records:
        if topic == '/ur10skku/ctlMode':
            modes.append((int(ns), str(values)))
            continue
        if topic not in TOPICS:
            continue
        a = np.asarray(values, float)
        required = 6 if topic.endswith('currentP') else 3
        if a.ndim != 1 or len(a) < required or not np.isfinite(a[:required]).all():
            malformed[topic] += 1
            continue
        target = poses if topic.endswith('currentP') else forces
        target.append((int(ns), a[:required]))
    poses.sort(key=lambda x: x[0]); forces.sort(key=lambda x: x[0]); modes.sort()
    pt = [p[0] for p in poses]; mt = [m[0] for m in modes]
    origin = min([x[0] for x in poses+forces], default=0)
    rows, tcp = [], []
    for ns, f in forces:
        index = bisect_right(pt, ns)-1
        pose_age = (ns-pt[index])/1e9 if index >= 0 else None
        valid = pose_age is not None and 0 <= pose_age <= max_pose_age_s
        p = poses[index][1] if valid else np.full(6, np.nan)
        transformed = Rotation.from_rotvec(p[3:6]).inv().apply(f) if valid else np.full(3, np.nan)
        if valid:
            tcp.append(transformed)
        mi = bisect_right(mt, ns)-1
        row = dict(bag_receipt_ns=ns, elapsed_s=(ns-origin)/1e9,
            pose_age_s=pose_age, fresh_pose=bool(valid), controller_mode=modes[mi][1] if mi >= 0 else '')
        row.update(zip(['Fx_base_N','Fy_base_N','Fz_base_N'], f.tolist()))
        row.update(zip(['x_mm','y_mm','z_mm','rx_rad','ry_rad','rz_rad'],
                       [float(x) if np.isfinite(x) else None for x in p]))
        row.update(zip(['Fx_tcp_N','Fy_tcp_N','Fz_tcp_N'],
                       [float(x) if np.isfinite(x) else None for x in transformed]))
        rows.append(row)
    gaps = np.diff([f[0] for f in forces])/1e9
    positions = np.asarray([p[1] for p in poses])
    excursion = None
    if len(positions):
        excursion = dict(max_translation_from_first_mm=float(np.linalg.norm(positions[:,:3]-positions[0,:3],axis=1).max()),
            max_rotation_from_first_rad=float((Rotation.from_rotvec(positions[:,3:6])*
                Rotation.from_rotvec(positions[0,3:6]).inv()).magnitude().max()))
    summary = dict(capture_status='recorded' if len(forces)>1 and len(tcp)>1 else 'insufficient_feedback',
        valid_pose_messages=len(poses), valid_force_messages=len(forces), fresh_force_pose_pairs=len(tcp),
        malformed_messages=dict(malformed), controller_modes=sorted({m[1] for m in modes}),
        force_recorded_span_s=(forces[-1][0]-forces[0][0])/1e9 if len(forces)>1 else 0.,
        max_force_receipt_gap_s=float(gaps.max()) if len(gaps) else None,
        force_base_N=statistics([f[1] for f in forces]), force_tcp_N=statistics(tcp),
        actual_pose_excursion=excursion, pose_pair_max_age_s=max_pose_age_s,
        phase='operator_prepared_unloaded_home', unloaded_state_independently_verified=False,
        frame_basis='Inspected currentF source publishes base vector; TCP conversion uses currentP rotvec inverse. Active sensor calibration not certified by this capture.',
        timestamp_basis='rosbag receipt time; headerless acquisition timestamps unknown',
        new_zero_subtracted=False, calibration_verified=False, force_axis_sign_validated=False,
        validated_constant_force_range_N=None, F0_frozen=False,
        robot_commands_sent_by_tool=False,
        scope='Unloaded baseline and actual-pose transform diagnostics only; does not apply or validate 23 N.')
    return summary, rows


def analyze_bag(folder):
    """Import ROS deserialization only for offline file reading; no rclpy.init."""
    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message
    folder = Path(folder)
    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=str(folder/'bag'), storage_id='sqlite3'),
                rosbag2_py.ConverterOptions(input_serialization_format='cdr', output_serialization_format='cdr'))
    actual = {topic.name: topic.type for topic in reader.get_all_topics_and_types()}
    for name, kind in actual.items():
        if name in TOPICS and TOPICS[name] != kind:
            raise ValueError('Unexpected recorded type for '+name+': '+kind)
    messages = {name: get_message(kind) for name, kind in actual.items() if name in TOPICS}
    def records():
        while reader.has_next():
            name, data, ns = reader.read_next()
            if name in messages:
                yield name, ns, deserialize_message(data, messages[name]).data
    summary, rows = summarize_records(records())
    summary['analyzed_at'] = datetime.now().astimezone().isoformat()
    (folder/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    with (folder/'force_pose.csv').open('w',encoding='utf-8-sig',newline='') as f:
        if rows:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    manifest=[]
    for p in sorted(folder.rglob('*')):
        if p.is_file() and p.name!='manifest.json':
            h=hashlib.sha256()
            with p.open('rb') as handle:
                for block in iter(lambda:handle.read(1048576),b''):h.update(block)
            manifest.append(dict(path=str(p.relative_to(folder)),bytes=p.stat().st_size,sha256=h.hexdigest()))
    (folder/'manifest.json').write_text(json.dumps(dict(files=manifest),indent=2)+'\n')
    return summary
