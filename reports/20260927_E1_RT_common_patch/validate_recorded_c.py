"""Replay five archived C inputs through archived/current engines; no ROS I/O."""
import csv
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path('/home/eunseop/nrs_imitation')
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT/'behavior_ws/src/nrs_imitation'))
from nrs_imitation import e2_timed_execution as current

KEYS = ['x','y','z','rx','ry','rz','fx','fy','fz']


def rows(p):
    with p.open() as f:
        return list(csv.DictReader(f))


def main():
    reports = []
    for folder in sorted((ROOT/'results/20260926/E1/C').iterdir()):
        if not (folder/'executor/metadata.json').exists():
            continue
        spec = importlib.util.spec_from_file_location('archived_c_engine',
            folder/'executor/artifacts/1_e2_timed_execution.py')
        archived = importlib.util.module_from_spec(spec);sys.modules[spec.name]=archived
        spec.loader.exec_module(archived)
        cfg = json.loads((folder/'launch_context/config.json').read_text())
        events = [json.loads(s) for s in (folder/'executor/events.jsonl').read_text().splitlines()]
        start = next(e['details']['monotonic_start'] for e in events if e['event']=='execution_start')
        receives = [e for e in events if e['event']=='plan_received']
        poses = rows(folder/'executor/tcp_pose.csv')
        pose_times = np.array([int(r['receipt_monotonic_ns'])/1e9 for r in poses])
        pose_values = np.array([[float(r[k]) for k in KEYS[:6]] for r in poses])
        initial = pose_values[max(0, np.searchsorted(pose_times,start,side='right')-1)]
        provider = rows(folder/'provider/commands.csv')
        plans = {}
        for e in receives:
            d = e['details'];pid=d['plan_id']
            selected = sorted((r for r in provider if r['command_stage']=='postprocessed' and int(r['plan_id'])==pid),key=lambda r:int(r['action_index']))
            assert len(selected)==128
            plans[pid]=dict(plan_id=pid,generated_at=d['source_plan_time'],time=np.arange(128)/30.,
                action=np.array([[float(r[k]) for k in KEYS] for r in selected]),phase=np.full(128,'unknown'),final=False)
        modules=[archived,current];engines=[]
        for module in modules:
            engine=module.TimedExecution(module.executor_settings(cfg,'il'))
            engine.accept(module.TimedPlan(**plans[1]),receives[0]['receipt_monotonic_ns']/1e9)
            engine.start(start,initial);engines.append(engine)
        index=1;ticks=0
        for r in rows(folder/'executor/commands.csv'):
            if r['command_stage']!='time_sampled':continue
            d=json.loads(r['details']);pid=int(r['plan_id'])
            now=(start if pid==1 else plans[pid]['generated_at'])+d['elapsed_s']
            while index<len(receives) and receives[index]['receipt_monotonic_ns']/1e9<=now:
                e=receives[index];at=e['receipt_monotonic_ns']/1e9
                for module,engine in zip(modules,engines):
                    engine.accept(module.TimedPlan(**plans[e['details']['plan_id']]),at)
                index+=1
            pose=pose_values[max(0,np.searchsorted(pose_times,now,side='right')-1)]
            force=[0.,0.,4. if d['contact'] else .2]
            results=[engine.tick(now,pose,force,0.,0.) for engine in engines]
            for stage in ['requested','conditioned','gated','sent']:
                assert results[0][stage].tobytes()==results[1][stage].tobytes(),(folder.name,ticks,stage)
            assert results[0]['contact']==results[1]['contact']
            ticks+=1
        reports.append(dict(run=folder.name,ticks=ticks,all_9d_stages_bitwise_equal=True))
        print(folder.name,ticks,'identical',flush=True)
    assert len(reports)==5
    out=dict(status='pass',runs=reports,total_ticks=sum(r['ticks'] for r in reports),
        comparison='Archived C engine vs current code, same archived inputs and recorded contact decisions.',
        hardware_executed=False,physical_response_inferred=False)
    (OUT/'recorded_c_validation.json').write_text(json.dumps(out,indent=2)+'\n')


if __name__=='__main__':main()
