#!/usr/bin/env python3
"""Select an explicit R force baseline from already recorded E1 commands; no ROS I/O.

Use the final selected C run. Reconstruct acknowledged SET_FORCE holds and take
an elapsed-time weighted median in the longest uninterrupted nonzero gate-ON
interval. Gate state is a software proxy, not a machining/contact ground truth.
B is reported as a sensitivity comparison and is not pooled into C's value.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def weighted_quantile(values, weights, q):
    if not 0 <= q <= 1 or len(values) != len(weights) or not values:
        raise ValueError('Invalid quantile input')
    pairs=sorted((float(v),float(w)) for v,w in zip(values,weights))
    if any(not math.isfinite(v) or not math.isfinite(w) or w<=0 for v,w in pairs):
        raise ValueError('Finite values and positive durations required')
    target=q*sum(w for _,w in pairs);cumulative=0.
    for value,weight in pairs:
        cumulative+=weight
        if cumulative>=target:return value
    return pairs[-1][0]


def reconstruct(rows, events):
    """Retain accepted targets and their hold durations, including zero-gate gaps."""
    acknowledgements={e['details']['command_id']:e['details'].get('success')
        for e in events if e['event']=='service_response'}
    end=min(e['receipt_monotonic_ns'] for e in events if e['event'] in ('run_interrupted','run_end'))
    sent=[r for r in rows if r['command_stage']=='node_sent' and r['command_mode']=='PTP9D_STREAM_SET_FORCE']
    if not sent:raise ValueError('No real SET_FORCE sends')
    sent.sort(key=lambda r:int(r['receipt_monotonic_ns']))
    origin=min(int(r['receipt_monotonic_ns']) for r in rows)
    intervals=[];runs=[];current=[]
    for i,r in enumerate(sent):
        command_id=int(r['command_id'])
        if acknowledgements.get(command_id) is not True:
            raise ValueError('Missing/failed force service response; target hold cannot be assumed')
        t=int(r['receipt_monotonic_ns'])
        until=int(sent[i+1]['receipt_monotonic_ns']) if i+1<len(sent) else end
        if until<=t:raise ValueError('Invalid force command time ordering')
        fz=float(r['fz']);details=json.loads(r['details'])
        if r['validity']!='valid' or not math.isfinite(fz):raise ValueError('Invalid force command')
        gate=details['contact_gate']
        if not isinstance(gate,bool):raise ValueError('Explicit gate state required')
        row=dict(command_id=command_id,start_s=(t-origin)/1e9,end_s=(until-origin)/1e9,
            duration_s=(until-t)/1e9,target_fz_N=fz,gate_on=gate,selected_interval=False)
        intervals.append(row)
        if gate and fz!=0.:current.append(row)
        elif current:runs.append(current);current=[]
    if current:runs.append(current)
    if not runs:raise ValueError('No acknowledged nonzero gate-ON interval')
    selected=max(runs,key=lambda rr:sum(r['duration_s'] for r in rr))
    for r in selected:r['selected_interval']=True
    values=[r['target_fz_N'] for r in selected];weights=[r['duration_s'] for r in selected]
    median=weighted_quantile(values,weights,.5)
    return dict(interval_start_s=selected[0]['start_s'],interval_end_s=selected[-1]['end_s'],
        interval_duration_s=sum(weights),command_count=len(selected),
        time_weighted_median_N=median,time_weighted_q25_N=weighted_quantile(values,weights,.25),
        time_weighted_q75_N=weighted_quantile(values,weights,.75),
        time_weighted_mean_N=sum(v*w for v,w in zip(values,weights))/sum(weights),
        min_target_N=min(values),max_target_N=max(values),
        positive_duration_s=sum(w for v,w in zip(values,weights) if v>0),
        negative_duration_s=sum(w for v,w in zip(values,weights) if v<0)),intervals


def analyze(archive, output):
    manifest=archive/'manifest.json';chosen=json.loads(manifest.read_text())['selected']
    report=dict(source_manifest=str(manifest.resolve()),source_manifest_sha256=sha256(manifest),
        primary_method='C',selection='longest continuous acknowledged nonzero contact-gate-ON interval; time-weighted empirical median',
        force_semantics='signed SET_FORCE target, N, controller-config-dependent frame; NOT measured normal force or controller-applied force',
        exclusions='zero target/gate-OFF periods; APPEND unused zero force columns; measured force spikes never enter selection',
        note='Existing gate is only a proxy; no operator processing_start/end labels exist in E1. No E2 outcome tuning.',runs={})
    all_intervals=[]
    for method,folder in [('B','B_force_obs_OFF'),('C','C_force_obs_ON')]:
        log=archive/folder/chosen[method]/'logs'
        with (log/'commands.csv').open() as f:rows=list(csv.DictReader(f))
        events=[json.loads(line) for line in (log/'events.jsonl').read_text().splitlines()]
        summary=json.loads((log/'summary.json').read_text())
        if summary.get('write_error_count') or not summary.get('drained') or any(v['dropped'] for v in summary.get('counts',{}).values()):
            raise ValueError('Source logger loss/error; force reconstruction is incomplete')
        result,intervals=reconstruct(rows,events)
        result.update(run_id=chosen[method],log_dir=str(log.resolve()),
            commands_sha256=sha256(log/'commands.csv'),events_sha256=sha256(log/'events.jsonl'))
        report['runs'][method]=result
        all_intervals.extend(dict(method=method,**r) for r in intervals)
    reference=float(round(report['runs']['C']['time_weighted_median_N']))
    if reference==0:raise ValueError('Rounded reference became zero; manual review required')
    report['force_reference_N']=reference
    report['rounding']='nearest 1 N, sign preserved'
    report['status']='E1-derived initial fixed-force baseline; not separately tuned/validated on R hardware'
    output.mkdir(parents=True,exist_ok=True)
    (output/'force_reference_from_e1.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    with (output/'force_reference_from_e1_intervals.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(all_intervals[0]));writer.writeheader();writer.writerows(all_intervals)
    return report


def main():
    root=Path(__file__).resolve().parents[1]
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive',type=Path,default=root/'results/20260920')
    parser.add_argument('--output',type=Path,default=root/'experiments/e2_rule_replay_20260920')
    args=parser.parse_args();report=analyze(args.archive,args.output)
    print(json.dumps(dict(force_reference_N=report['force_reference_N'],runs=report['runs']),ensure_ascii=False,indent=2))


if __name__=='__main__':main()
