#!/usr/bin/env python3
"""Record passing offline checks, snapshot diff, and verify E1 stayed immutable."""
from pathlib import Path
from datetime import datetime
import difflib
import json
import sys
import xml.etree.ElementTree as ET

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'behavior_ws/src/nrs_imitation'))
from nrs_imitation.e2_ablation import EXPERIMENT,digest,object_hash,execution_contract,readiness,cohort_id


def main():
    cfg=json.loads((EXPERIMENT/'config.json').read_text())
    xml=ET.parse(EXPERIMENT/'checks/unit_tests.xml').getroot()
    suites=list(xml.iter('testsuite'))
    assert suites and all(int(s.get('errors','0'))==int(s.get('failures','0'))==0 for s in suites)
    tests=sum(int(s.get('tests','0')) for s in suites)
    assert tests>=145
    archive=ROOT/'results/20260926/E1';manifest=json.loads((archive/'manifest.json').read_text())
    for entry in manifest['files']:assert digest(archive/entry['path'])==entry['sha256'],entry['path']
    report_sha=digest(ROOT/'results/20260926/E1_final_report_20260926.pdf')
    assert report_sha=='bbc4332071de0846dd65934f12355b42a4cabd68ec2afa2d2a84f009f160951e'
    # Preparation hash covers current source/settings without approving any
    # unreviewed protocol, calibration, F0 or physical execution.
    if not cfg['protocol'].get('reviewed'):
        cfg['execution_contract_hash']=object_hash(execution_contract(cfg))
        cfg['cohort_id']=cohort_id(cfg)
        (EXPERIMENT/'config.json').write_text(json.dumps(cfg,ensure_ascii=False,indent=2)+'\n')
    # A later check never approves changed settings/code for a reviewed protocol.
    # Explicit protocol review/seal is required to refresh that contract.
    status=readiness(cfg)
    status['checked_at']=datetime.now().astimezone().isoformat()
    status['execution_contract_hash']=cfg['execution_contract_hash'];status['cohort_id']=cfg['cohort_id']
    status['tests_passed']=tests;status['E1_archive_hashes_unchanged']=True
    status['E1_manifest_sha256']=digest(archive/'manifest.json')
    status['E1_report_sha256']=report_sha
    status['training_smoke_report']='checks/training_smoke.json'
    status['code_preparation_complete']=True;status['hardware_executed']=False
    (EXPERIMENT/'status.json').write_text(json.dumps(status,ensure_ascii=False,indent=2)+'\n')
    diff=[];paths=[]
    before=EXPERIMENT/'before'
    for p in sorted(before.rglob('*')):
        if p.is_file() and '__pycache__' not in p.parts:
            relative=p.relative_to(before);current=ROOT/relative
            if current.is_file():
                lines=list(difflib.unified_diff(p.read_text().splitlines(True),current.read_text().splitlines(True),
                                               fromfile='a/'+str(relative),tofile='b/'+str(relative)))
                if lines:diff.extend(lines);paths.append(str(relative))
    new_files=[
        'behavior_ws/src/nrs_imitation/nrs_imitation/e2_ablation.py',
        'behavior_ws/src/nrs_imitation/nrs_imitation/e2_run_context.py',
        'behavior_ws/src/nrs_imitation/nrs_imitation/e2_force_analysis.py',
        'behavior_ws/src/nrs_imitation/launch/e2_abc.launch.py',
        'behavior_ws/src/nrs_imitation/test/test_e2_ablation.py',
        'behavior_ws/src/nrs_imitation/test/test_e2_ablation_ros_adapters.py',
        'behavior_ws/src/nrs_imitation/test/test_e2_direct_launch.py',
        'scripts/e2_ablation.py','scripts/flow/e2_ablation_training.py','scripts/e2_finalize_preparation.py']
    new_files += [str(p.relative_to(ROOT)) for p in sorted(EXPERIMENT.iterdir())
                  if p.suffix in ('.md','.sh','.csv') or p.name in ('config.json','training_configs.json',
                      'calibration.template.json','annotations.template.json','phase_annotations.template.json',
                      'stylus_metrology.template.csv')]
    for name in new_files:
        diff.extend(difflib.unified_diff([], (ROOT/name).read_text().splitlines(True),fromfile='/dev/null',tofile='b/'+name))
        paths.append(name)
    (EXPERIMENT/'changes.patch').write_text(''.join(diff))
    (EXPERIMENT/'changed_files.json').write_text(json.dumps({p:digest(ROOT/p) for p in paths},indent=2)+'\n')
    print(json.dumps(status,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
