"""Read-only robot-PC evidence capture. No ROS commands, signals or file writes."""
import datetime, hashlib, json, os
from pathlib import Path
import subprocess

base = Path('/home/nrs_forcecon/dev_ws/src/y2_ur10skku_control')
def file_info(path, with_text=True):
    p = Path(path)
    item = dict(path=str(p), exists=p.is_file())
    if not p.is_file():
        return item
    data=p.read_bytes()
    item.update(resolved_path=str(p.resolve()), sha256=hashlib.sha256(data).hexdigest(),
                bytes=len(data), mtime_utc=datetime.datetime.fromtimestamp(p.stat().st_mtime, datetime.timezone.utc).isoformat())
    if with_text:
        item['text']=data.decode('utf-8', errors='replace')
    return item

relative = [
 'Y2RobMotion/config/setup_parameters.yaml', 'Y2FT_AQ/config/spindle_gravity.yaml',
 'Y2RobMotion/src/runtime_config.cpp', 'Y2RobMotion/src/force_control.cpp',
 'Y2RobMotion/src/robot_motion.cpp', 'Y2RobMotion/include/Y2RobMotion/robot_motion.hpp',
 'Y2RobMotion/src/singleArm_motion.cpp', 'Y2RobMotion/src/robot_command.cpp',
 'Y2RobMotion/src/singleArm_cmd.cpp',
 'Y2FT_AQ/src/FTGetMain.cpp', 'Y2FT_AQ/src/FT_EtherGet.cpp',
 'Y2FT_AQ/include/Y2FT_AQ/FT_EtherGet.hpp', 'Y2FT_AQ/include/Y2FT_AQ/FT_Acquisition.hpp']
files = [file_info(base/name) for name in relative]
for name in ('Y2RobMotion/config/setup_parameters.yaml','Y2FT_AQ/config/spindle_gravity.yaml'):
    package, tail=name.split('/',1)
    files.append(file_info(Path('/home/nrs_forcecon/dev_ws/install')/package/'share'/package/tail))
processes=[]
btime=int(next(x.split()[1] for x in Path('/proc/stat').read_text().splitlines() if x.startswith('btime ')))
for proc in Path('/proc').iterdir():
    if not proc.name.isdigit(): continue
    try:
        exe=Path(os.readlink(proc/'exe'))
        if exe.name not in ('FTGetMain','singleArm_motion','singleArm_cmd','ur_ros2_control_node'):
            continue
        argv=(proc/'cmdline').read_bytes().decode(errors='replace').strip('\0').split('\0')
        env=(proc/'environ').read_bytes().decode(errors='replace').split('\0')
        env=dict(s.split('=',1) for s in env if '=' in s and s.split('=',1)[0] in ('ROS_DOMAIN_ID','ROS_LOCALHOST_ONLY','RMW_IMPLEMENTATION','ROS_NAMESPACE'))
        stat=(proc/'stat').read_text().rsplit(')',1)[1].split()
        started=btime+int(stat[19])/os.sysconf('SC_CLK_TCK')
        item=dict(pid=int(proc.name),executable=str(exe),argv=argv,ros_environment=env,
                  started_at_utc=datetime.datetime.fromtimestamp(started,datetime.timezone.utc).isoformat(),
                  running_image=file_info(proc/'exe',False))
        item['parameter_files']=[file_info(argv[i+1]) for i,a in enumerate(argv[:-1]) if a=='--params-file']
        item['logs']=[]
        for p in sorted(Path('/home/nrs_forcecon/.ros/log').glob('*'+proc.name+'*.log'))[:4]:
            with p.open(errors='replace') as f: head=f.read(48000)
            lines=[s for s in head.splitlines() if any(k in s.lower() for k in ('config','yaml','gravity','initializ','safety','ether','force control','force_control','control coordinate'))]
            item['logs'].append(dict(path=str(p),startup_matching_lines=lines[:40],scan='first 48000 characters'))
        processes.append(item)
    except (OSError,ValueError,IndexError) as exc:
        processes.append(dict(pid=proc.name,error=str(exc)))
status=subprocess.run(['git','-C',str(base),'status','--short'],text=True,capture_output=True)
clock=subprocess.run(['timedatectl','show','-p','NTPSynchronized','-p','Timezone'],text=True,capture_output=True)
print(json.dumps(dict(captured_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
 host=os.uname().nodename,source='live robot PC filesystem and /proc',read_only=True,
 hardware_commands_sent=False,files=files,processes=processes,
 git_status=status.stdout,clock=dict(returncode=clock.returncode,stdout=clock.stdout,stderr=clock.stderr))))

