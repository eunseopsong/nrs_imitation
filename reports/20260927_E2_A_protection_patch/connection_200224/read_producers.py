"""Read-only producer graph and loaded binaries; no control services or publishers."""
import datetime,hashlib,json,os,time
from pathlib import Path
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile,ReliabilityPolicy
from std_msgs.msg import String
base=Path('/home/nrs_forcecon/dev_ws/src/y2_ur10skku_control')
names=['Y2FT_AQ/include/Y2FT_AQ/FT_Acquisition.hpp','Y2FT_AQ/include/Y2FT_AQ/FT_EtherGet.hpp','Y2FT_AQ/src/FT_EtherGet.cpp','Y2FT_AQ/src/FTGetMain.cpp','Y2RobMotion/include/Y2RobMotion/robot_motion.hpp','Y2RobMotion/src/robot_motion.cpp','Y2RobMotion/config/setup_parameters.yaml','Y2FT_AQ/config/spindle_gravity.yaml']
out=dict(captured_at=datetime.datetime.now().astimezone().isoformat(),host=os.uname().nodename,read_only=True,files={},processes=[],topics={})
for name in names:
 p=base/name
 out['files'][name]=dict(path=str(p),exists=p.is_file())
 if p.is_file():
  b=p.read_bytes();out['files'][name].update(sha256=hashlib.sha256(b).hexdigest(),text=b.decode(),bytes=len(b))
for p in Path('/proc').iterdir():
 if not p.name.isdigit():continue
 try:
  exe=(p/'exe').resolve()
  if exe.name not in ('FTGetMain','singleArm_motion','singleArm_cmd'):continue
  out['processes'].append(dict(pid=int(p.name),exe=str(exe),sha256=hashlib.sha256((p/'exe').read_bytes()).hexdigest(),argv=(p/'cmdline').read_bytes().decode().strip('\0').split('\0')))
 except (OSError,ValueError):pass
rclpy.init(args=['--ros-args','--disable-rosout-logs','--disable-external-lib-logs'])
node=Node('e2_producer_read_only',enable_rosout=False,start_parameter_services=False)
try:
 end=time.monotonic()+3
 while time.monotonic()<end:rclpy.spin_once(node,timeout_sec=.05)
 out['domain_id']=os.getenv('ROS_DOMAIN_ID')
 for topic in ('/ur10skku/ft_acquisition','/ur10skku/currentF_provenance'):
  pubs=node.get_publishers_info_by_topic(topic)
  row=dict(publishers=[dict(node=p.node_name,namespace=p.node_namespace,type=p.topic_type) for p in pubs],count=0,last=None)
  out['topics'][topic]=row
  def cb(msg,row=row):
   row['count']+=1
   try:row['last']=json.loads(msg.data)
   except ValueError:row['last']={'invalid_json':True}
  node.create_subscription(String,topic,cb,QoSProfile(depth=1,reliability=ReliabilityPolicy.RELIABLE))
 end=time.monotonic()+3
 while time.monotonic()<end:rclpy.spin_once(node,timeout_sec=.05)
 out['e2_services']={n:t for n,t in node.get_service_names_and_types() if n.startswith('/e2_executor/')}
finally:
 node.destroy_node();rclpy.shutdown()
print(json.dumps(out))

