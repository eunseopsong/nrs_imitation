import datetime,json,time
import rclpy
from rclpy.node import Node
from rosidl_runtime_py.utilities import get_message,get_service
from rosidl_runtime_py.convert import message_to_ordereddict
from rcl_interfaces.srv import GetParameters,ListParameters
rclpy.init(args=['--ros-args','--disable-rosout-logs','--disable-external-lib-logs'])
n=Node('e2_readonly_protection_status',enable_rosout=False,start_parameter_services=False)
out=dict(captured_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),read_only=True,
 command_publishers_created=0,mutating_service_calls=0,topics={},queries={})
subscriptions=[]
try:
 end=time.monotonic()+2.
 while time.monotonic()<end:rclpy.spin_once(n,timeout_sec=.05)
 graph=dict(n.get_topic_names_and_types());services=dict(n.get_service_names_and_types())
 for name in ('/io_and_status_controller/safety_mode','/io_and_status_controller/robot_mode',
              '/io_and_status_controller/robot_program_running','/dynamic_joint_states'):
  pubs=n.get_publishers_info_by_topic(name);entry=dict(publishers=len(pubs),messages_received=0,last=None);out['topics'][name]=entry
  if not pubs:continue
  cls=get_message(graph[name][0])
  entry['constants']={k:getattr(cls,k) for k in dir(cls) if k.isupper() and isinstance(getattr(cls,k),(str,int,float,bool))}
  def cb(msg,e=entry,topic=name):
   e['messages_received']+=1
   if topic=='/dynamic_joint_states':
    data={}
    for joint,values in zip(msg.joint_names,msg.interface_values):
     for interface,value in zip(values.interface_names,values.values):
      key=joint+'/'+interface
      if any(k in key for k in ('safety','robot_mode','program_running','robot_status','get_robot_software_version')):
       data[key]=value
    e['last']=dict(header=message_to_ordereddict(msg.header),values=data)
   else:e['last']=message_to_ordereddict(msg)
  subscriptions.append(n.create_subscription(cls,name,cb,pubs[0].qos_profile))
 end=time.monotonic()+3.
 while time.monotonic()<end:rclpy.spin_once(n,timeout_sec=.03)
 def call(cls,name,msg):
  cli=n.create_client(cls,name)
  try:
   if not cli.wait_for_service(timeout_sec=1.):return dict(available=False)
   f=cli.call_async(msg);rclpy.spin_until_future_complete(n,f,timeout_sec=2.)
   return dict(available=True,response=message_to_ordereddict(f.result())) if f.done() else dict(available=True,error='timeout')
  finally:n.destroy_client(cli)
 for name in ('/dashboard_client/get_safety_mode','/dashboard_client/get_robot_mode','/dashboard_client/get_program_state'):
  if name not in services:out['queries'][name]=dict(available=False);continue
  cls=get_service(services[name][0])
  out['queries'][name]=call(cls,name,cls.Request())
 listed=call(ListParameters,'/controller_stopper/list_parameters',ListParameters.Request(prefixes=[],depth=0))
 names=[x for x in listed.get('response',{}).get('result',{}).get('names',[]) if not x.startswith('qos_')]
 out['controller_stopper']=dict(list_result=listed,names=names)
 if names:out['controller_stopper']['values']=call(GetParameters,'/controller_stopper/get_parameters',GetParameters.Request(names=names))
 out['ended_at_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
finally:
 n.destroy_node();rclpy.shutdown()
print(json.dumps(out))

