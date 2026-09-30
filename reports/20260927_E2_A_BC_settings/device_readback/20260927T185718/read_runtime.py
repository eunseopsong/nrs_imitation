"""Read-only ROS graph, feedback subscriptions and parameter/list queries."""
import datetime,json,os,time
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile,ReliabilityPolicy,DurabilityPolicy
from rosidl_runtime_py.utilities import get_message
from rosidl_runtime_py.convert import message_to_ordereddict
from rcl_interfaces.srv import ListParameters,GetParameters
from controller_manager_msgs.srv import ListControllers

rclpy.init(args=['--ros-args','--disable-rosout-logs','--disable-external-lib-logs'])
node=Node('e2_readonly_device_evidence',enable_rosout=False,start_parameter_services=False)
out=dict(captured_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
 source='Live ROS domain of the running robot drivers',read_only=True,
 ROS_DOMAIN_ID=os.getenv('ROS_DOMAIN_ID'),RMW_IMPLEMENTATION=os.getenv('RMW_IMPLEMENTATION'),
 command_publishers_created=0,mutating_service_calls=0)
subscriptions=[]
try:
    end=time.monotonic()+3.
    while time.monotonic()<end:rclpy.spin_once(node,timeout_sec=.05)
    graph=dict(node.get_topic_names_and_types())
    out['nodes']=[dict(name=n,namespace=s) for n,s in node.get_node_names_and_namespaces()]
    expected=['/ur10skku/currentF','/ur10skku/currentP','/ur10skku/ctlMode',
      '/ur10skku/ft_acquisition','/ur10skku/currentF_provenance',
      '/io_and_status_controller/safety_mode','/io_and_status_controller/robot_mode',
      '/io_and_status_controller/robot_program_running',
      '/speed_scaling_state_broadcaster/speed_scaling',
      '/force_torque_sensor_broadcaster/wrench']
    expected+=sorted(t for t in graph if 'ftdata' in t.lower())
    out['topics']={}
    for topic in dict.fromkeys(expected):
        pubs=node.get_publishers_info_by_topic(topic)
        entry=dict(types=graph.get(topic,[]),publisher_count=len(pubs),
          publishers=[dict(node_name=p.node_name,node_namespace=p.node_namespace,
            type=p.topic_type,reliability=str(p.qos_profile.reliability),
            durability=str(p.qos_profile.durability)) for p in pubs],
          messages_received=0,first=None,last=None,max_receipt_gap_s=None)
        out['topics'][topic]=entry
        if not pubs or not graph.get(topic):continue
        try:
            cls=get_message(graph[topic][0])
            constants={k:getattr(cls,k) for k in dir(cls) if k.isupper() and isinstance(getattr(cls,k),(str,int,float,bool))}
            if constants:entry['message_constants']=constants
            qos=QoSProfile(depth=10,reliability=ReliabilityPolicy.BEST_EFFORT)
            qos.durability=pubs[0].qos_profile.durability
            def receive(msg, e=entry):
                now=time.monotonic()
                data=message_to_ordereddict(msg)
                if e['messages_received']==0:
                    e['first']=data;e['first_receipt_monotonic_s']=now
                else:
                    gap=now-e['last_receipt_monotonic_s']
                    e['max_receipt_gap_s']=max(gap,e['max_receipt_gap_s'] or 0.)
                e['last']=data;e['last_receipt_monotonic_s']=now;e['messages_received']+=1
            subscriptions.append(node.create_subscription(cls,topic,receive,qos))
        except Exception as exc:entry['observation_error']=str(exc)
    end=time.monotonic()+5.
    while time.monotonic()<end:rclpy.spin_once(node,timeout_sec=.02)
    out['services']={name:types for name,types in node.get_service_names_and_types()
                     if any(s in name.lower() for s in ('safety','stop','controller','force','ftget','singlearm'))}
    def request(cls,name,msg,timeout=1.5):
        client=node.create_client(cls,name)
        try:
            if not client.wait_for_service(timeout_sec=timeout):return dict(available=False)
            future=client.call_async(msg);rclpy.spin_until_future_complete(node,future,timeout_sec=timeout)
            if not future.done():return dict(available=True,error='read-only request timed out')
            return dict(available=True,response=message_to_ordereddict(future.result()))
        finally:node.destroy_client(client)
    out['controllers']=request(ListControllers,'/controller_manager/list_controllers',ListControllers.Request())
    actual_nodes={s.rstrip('/')+'/'+n for n,s in node.get_node_names_and_namespaces()}
    out['parameters']={}
    for target in ('/SingleArm','/singleArm_cmd','/FTGetMain','/controller_manager','/ur_hardware_interface'):
        if target not in actual_nodes:continue
        names_result=request(ListParameters,target+'/list_parameters',ListParameters.Request(prefixes=[],depth=0))
        entry=dict(list_result=names_result)
        names=names_result.get('response',{}).get('result',{}).get('names',[])
        names=[n for n in names if n!='robot_description' and any(k in n.lower() for k in
          ('setup','gravity','safety','force','wrench','sensor','timeout','watch','workspace','frame','period','kinemat','joint','speed','update_rate'))][:120]
        if names:
            entry['requested_names']=names
            entry['values']=request(GetParameters,target+'/get_parameters',GetParameters.Request(names=names))
        out['parameters'][target]=entry
    out['ended_at_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
finally:
    node.destroy_node();rclpy.shutdown()
print(json.dumps(out))

