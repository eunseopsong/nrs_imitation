"""Compile the production serializers against fake I/O; never open ROS or sockets.

The C++ blocks below are read verbatim from the staged producers, not rewritten
Python payload builders. Limits/approvals come only from temporary mock fixtures.
"""
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace as NS
from types import MethodType

import numpy as np
import pytest

from nrs_imitation.e2_ablation import ROOT
from nrs_imitation.e2_executor_node import E2Executor
from nrs_imitation.e2_protection import ProtectionMonitor, ProtectionFault
from test_e2_A_pilot import approved_mock
from test_e2_protection import record, settings, protected_bridge, source_callbacks

DRIVER = ROOT/'experiments/e2_A_protection_20260927/driver_src'


@pytest.fixture(scope='module')
def producer_payloads(tmp_path_factory):
    ft = (DRIVER/'Y2FT_AQ/src/FTGetMain.cpp').read_text()
    robot = (DRIVER/'Y2RobMotion/src/robot_motion.cpp').read_text()
    raw_block = ft.split('    // Separate provenance channel:', 1)[1].split('    #endif', 1)[0]
    raw_block = raw_block[raw_block.index('\n'):]
    stamp_block = ft.split('    ft1_msg.header.stamp = rclcpp::Time(acquisition_ros_ns_);', 1)[1].split('    #else', 1)[0]
    stamp_block = 'ft1_msg.header.stamp = rclcpp::Time(acquisition_ros_ns_);' + stamp_block
    callback = robot.split('void robot_motion::ftsensorCB(', 1)[1].split('/* joint state mapping function */', 1)[0]
    callback = callback.split('{', 1)[1].rsplit('}', 1)[0]
    base_block = robot.split('    bool valid_wrench = true;', 1)[1].split('    currentMDK_pub->publish', 1)[0]
    base_block = 'bool valid_wrench = true;' + base_block
    source = r'''
#include "Y2FT_AQ/FT_Acquisition.hpp"
#include <cassert>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>
namespace rclcpp {
struct Time { std::int64_t ns; explicit Time(std::int64_t n):ns(n){}
    std::int64_t nanoseconds() const {return ns;} };
}
namespace std_msgs { namespace msg {struct String {std::string data;};}}
namespace geometry_msgs { namespace msg {
struct WrenchStamped {
    using SharedPtr=std::shared_ptr<WrenchStamped>;
    struct Stamp {std::int32_t sec=0; std::uint32_t nanosec=0;
        Stamp& operator=(rclcpp::Time t){sec=t.ns/1000000000LL;nanosec=t.ns%1000000000LL;return *this;}};
    struct Header {Stamp stamp;std::string frame_id;} header;
    struct XYZ {double x=0,y=0,z=0;};
    struct Wrench {XYZ force,torque;} wrench;
};}}
struct Publisher {std::vector<std::string> data;
    void publish(const std_msgs::msg::String& m){data.push_back(m.data);}};
struct Reader {FTAcquisition sample;bool ready=true;
    const FTAcquisition& acquisition() const {return sample;}
    bool initialized() const {return ready;}};
struct FTMock {
    Reader FT1SensorGet;
    std::shared_ptr<Publisher> acquisition_pub_=std::make_shared<Publisher>();
    std::uint64_t acquisition_published_sequence_=0;
    std::int64_t acquisition_ros_ns_=0,now_ns=10000000000LL;
    const std::string acquisition_boot_id_="MOCK_PROCESS_BOOT";
    geometry_msgs::msg::WrenchStamped ft1_msg;
    rclcpp::Time now() const {return rclcpp::Time(now_ns);}
    void publish(){RAW_BLOCK
        STAMP_BLOCK
    }
};
struct RobotMock {
    std::int64_t ft1_source_ros_ns_=0;bool ft1_source_contract_=false;
    std::vector<double> ft1data=std::vector<double>(6,0.);
    std::shared_ptr<Publisher> currentF_provenance_pub_=std::make_shared<Publisher>();
    void receive(const geometry_msgs::msg::WrenchStamped::SharedPtr msg){CALLBACK}
    void publish(){BASE_BLOCK}
};
void encode(char* p,float v){std::uint32_t bits;std::memcpy(&bits,&v,4);
    for(int i=0;i<4;++i)p[i]=static_cast<char>(bits>>(24-8*i));}
int main(){
    FTMock f;RobotMock r;char bytes[48]{};
    const float values[]={1.25f,-2.5f,7.f,.125f,-.25f,.5f};
    for(int i=0;i<6;++i)encode(bytes+4*i,values[i]);
    f.publish();assert(f.acquisition_pub_->data.empty());
    f.FT1SensorGet.sample.received(bytes,48,FTAcquisition::Clock::now());
    f.publish();std::cout<<f.acquisition_pub_->data.back()<<'\n';
    auto msg=std::make_shared<geometry_msgs::msg::WrenchStamped>(f.ft1_msg);
    msg->wrench.force.z=6.;msg->wrench.torque.x=.125;
    r.receive(msg);r.publish();std::cout<<r.currentF_provenance_pub_->data.back()<<'\n';
    f.now_ns+=80000000;f.publish();assert(f.acquisition_pub_->data.size()==1);
    r.publish();std::cout<<r.currentF_provenance_pub_->data.back()<<'\n';
    f.now_ns+=1000000;
    f.FT1SensorGet.sample.received(bytes,48,FTAcquisition::Clock::now());
    f.publish();std::cout<<f.acquisition_pub_->data.back()<<'\n';
    f.FT1SensorGet.sample.received(bytes,23,FTAcquisition::Clock::now());
    f.publish();std::cout<<f.acquisition_pub_->data.back()<<'\n';
    f.FT1SensorGet.sample.received(bytes,48,FTAcquisition::Clock::now());
    f.publish();std::cout<<f.acquisition_pub_->data.back()<<'\n';
}
'''
    for key, block in [('RAW_BLOCK',raw_block),('STAMP_BLOCK',stamp_block),
                       ('CALLBACK',callback),('BASE_BLOCK',base_block)]:
        source = source.replace(key,block)
    folder=tmp_path_factory.mktemp('actual_producer_cpp')
    cpp=folder/'producer.cpp';cpp.write_text(source)
    binary=folder/'producer'
    subprocess.run(['g++','-std=c++17','-O0','-Wall','-Wextra','-Werror',
                    '-I'+str(DRIVER/'Y2FT_AQ/include'),str(cpp),'-o',str(binary)],check=True,capture_output=True)
    output=subprocess.run([str(binary)],check=True,capture_output=True,text=True).stdout
    return [json.loads(line) for line in output.splitlines()]


def test_actual_udp_decode_serializers_and_consumer(producer_payloads):
    raw,base,cached,next_raw,invalid,recovered=producer_payloads
    assert raw['sensor_wrench']==[1.25,-2.5,7.,.125,-.25,.5]
    assert raw['wrench_frame']=='sensor' and base['wrench_frame']=='robot_base'
    assert raw['source_ros_ns']==base['source_ros_ns']==cached['source_ros_ns']
    assert base==cached and raw['boot_id']=='MOCK_PROCESS_BOOT'
    assert next_raw['sequence']==raw['sequence']+1
    assert next_raw['source_ros_ns']>raw['source_ros_ns']
    assert invalid['valid'] is False and recovered['valid'] is False
    assert base['base_wrench']==[0.,0.,6.,.125,0.,0.]
    g=ProtectionMonitor(record(),settings())
    now_ns=raw['source_ros_ns']+1_000_000
    g.ingest_raw(raw,10.,now_ns);g.ingest_base(base,10.,now_ns)
    assert g.check(10.,now_ns,np.zeros(6),0.,0.,0.)['raw_sequence']==raw['sequence']
    g.ingest_raw(raw,10.08,now_ns+80_000_000)
    g.ingest_base(cached,10.08,now_ns+80_000_000)
    assert g.raw_at==g.base_at==10.
    with pytest.raises(ProtectionFault,match='stopped advancing'):
        g.check(10.11,now_ns+110_000_000,np.zeros(6),0.,0.,0.)


@pytest.mark.parametrize('raw_side',[True,False])
@pytest.mark.parametrize('key,bad',[('wrench_frame','tcp'),('force_unit','kN'),
    ('torque_unit','Nmm'),('timestamp_origin','current_publish_time')])
def test_actual_producer_metadata_cannot_be_misinterpreted(producer_payloads,raw_side,key,bad):
    obj=dict(producer_payloads[0 if raw_side else 1]);obj[key]=bad
    g=ProtectionMonitor(record(),settings())
    method=g.ingest_raw if raw_side else g.ingest_base
    with pytest.raises(ProtectionFault,match='contract mismatch'):
        method(obj,10.,obj['source_ros_ns']+1_000_000)


def test_real_producer_payload_reaches_executor_then_source_loss_stops(monkeypatch,approved_mock,producer_payloads):
    c,r=approved_mock;n,t,future=protected_bridge(monkeypatch,c,r)
    raw,base=producer_payloads[:2]
    t[0]=(raw['source_ros_ns']+1_000_000)/1e9
    n.pose_at=n.force_at=n.mode_at=n.provider_at=t[0]
    n.state='waiting_feedback'
    n.on_acquisition(NS(data=json.dumps(raw)))
    n.on_wrench_provenance(NS(data=json.dumps(base)))
    assert n.protection_check(t[0])
    n.state='running';n.tick()
    assert n.sent and n.sent[-1][8]==0.
    t[0]+=.2;n.pose_at=n.force_at=n.mode_at=n.provider_at=t[0]
    n.tick()
    assert n.state=='stopping' and n.engine.closed and n.engine.plan is None
    assert n.requests[-1].command_mode=='PTP9D_STREAM_STOP'
    assert n.modes[-1]=='Idling'
    assert not n.statuses or not n.statuses[-1]['physical_stop_verified']


def test_A_operator_sequence_status_log_and_verified_stop(monkeypatch,approved_mock):
    c,r=approved_mock;n,t,future=protected_bridge(monkeypatch,c,r)
    n.state='waiting_feedback';source_callbacks(n,t);assert n.protection_check(t[0])
    n.state='running';n.status()
    assert n.statuses[-1]['external_force_fz_N']==0.
    assert not n.operator_event('processing_end',NS()).success
    assert n.operator_event('processing_start',NS()).success
    assert n.statuses[-1]['state']=='running' and n.statuses[-1]['processing']
    assert n.statuses[-1]['force_source']=='external_F0'
    assert n.statuses[-1]['external_force_fz_N']==23.
    assert not n.operator_event('processing_start',NS()).success
    assert not n.operator_event('phase_next',NS()).success  # single reviewed phase
    assert n.operator_event('processing_end',NS()).success
    assert n.statuses[-1]['external_force_fz_N']==0.
    assert not n.statuses[-1]['processing']
    assert not n.operator_event('processing_end',NS()).success
    assert not n.operator_event('processing_start',NS()).success
    assert n.operator_event('finish',NS()).success
    assert n.statuses[-1]['state']=='stopping' and not n.statuses[-1]['physical_stop_verified']
    assert not n.statuses[-1]['command_supply_active']
    assert not n.operator_event('finish',NS()).success
    assert not n.operator_event('abort',NS()).success
    future.cb(future);n.status()
    assert n.statuses[-1]['queue_cancel_ack'] and not n.statuses[-1]['physical_stop_verified']
    for sequence in range(2,62):
        t[0]+=.01;n.pose_at=n.force_at=n.mode_at=t[0];n.mode='Position'
        source_callbacks(n,t,sequence);n.tick()
    assert n.state=='stopped' and n.statuses[-1]['physical_stop_verified']
    assert any(e['event']=='physical_hold_verified' for e in n.events)
    assert any(e['event']=='operator_event_rejected' for e in n.events)
    # Exercise the actual event logger, which the bridge otherwise intercepts.
    captured=[];n.recorder.event=lambda *a,**kw:captured.append(kw);n.timing=lambda:dict()
    MethodType(E2Executor.event,n)('review_test')
    assert captured[-1]['state']=='stopped' and captured[-1]['processing'] is False
    assert captured[-1]['force_source']=='external_F0'


@pytest.mark.parametrize('state',['waiting_feedback','ready','arming','stopping','stop_failed','stopped'])
def test_phase_and_processing_events_require_running(monkeypatch,approved_mock,state):
    c,r=approved_mock;n,t,f=protected_bridge(monkeypatch,c,r)
    n.state=state;n.processing=True;n.phase_index=0
    n.config['protocol']['phase_ids']=['MOCK_phase_1','MOCK_phase_2']
    for event in ('phase_next','processing_start','processing_end','finish'):
        assert not n.operator_event(event,NS()).success
    assert n.phase_index==0 and n.processing is True
    assert not n.sent and not n.requests
