#!/usr/bin/env python3
"""Read-only E2 feedback diagnosis. Publishes no topics or service requests."""
import argparse
import json
import time

import rclpy
from rclpy.qos import QoSProfile, ReliabilityPolicy
from std_msgs.msg import Float64MultiArray, String


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--duration', type=float, default=5.)
    args = parser.parse_args()
    rclpy.init()
    node = rclpy.create_node('e2_feedback_diagnosis')
    observations = {}
    topics = {'/ur10skku/currentP': Float64MultiArray, '/ur10skku/currentF': Float64MultiArray,
              '/ur10skku/ctlMode': String, '/ur10skku/ft_acquisition': String,
              '/ur10skku/currentF_provenance': String}
    def receive(topic, message):
        obj = observations.setdefault(topic, dict(count=0, source_ages_s=[]))
        obj['count'] += 1
        if topic.endswith(('ft_acquisition', 'currentF_provenance')):
            try:
                packet = json.loads(message.data)
                source = packet.get('source_ros_ns')
                if isinstance(source, int):
                    obj['source_ages_s'].append((node.get_clock().now().nanoseconds-source)/1e9)
                obj['last_packet'] = packet
            except (ValueError, TypeError):
                obj['invalid_json'] = True
        else:
            obj['last'] = message.data if isinstance(message.data, str) else list(message.data)
    qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)
    for topic, kind in topics.items():
        node.create_subscription(kind, topic, lambda message, t=topic: receive(t, message), qos)
    stop = time.monotonic()+args.duration
    try:
        while time.monotonic() < stop:
            rclpy.spin_once(node, timeout_sec=.1)
        for topic in topics:
            obj = observations.setdefault(topic, dict(count=0, source_ages_s=[]))
            ages = obj.pop('source_ages_s')
            obj['source_age_range_s'] = [min(ages), max(ages)] if ages else None
            obj['fresh_source_samples'] = sum(0 <= age <= .2 for age in ages)
            obj['publishers'] = [dict(node=info.node_namespace.rstrip('/')+'/'+info.node_name,
                reliability=info.qos_profile.reliability.name,
                durability=info.qos_profile.durability.name)
                for info in node.get_publishers_info_by_topic(topic)]
        print(json.dumps(observations, indent=2))
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
