#!/usr/bin/env python
# -*- coding: utf-8 -*-
import threading
import time
from typing import Dict

import numpy as np
import cv2
import rospy
from sensor_msgs.msg import JointState, Image
from message_filters import Subscriber, ApproximateTimeSynchronizer
from geometry_msgs.msg import PoseStamped

# ROS1没有logging_mp，替换为标准logging
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

CONNECT_TIMEOUT_FRAME = 10


class LEJUKuavoRos1RobotNode:
    def __init__(self):
        # ROS1节点初始化
        rospy.init_node('ros1_recv_pub_driver', anonymous=False)
        self.stop_spin = False  # 初始化停止标志
        
        # ROS1没有QoSProfile类，直接在订阅时指定队列大小，可靠性通过传输方式保证
        self.queue_size = 10
        self.best_effort_queue_size = 10

        # 创建发布者（ROS1不需要显式QoS配置，通过队列大小控制）
        self.publisher_left_arm = rospy.Publisher(
            "/kuavo_arm_traj", JointState, queue_size=self.queue_size
        )
        self.publisher_right_arm = rospy.Publisher(
            "/kuavo_arm_traj", JointState, queue_size=self.queue_size
        )
        self.publisher_left_hand = rospy.Publisher(
            "/control_robot_hand_position",robotHandPosition , queue_size=self.queue_size
        )
        self.publisher_right_hand = rospy.Publisher(
            "/control_robot_hand_position",robotHandPosition , queue_size=self.queue_size
        )
        self.publisher_head = rospy.Publisher(
            "/robot_head_motion_data", robotHeadMotionData, queue_size=self.queue_size
        )
     
        self.last_main_send_time_ns = 0
        self.last_follow_send_time_ns = 0
        self.min_interval_ns = 1e9 / 30  # 30Hz
        self.lock = threading.Lock()
        self.recv_images: Dict[str, float] = {}
        self.recv_leader: Dict[str, float] = {}
        self.recv_follower: Dict[str, float] = {}
        self.recv_images_status: Dict[str, int] = {}
        self.recv_leader_status: Dict[str, int] = {}
        self.recv_follower_status: Dict[str, int] = {}

        self._init_message_main_filters()
        self._init_message_follow_filters()
        self._init_image_message_filters()



    def _init_message_follow_filters(self):
        # ROS1的message_filters.Subscriber不需要显式指定节点，直接指定话题名
        sub_arm_left , sub_arm_right ,  sub_head = Subscriber("/sensor_data_raw",sensorsData)
        sub_hand_left , sub_hand_right = Subscriber('/dexhand/state',JointState)
        self.sync = ApproximateTimeSynchronizer(
            [sub_arm_left, sub_arm_right, sub_hand_left, sub_hand_right, sub_head],
            queue_size=10,
            slop=0.01
        )
        self.sync.registerCallback(self.synchronized_follow_callback)
 
    def synchronized_follow_callback(self, arm_left, arm_right, hand_left, hand_right, head):
        try:
            current_time_ns = time.time_ns()
            if (current_time_ns - self.last_follow_send_time_ns) < self.min_interval_ns:
                return
            self.last_follow_send_time_ns = current_time_ns
 
            left_pos = np.array(arm_left.position, dtype=np.float32)
            right_pos = np.array(arm_right.position, dtype=np.float32)
            left_arm_data = left_pos
            right_arm_data = right_pos
 
            hand_left_pos = np.array(hand_left.position, dtype=np.float32)
            hand_right_pos = np.array(hand_right.position, dtype=np.float32)
            head_pos = np.array(head.position, dtype=np.float32)
            head_pos = head_pos[:-1]
           
            merged_data = np.concatenate([left_arm_data, hand_left_pos, right_arm_data, hand_right_pos, head_pos])
            
            with self.lock:
                self.recv_follower['follower_arms'] = merged_data
                self.recv_follower_status['follower_arms'] = CONNECT_TIMEOUT_FRAME
        except Exception as e:
            rospy.logerr(f"Synchronized follow callback error: {e}")

        def _init_image_message_filters(self):
         sub_camera_top = Subscriber('/camera/color/image_raw', Image)
         sub_camera_wrist_left = Subscriber('/right_wrist_camera/color/image_raw', Image)
         sub_camera_wrist_right = Subscriber('/left_wrist_camera/color/image_raw', Image)
 
         self.image_sync = ApproximateTimeSynchronizer(
            [sub_camera_top, sub_camera_wrist_left, sub_camera_wrist_right],
            queue_size=10,
            slop=0.1
        )
        self.image_sync.registerCallback(self.image_synchronized_callback)

    def image_synchronized_callback(self, top, wrist_left, wrist_right):
        try:
            self.images_recv(top, "image_top_right", 1280, 720)
            self.images_recv(wrist_left, "image_wrist_left", 640, 360)
            self.images_recv(wrist_right, "image_wrist_right", 640, 360)
        except Exception as e:
            rospy.logerr(f"Image `synchronized` callback error: {e}")
    
    def images_recv(self, msg, event_id, width, height, encoding="jpeg"):
        try:
            if 'image' in event_id:
                img_array = np.frombuffer(msg.data, dtype=np.uint8)
                if encoding == "bgr8":
                    channels = 3
                    frame = img_array.reshape((height, width, channels)).copy()
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                elif encoding == "rgb8":
                    channels = 3
                    frame = img_array.reshape((height, width, channels))
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                elif encoding in ["jpeg", "jpg", "jpe", "bmp", "webp", "png"]:
                    channels = 3
                    frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                elif encoding == "depth16":
                    frame = np.frombuffer(msg.data, dtype=np.uint16).reshape(height, width, 1)
                
                if frame is not None:
                    with self.lock:
                        self.recv_images[event_id] = frame
                        self.recv_images_status[event_id] = CONNECT_TIMEOUT_FRAME
        except Exception as e:
            logger.error(f"recv image error: {e}")

    def ros_replay(self, array):
        try:
            def normalize_precision(val, decimals=3):
                val = float(val)
                if np.isnan(val) or np.isinf(val):
                    rospy.logwarn(f"检测到非法值 {val}，替换为0.0")
                    return 0.0
                return round(val, decimals)
            
            left_arm = [normalize_precision(v) for v in array[:6]]
            left_hand = [normalize_precision(v) for v in array[6:7]]
            right_arm = [normalize_precision(v) for v in array[7:13]]
            right_hand = [normalize_precision(v) for v in array[13:14]]
            head = [normalize_precision(v) for v in array[14:17]]
    
            msg = JointState()
            msg.position = left_arm 
            self.publisher_left_arm.publish(msg)

            msg = JointState()
            msg.position = right_arm
            self.publisher_right_arm.publish(msg)

            msg = JointState()
            msg.position = left_hand
            self.publisher_left_hand.publish(msg)

            msg = JointState()
            msg.position = right_hand
            self.publisher_right_hand.publish(msg)

        except Exception as e:
            rospy.logerr(f"Error during replay at frame: {e}")
            raise

    def destroy(self):
        self.stop_spin = True
        # ROS1不需要显式调用destroy_node，关闭节点即可
        rospy.signal_shutdown("Node shutdown requested")

    def _add_debug_subscribers(self):
        # ROS1订阅函数
        rospy.Subscriber(
            '/sensor_data_raw',
            sensorsData,
            lambda msg: rospy.loginfo(f"独立订阅-左臂关节: position={msg.position}"),
            queue_size=self.best_effort_queue_size
        )
        rospy.Subscriber(
            '/dexhand/state',
            JointState,
            lambda msg: rospy.loginfo(f"独立订阅-左手: position={msg.position}"),
            queue_size=self.best_effort_queue_size
        )
        rospy.Subscriber(
            '/sensor_data_raw',
            sensorsData,
            lambda msg: rospy.loginfo(f"独立订阅-右臂关节: position={msg.position}"),
            queue_size=self.best_effort_queue_size
        )
        rospy.Subscriber(
            '/dexhand/state',
            JointState,
            lambda msg: rospy.loginfo(f"独立订阅-右手: position={msg.position}"),
            queue_size=self.best_effort_queue_size
        )



        rospy.Subscriber(
            '/sensor_data_raw',
            sensorsData,
            lambda msg: rospy.loginfo(f"独立订阅-头部joint: position={msg.position}"),
            queue_size=self.best_effort_queue_size
        )

# ROS1的spin线程函数
def ros_spin_thread(node):
    while not rospy.is_shutdown() and not getattr(node, "stop_spin", False):
        try:
            # ROS1的spin_once通过rate控制
            rospy.Rate(100).sleep()  # 100Hz的循环频率
        except rospy.ROSInterruptException:
            break

# 主函数示例
if __name__ == "__main__":
    try:
        node = LEJUKuavoRos1RobotNode()
        # 启动spin线程
        spin_thread = threading.Thread(target=ros_spin_thread, args=(node,))
        spin_thread.start()
        
        # 保持节点运行
        while not rospy.is_shutdown() and not node.stop_spin:
            time.sleep(0.1)
        
        # 关闭节点
        node.destroy()
        spin_thread.join()
    except rospy.ROSInterruptException:
        pass
