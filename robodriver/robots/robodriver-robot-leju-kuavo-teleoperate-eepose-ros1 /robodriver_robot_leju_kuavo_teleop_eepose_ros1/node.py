#!/usr/bin/env python
# -*- coding: utf-8 -*-
import threading
import time
from typing import Dict

import numpy as np
import cv2
import rospy
from sensor_msgs.msg import JointState, Image
from std_msgs.msg import Float32MultiArray
from message_filters import Subscriber, ApproximateTimeSynchronizer

try:
    # 优先使用ROS原生消息包，确保与rostopic type一致（kuavo_msgs/*）
    from kuavo_msgs.msg import robotHandPosition, robotHeadMotionData, sensorsData

except Exception:
    # 兼容旧SDK导入路径
    from kuavo_humanoid_sdk.msg.kuavo_msgs.msg import robotHandPosition, robotHeadMotionData, sensorsData

try:
    # IK官方接口优先使用 motion_capture_ik 包
    from motion_capture_ik.msg import twoArmHandPose, twoArmHandPoseCmd
except Exception:
    try:
        from kuavo_msgs.msg import twoArmHandPose, twoArmHandPoseCmd
    except Exception:
        from kuavo_humanoid_sdk.msg.kuavo_msgs.msg import twoArmHandPose, twoArmHandPoseCmd




# ROS1没有logging_mp，替换为标准logging
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

CONNECT_TIMEOUT_FRAME = 10
MAX_SYNC_ERROR_SEC = 0.01


class LEJUKuavoEEPOSERos1Node:
    def __init__(self):
        # ROS1节点初始化 - 检查是否已经初始化
        if not rospy.core.is_initialized():
            rospy.init_node('ros1_recv_pub_driver', anonymous=False, log_level=rospy.INFO)
            rospy.loginfo("ROS1节点初始化完成，开始设置话题订阅和发布")
        else:
            rospy.loginfo(f"ROS节点已存在，复用现有节点: {rospy.get_name()}")
        
        self.stop_spin = False  # 初始化停止标志
        
        # ROS1没有QoSProfile类，直接在订阅时指定队列大小，可靠性通过传输方式保证
        self.queue_size = 100
        self.best_effort_queue_size = 100

        # EEPose控制优先通过IK命令话题下发
        rospy.loginfo(f"创建发布者: /ik/two_arm_hand_pose_cmd (队列大小: {self.queue_size})")
        self.publisher_arm_eepose = rospy.Publisher('/ik/two_arm_hand_pose_cmd', twoArmHandPoseCmd, queue_size=self.queue_size)
        
        rospy.loginfo(f"创建发布者: /control_robot_hand_position (队列大小: {self.queue_size})")
        self.publisher_hand_eepose = rospy.Publisher('/control_robot_hand_position', robotHandPosition, queue_size=self.queue_size)

        # IK求解结果订阅（用于回读末端位姿）
        rospy.loginfo(f"创建订阅者: /ik/result (队列大小: {self.queue_size})")
        self.ik_result_sub = rospy.Subscriber('/ik/result', twoArmHandPose, self.ik_result_callback, queue_size=self.queue_size)

        rospy.loginfo("创建订阅者: /ik/debug/time_cost")
        self.ik_time_cost_sub = rospy.Subscriber('/ik/debug/time_cost', Float32MultiArray, self.ik_time_cost_callback, queue_size=10)

        rospy.loginfo("创建订阅者: /kuavo_arm_traj")
        self.ik_joint_traj_sub = rospy.Subscriber('/kuavo_arm_traj', JointState, self.ik_joint_traj_callback, queue_size=self.queue_size)
        
        self.last_main_send_time_ns = 0
        self.last_follow_send_time_ns = 0
        self.min_interval_ns = 1e9 / 30  # 30Hz
        self.lock = threading.Lock()
        self.recv_images: Dict[str, np.ndarray] = {}
        self.recv_follower: Dict[str, np.ndarray] = {}
        self.recv_images_status: Dict[str, int] = {}
        self.recv_follower_status: Dict[str, int] = {}
        self.latest_body_msg = None
        self.latest_hand_msg = None
        self.body_msg_count = 0
        self.hand_msg_count = 0
        self.ik_result_count = 0
        self._ik_result_received = False
        self.last_ik_loop_ms = None
        self.last_ik_solve_ms = None
        self._warned_headerless_follow = False
        self._warned_headerless_image = False

        rospy.loginfo("初始化消息过滤器...")
        self._init_message_follow_filters()
        self._init_image_message_filters()
        rospy.loginfo("节点初始化完成，等待话题消息...")

    def _fill_single_hand_pose(self, pose_obj, pos_xyz, quat_xyzw):
        pose_obj.pos_xyz = list(pos_xyz)
        pose_obj.quat_xyzw = list(quat_xyzw)
        pose_obj.elbow_pos_xyz = [0.0, 0.0, 0.0]
        pose_obj.joint_angles = [0.0] * 7

    def _fill_two_arm_pose_msg(self, pose_msg, left_pos, left_quat, right_pos, right_quat):
        # 先按文档中的常见命名进行固定赋值
        if hasattr(pose_msg, "left_hand_pose") and hasattr(pose_msg, "right_hand_pose"):
            self._fill_single_hand_pose(pose_msg.left_hand_pose, left_pos, left_quat)
            self._fill_single_hand_pose(pose_msg.right_hand_pose, right_pos, right_quat)
            return

        if hasattr(pose_msg, "left") and hasattr(pose_msg, "right"):
            self._fill_single_hand_pose(pose_msg.left, left_pos, left_quat)
            self._fill_single_hand_pose(pose_msg.right, right_pos, right_quat)
            return

        raise AttributeError(f"twoArmHandPose字段不匹配: {list(getattr(pose_msg, '__slots__', []))}")

    def _extract_two_arm_pose_msg(self, pose_msg):
        if hasattr(pose_msg, "left_hand_pose") and hasattr(pose_msg, "right_hand_pose"):
            return pose_msg.left_hand_pose, pose_msg.right_hand_pose
        if hasattr(pose_msg, "left") and hasattr(pose_msg, "right"):
            return pose_msg.left, pose_msg.right
        raise AttributeError(f"twoArmHandPose字段不匹配: {list(getattr(pose_msg, '__slots__', []))}")

    def _extract_single_hand_pose(self, pose_obj):
        pos = np.array(list(getattr(pose_obj, "pos_xyz", [0.0, 0.0, 0.0]))[:3], dtype=np.float32)
        quat = np.array(list(getattr(pose_obj, "quat_xyzw", [0.0, 0.0, 0.0, 1.0]))[:4], dtype=np.float32)
        elbow = np.array(list(getattr(pose_obj, "elbow_pos_xyz", [0.0, 0.0, 0.0]))[:3], dtype=np.float32)
        joints = np.array(list(getattr(pose_obj, "joint_angles", [0.0] * 7))[:7], dtype=np.float32)
        if pos.shape[0] < 3:
            pos = np.pad(pos, (0, 3 - pos.shape[0]))
        if quat.shape[0] < 4:
            quat = np.pad(quat, (0, 4 - quat.shape[0]))
            quat[3] = 1.0
        if elbow.shape[0] < 3:
            elbow = np.pad(elbow, (0, 3 - elbow.shape[0]))
        if joints.shape[0] < 7:
            joints = np.pad(joints, (0, 7 - joints.shape[0]))
        return pos, quat, elbow, joints

    def ik_result_callback(self, msg):
        try:
            self.ik_result_count += 1
            if self.ik_result_count == 1:
                rospy.loginfo("首次收到 /ik/result")

            left_pose, right_pose = self._extract_two_arm_pose_msg(msg)
            l_pos, l_quat, l_elbow, l_joints = self._extract_single_hand_pose(left_pose)
            r_pos, r_quat, r_elbow, r_joints = self._extract_single_hand_pose(right_pose)

            left_eepose = np.concatenate([l_pos, l_quat]).astype(np.float32)
            right_eepose = np.concatenate([r_pos, r_quat]).astype(np.float32)

            with self.lock:
                # EEPose链路: left_arm/right_arm 保存 [x,y,z,qx,qy,qz,qw]
                self.recv_follower['left_arm'] = left_eepose
                self.recv_follower['right_arm'] = right_eepose
                self.recv_follower['left_elbow'] = l_elbow
                self.recv_follower['right_elbow'] = r_elbow
                self.recv_follower['left_arm_joint'] = l_joints
                self.recv_follower['right_arm_joint'] = r_joints

                self.recv_follower_status['left_arm'] = CONNECT_TIMEOUT_FRAME
                self.recv_follower_status['right_arm'] = CONNECT_TIMEOUT_FRAME
                self.recv_follower_status['left_elbow'] = CONNECT_TIMEOUT_FRAME
                self.recv_follower_status['right_elbow'] = CONNECT_TIMEOUT_FRAME
                self.recv_follower_status['left_arm_joint'] = CONNECT_TIMEOUT_FRAME
                self.recv_follower_status['right_arm_joint'] = CONNECT_TIMEOUT_FRAME

                self._ik_result_received = True
        except Exception as e:
            rospy.logwarn(f"ik_result_callback error: {e}")

    def ik_time_cost_callback(self, msg):
        try:
            data = list(getattr(msg, "data", []))
            if len(data) >= 2:
                self.last_ik_loop_ms = float(data[0])
                self.last_ik_solve_ms = float(data[1])
                rospy.logdebug_throttle(2.0, f"IK耗时: loop={self.last_ik_loop_ms:.2f}ms, solve={self.last_ik_solve_ms:.2f}ms")
        except Exception as e:
            rospy.logwarn(f"ik_time_cost_callback error: {e}")

    def ik_joint_traj_callback(self, msg):
        try:
            pos = np.array(list(getattr(msg, "position", [])), dtype=np.float32)
            if pos.shape[0] >= 14:
                with self.lock:
                    self.recv_follower['left_arm_joint_traj'] = pos[0:7]
                    self.recv_follower['right_arm_joint_traj'] = pos[7:14]
                    self.recv_follower_status['left_arm_joint_traj'] = CONNECT_TIMEOUT_FRAME
                    self.recv_follower_status['right_arm_joint_traj'] = CONNECT_TIMEOUT_FRAME
        except Exception as e:
            rospy.logwarn(f"ik_joint_traj_callback error: {e}")

    def _init_message_follow_filters(self):
        sub_body = Subscriber('/sensors_data_raw', sensorsData)
        sub_hand = Subscriber('/dexhand/state', JointState)

        self.follow_sync = ApproximateTimeSynchronizer(
            [sub_body, sub_hand],
            queue_size=50,
            slop=MAX_SYNC_ERROR_SEC,
            allow_headerless=True,
        )
        self.follow_sync.registerCallback(self.synchronized_follow_callback)
        rospy.loginfo(
            f"已启用从臂话题同步: /sensors_data_raw + /dexhand/state, 最大时间误差={MAX_SYNC_ERROR_SEC:.3f}s"
        )

    def _get_msg_stamp_sec(self, msg):
        header = getattr(msg, 'header', None)
        stamp = getattr(header, 'stamp', None)
        if stamp is None:
            return None
        try:
            return stamp.to_sec()
        except Exception:
            return None

    def _process_body_msg(self, body_msg):
        try:
            joint_data = getattr(body_msg, "joint_data", None)
            if joint_data is None:
                return

            if hasattr(joint_data, "position"):
                body_pos = joint_data.position
            elif hasattr(joint_data, "joint_q"):
                body_pos = joint_data.joint_q
            else:
                return

            if len(body_pos) < 26:
                rospy.logwarn(f"Body joint data too short for arms: {len(body_pos)}")
                return

            left_arm = np.array(body_pos[12:19], dtype=np.float32)
            right_arm = np.array(body_pos[19:26], dtype=np.float32)
            if len(body_pos) >= 28:
                head = np.array(body_pos[26:28], dtype=np.float32)
            else:
                head = np.zeros(2, dtype=np.float32)

            with self.lock:
                # 未收到IK结果前，使用关节角作为占位；收到后不再覆盖EEPose
                if not self._ik_result_received:
                    self.recv_follower['right_arm'] = right_arm
                    self.recv_follower['left_arm'] = left_arm
                self.recv_follower['head'] = head
                if not self._ik_result_received:
                    self.recv_follower_status['right_arm'] = CONNECT_TIMEOUT_FRAME
                    self.recv_follower_status['left_arm'] = CONNECT_TIMEOUT_FRAME
                self.recv_follower_status['head'] = CONNECT_TIMEOUT_FRAME
        except Exception as e:
            rospy.logwarn(f"_process_body_msg error: {e}")

    def _process_hand_msg(self, hand_msg):
        try:
            hand_pos = list(getattr(hand_msg, "position", []))
            if len(hand_pos) < 12:
                rospy.logwarn(f"Hand position length < 12: {len(hand_pos)}")
                hand_pos = hand_pos + [0.0] * (12 - len(hand_pos))

            left_dexhand = np.array(hand_pos[0:6], dtype=np.float32)
            right_dexhand = np.array(hand_pos[6:12], dtype=np.float32)

            with self.lock:
                self.recv_follower['right_dexhand'] = right_dexhand
                self.recv_follower['left_dexhand'] = left_dexhand
                self.recv_follower_status['right_dexhand'] = CONNECT_TIMEOUT_FRAME
                self.recv_follower_status['left_dexhand'] = CONNECT_TIMEOUT_FRAME
        except Exception as e:
            rospy.logwarn(f"_process_hand_msg error: {e}")

    def synchronized_follow_callback(self, body_msg, hand_msg):
        try:
            self.body_msg_count += 1
            self.hand_msg_count += 1
            if self.body_msg_count == 1:
                rospy.loginfo("首次收到 /sensors_data_raw")
            if self.hand_msg_count == 1:
                rospy.loginfo("首次收到 /dexhand/state")

            body_stamp = self._get_msg_stamp_sec(body_msg)
            hand_stamp = self._get_msg_stamp_sec(hand_msg)
            if body_stamp is not None and hand_stamp is not None:
                sync_error = abs(body_stamp - hand_stamp)
                if sync_error > MAX_SYNC_ERROR_SEC:
                    rospy.logwarn_throttle(
                        1.0,
                        f"从臂话题时间误差超限: {sync_error:.6f}s > {MAX_SYNC_ERROR_SEC:.3f}s"
                    )
                    return
            elif not self._warned_headerless_follow:
                rospy.logwarn("从臂话题缺少header.stamp，已使用接收时间近似同步，无法严格保证0.01s时间误差")
                self._warned_headerless_follow = True

            joint_data = getattr(body_msg, "joint_data", None)
            if joint_data is None:
                rospy.logwarn("body_msg 中不存在 joint_data 字段")
                return

            # 兼容不同消息定义：position 或 joint_q
            if hasattr(joint_data, "position"):
                body_pos = joint_data.position
            elif hasattr(joint_data, "joint_q"):
                body_pos = joint_data.joint_q
            else:
                rospy.logwarn("joint_data 不包含 position/joint_q 字段")
                return

            # 打印接收到的消息信息
            #rospy.loginfo(f"Received follow data: body_msg joints={len(body_pos)}, hand_msg positions={len(hand_msg.position)}")
            
            # 检测话题数据
            if len(body_pos) > 0:
                # 打印前几个关节位置作为示例
                sample_positions = body_pos[:5] if len(body_pos) >= 5 else body_pos
                #rospy.loginfo(f"Body joint sample positions: {sample_positions}")
                
                # 检测数据范围
                for i, pos in enumerate(body_pos):
                    if abs(pos) > 1000:  # 假设关节位置不应该超过1000
                        rospy.logwarn(f"Joint {i} position out of range: {pos}")
            
            if len(hand_msg.position) > 0:
                sample_hand = hand_msg.position[:3] if len(hand_msg.position) >= 3 else hand_msg.position
                #rospy.loginfo(f"Hand position sample: {sample_hand}")
                
                # 检测手部数据范围
                for i, pos in enumerate(hand_msg.position):
                    if pos < 0 or pos > 100:  # 假设手部位置在0-100范围内
                        rospy.logwarn(f"Hand position {i} out of range [0,100]: {pos}")
            
            # 独立更新，避免双话题严格同步导致无数据
            self._process_body_msg(body_msg)
            self._process_hand_msg(hand_msg)

            current_time_ns = time.time_ns()
            if (current_time_ns - self.last_follow_send_time_ns) >= self.min_interval_ns:
                self.last_follow_send_time_ns = current_time_ns
        except Exception as e:
            rospy.logerr(f"Synchronized follow callback error: {e}")

    def _init_image_message_filters(self):
        sub_camera_top = Subscriber('/camera/color/image_raw', Image)
        sub_camera_wrist_left = Subscriber('/left_wrist_camera/color/image_raw', Image)
        sub_camera_wrist_right = Subscriber('/right_wrist_camera/color/image_raw', Image)
 
        self.image_sync = ApproximateTimeSynchronizer(
            [sub_camera_top, sub_camera_wrist_left, sub_camera_wrist_right],
            queue_size=10,
            slop=MAX_SYNC_ERROR_SEC,
            allow_headerless=True,
        )
        self.image_sync.registerCallback(self.image_synchronized_callback)

    def image_synchronized_callback(self, top, wrist_left, wrist_right):
        try:
            top_stamp = self._get_msg_stamp_sec(top)
            left_stamp = self._get_msg_stamp_sec(wrist_left)
            right_stamp = self._get_msg_stamp_sec(wrist_right)
            stamps = [s for s in [top_stamp, left_stamp, right_stamp] if s is not None]
            if len(stamps) == 3:
                sync_error = max(stamps) - min(stamps)
                if sync_error > MAX_SYNC_ERROR_SEC:
                    rospy.logwarn_throttle(
                        1.0,
                        f"图像话题时间误差超限: {sync_error:.6f}s > {MAX_SYNC_ERROR_SEC:.3f}s"
                    )
                    return
            elif not self._warned_headerless_image:
                rospy.logwarn("图像话题缺少header.stamp，已使用接收时间近似同步，无法严格保证0.01s时间误差")
                self._warned_headerless_image = True

            # 打印接收到的图像消息信息
            #rospy.loginfo(f"Received synchronized images: top={top.header.seq if top.header.seq else 'N/A'}, "
                         #f"wrist_left={wrist_left.header.seq if wrist_left.header.seq else 'N/A'}, "
                         #f"wrist_right={wrist_right.header.seq if wrist_right.header.seq else 'N/A'}")
            
            # 检测图像数据
            #rospy.loginfo(f"Image data sizes: top={len(top.data)} bytes, "
                         #f"wrist_left={len(wrist_left.data)} bytes, "
                         #f"wrist_right={len(wrist_right.data)} bytes")
            
            # 检查图像数据是否为空
            if len(top.data) == 0:
                rospy.logwarn("Top camera image data is empty!")
            if len(wrist_left.data) == 0:
                rospy.logwarn("Left wrist camera image data is empty!")
            if len(wrist_right.data) == 0:
                rospy.logwarn("Right wrist camera image data is empty!")
            
            self.images_recv(top, "image_top", 424, 240)
            self.images_recv(wrist_left, "image_wrist_left", 848, 480)
            self.images_recv(wrist_right, "image_wrist_right", 848, 480)
        except Exception as e:
            rospy.logerr(f"Image synchronized callback error: {e}")
    
    def images_recv(self, msg, event_id, width, height, encoding="jpeg"):
        try:
           
            
            
           
            
            if 'image' in event_id:
                img_array = np.frombuffer(msg.data, dtype=np.uint8)
                
                # 检测图像数据大小是否合理
                expected_size = width * height * 3  # 假设RGB图像
                if len(img_array) < expected_size * 0.5:  # 至少50%的预期大小
                    rospy.logwarn(f"Image data size suspiciously small: {len(img_array)} bytes, "
                                 f"expected around {expected_size} bytes for {width}x{height} RGB image")
                
                if msg.encoding == "bgr8":
                    channels = 3
                    frame = img_array.reshape((height, width, channels)).copy()
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                elif msg.encoding == "rgb8":
                    channels = 3
                    frame = img_array.reshape((height, width, channels))
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                elif msg.encoding in ["jpeg", "jpg", "jpe", "bmp", "webp", "png"]:
                    channels = 3
                    frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                elif msg.encoding == "depth16":
                    frame = np.frombuffer(msg.data, dtype=np.uint16).reshape(height, width, 1)
                else:
                    # 尝试通用解码
                    frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                    if frame is not None:
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                if frame is not None:
                    # 修正 top 相机画面方向（当前倒置）
                    if event_id == "image_top":
                        frame = cv2.rotate(frame, cv2.ROTATE_180)

                    # 打印图像处理结果
                    #rospy.loginfo(f"Successfully decoded image {event_id}: shape={frame.shape}, dtype={frame.dtype}")
                    
                    # 检测图像质量
                    if frame.size > 0:
                        mean_val = np.mean(frame)
                        #rospy.loginfo(f"Image {event_id} mean pixel value: {mean_val:.2f}")
                        
                        # 检查图像是否全黑或全白
                        if mean_val < 10:
                            rospy.logwarn(f"Image {event_id} might be too dark (mean={mean_val:.2f})")
                        elif mean_val > 245:
                            rospy.logwarn(f"Image {event_id} might be too bright (mean={mean_val:.2f})")
                    
                    with self.lock:
                        self.recv_images[event_id] = frame
                        self.recv_images_status[event_id] = CONNECT_TIMEOUT_FRAME
                else:
                    rospy.logwarn(f"Failed to decode image for {event_id} with encoding {msg.encoding}")
        except Exception as e:
            logger.error(f"recv image error: {e}")
            rospy.logerr(f"Error processing image {event_id}: {e}")

    def ros_replay(self, array):
        try:
            def normalize_precision(val, decimals=3):
                val = float(val)
                if np.isnan(val) or np.isinf(val):
                    rospy.logwarn(f"检测到非法值 {val}，替换为0.0")
                    return 0.0
                return round(val, decimals)

            def to_uint_list(values, vmin=0, vmax=100):
                out = []
                for v in values:
                    fv = normalize_precision(v)
                    fv = max(vmin, min(vmax, fv))
                    out.append(int(round(fv)))
                return out

            # EEPose动作格式与星海图一致: 14维位姿 + 可选12维手部
            if len(array) >= 26:
                left_arm_pose = [normalize_precision(v) for v in array[0:7]]
                right_arm_pose = [normalize_precision(v) for v in array[7:14]]
                left_dexhand = to_uint_list(array[14:20])
                right_dexhand = to_uint_list(array[20:26])
            else:
                raise ValueError(f"Action vector too short: {len(array)}")

            left_pos = left_arm_pose[0:3]
            left_quat = left_arm_pose[3:7]
            right_pos = right_arm_pose[0:3]
            right_quat = right_arm_pose[3:7]

            # --- IK命令控制（双臂EEPose） ---
            ik_cmd = twoArmHandPoseCmd()
            if hasattr(ik_cmd, "header"):
                ik_cmd.header.stamp = rospy.Time.now()
            if hasattr(ik_cmd, "use_custom_ik_param"):
                ik_cmd.use_custom_ik_param = False
            if hasattr(ik_cmd, "joint_angles_as_q0"):
                ik_cmd.joint_angles_as_q0 = False

            pose_msg = twoArmHandPose()
            self._fill_two_arm_pose_msg(pose_msg, left_pos, left_quat, right_pos, right_quat)

            if hasattr(ik_cmd, "hand_poses"):
                ik_cmd.hand_poses = pose_msg
            elif hasattr(ik_cmd, "two_arm_hand_pose"):
                ik_cmd.two_arm_hand_pose = pose_msg
            else:
                raise AttributeError(f"twoArmHandPoseCmd字段不匹配: {list(getattr(ik_cmd, '__slots__', []))}")

            self.publisher_arm_eepose.publish(ik_cmd)

            # --- 手部控制 ---
            hand_msg = robotHandPosition()
            hand_msg.left_hand_position = left_dexhand    # 6维 [0~100]
            hand_msg.right_hand_position = right_dexhand  # 6维 [0~100]
            self.publisher_hand_eepose.publish(hand_msg)

        except Exception as e:
            rospy.logerr(f"Error during replay at frame: {e}")
            raise

    def destroy(self):
        self.stop_spin = True
        rospy.signal_shutdown("Node shutdown requested")


# ROS1的spin线程函数
def ros_spin_thread(node):
    while not rospy.is_shutdown() and not getattr(node, "stop_spin", False):
        try:
            rospy.Rate(100).sleep()
        except rospy.ROSInterruptException:
            break


