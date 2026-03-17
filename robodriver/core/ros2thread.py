import threading
import time

from typing import List, Optional

import rospy

import logging_mp

logger = logging_mp.get_logger(__name__)


class ROS2_NodeManager():
    def __init__(self):
        """初始化 ROS1 节点管理器（保留历史类名以兼容旧代码）"""
        self._lock = threading.Lock()
        self._nodes: List[object] = []
        self._spin_thread: Optional[threading.Thread] = None
        self.running = False
        self._initialized = False

        self._init_ros1()

        self._initialized = True

    def _init_ros1(self):
        """初始化 ROS1（线程安全）"""
        with self._lock:
            if not self._initialized:
                if not rospy.core.is_initialized():
                    rospy.init_node("roboautotask_node_manager", anonymous=True, disable_signals=True)
                self._initialized = True
                logger.info("[ROS1] rospy initialized")

    def _node_name(self, node: object) -> str:
        for attr in ("get_name", "name", "node_name"):
            if hasattr(node, attr):
                value = getattr(node, attr)
                try:
                    return value() if callable(value) else str(value)
                except Exception:
                    continue
        return node.__class__.__name__

    def add_node(self, node: object):
        """添加节点/对象到管理器（ROS1下主要用于统一生命周期管理）"""
        self._init_ros1()
        
        with self._lock:
            if node not in self._nodes:
                self._nodes.append(node)
                logger.info(f"[ROS1] Node '{self._node_name(node)}' added")

    def remove_node(self, node: object):
        """从管理器移除节点/对象"""
        with self._lock:
            if node in self._nodes:
                self._nodes.remove(node)
                if hasattr(node, "destroy_node"):
                    try:
                        node.destroy_node()
                    except Exception as e:
                        logger.warning(f"[ROS1] destroy_node failed: {e}")
                elif hasattr(node, "destroy"):
                    try:
                        node.destroy()
                    except Exception as e:
                        logger.warning(f"[ROS1] destroy failed: {e}")
                logger.info(f"[ROS1] Node '{self._node_name(node)}' removed")

    def create_node(self, node_name: str, **kwargs):
        """ROS1 下不提供动态创建 Node API，保留接口兼容。"""
        raise NotImplementedError("ROS1 mode does not support create_node in this manager")

    def start(self):
        """启动 ROS1 管理线程（ROS1 回调由 rospy 内部线程处理）"""
        if self.running:
            logger.warning("[ROS1] Already running")
            return

        if not self._initialized:
            self._init_ros1()

        if not self._nodes:
            logger.warning("[ROS1] No nodes registered")

        self.running = True
        self._spin_thread = threading.Thread(target=self._spin_loop, daemon=True)
        self._spin_thread.start()
        logger.info("[ROS1] Manager loop started")

    def _spin_loop(self):
        """独立线程维持运行状态（ROS1 无需显式 spin_once）"""
        try:
            while self.running and (not rospy.is_shutdown()):
                time.sleep(0.01)
        except Exception as e:
            logger.error(f"[ROS1] Loop error: {e}")
        finally:
            self._cleanup()

    def _cleanup(self):
        """清理资源"""
        with self._lock:
            # 移除所有节点
            for node in self._nodes[:]:  # 使用副本遍历
                self.remove_node(node)

            # 不主动 shutdown ROS1，避免误关同进程其他组件
            self._initialized = rospy.core.is_initialized()
            
            self.running = False
            logger.info("[ROS1] Cleanup completed")

    def stop(self):
        """停止管理器"""
        if not self.running:
            return

        self.running = False
        
        # 等待spin线程结束
        if self._spin_thread is not None:
            self._spin_thread.join(timeout=2.0)
            if self._spin_thread.is_alive():
                logger.warning("[ROS1] Manager loop did not exit cleanly")

        logger.info("[ROS1] Node manager stopped")

    def get_nodes(self) -> List[object]:
        """获取所有节点列表"""
        with self._lock:
            return self._nodes.copy()

    def __del__(self):
        """析构函数，确保资源清理"""
        if self.running:
            self.stop()


class ROS1_NodeManager(ROS2_NodeManager):
    """ROS1 别名类，便于新代码使用。"""
    pass
