from abc import ABC, abstractmethod
from typing import Any, Dict, Literal, Tuple, Optional
import numpy as np
from gymnasium import spaces
import gymnasium as gym
import time
import logging
import os

from gym_hil.envs.camera_orbbec_net import OrbbecCamera as Camera_ex
from gym_hil.envs.camera_orbbec_usb import Camera as Camera_wrist

# 尝试导入 ROS2 Robot 相关模块
try:
    from lerobot_robot_ros2 import ROS2RobotConfig, ROS2Robot, ROS2RobotInterfaceConfig, ControlType
    from lerobot_camera_ros2 import ROS2CameraConfig
    ROS2_AVAILABLE = True
except ImportError:
    ROS2_AVAILABLE = False
    logging.warning("ROS2 Robot modules not available. RealCR5PickCubeGymEnv will not work.")

# 常量定义
_PANDA_HOME = np.asarray((0, 0.195, 0, -2.43, 0, 2.62, 0.785))
_CR5_HOME_POSE = {
    "position": {
        "x": 0.6944005394761854,
        "y": 0.3327791993852288,
        "z": 0.2682457175390543
    },
    "orientation": {
        "x": -0.45627227812278565,
        "y": 0.5583642775482569,
        "z": -0.4770099291881003,
        "w": 0.5025002181398731
    }
}
_CARTESIAN_BOUNDS = np.asarray([[0.2, -0.3, 0], [0.6, 0.3, 0.5]])
_SAMPLING_BOUNDS = np.asarray([[0.3, -0.15], [0.5, 0.15]])


class RealRobotGymEnv(gym.Env, ABC):
    """真实机器人环境的基类，提供与 MuJoCo 环境相同的接口但不依赖 MuJoCo。"""

    def __init__(
        self,
        seed: int = 0,
        control_dt: float = 0.1,
        render_mode: Literal["rgb_array", "human"] = "rgb_array",
        image_obs: bool = False,
        reward_type: str = "sparse",
        random_block_position: bool = False,
        image_height: int = 128,
        image_width: int = 128,
    ):
        super().__init__()

        self.reward_type = reward_type
        self.control_dt = control_dt
        self.render_mode = render_mode
        self.image_obs = image_obs
        self.random_block_position = random_block_position
        self.image_height = image_height
        self.image_width = image_width

        # 初始化随机数生成器
        self.np_random = np.random.RandomState(seed)

        # 任务相关设置
        self._block_z = 0.025  # 方块高度的一半
        self._z_init = None
        self._z_success = None

        # 机器人状态缓存
        self._current_robot_state = None
        self._current_block_position = None

        # 设置观察空间
        self._setup_observation_space()

        # 设置动作空间
        self._setup_action_space()

        # 元数据
        self.metadata = {
            "render_modes": ["human", "rgb_array"],
            "render_fps": int(np.round(1.0 / self.control_dt)),
        }

    def _setup_observation_space(self):
        """设置观察空间"""
        # 机器人状态维度：关节位置(7) + 关节速度(7) + 夹爪位置(1) + TCP位置(3) = 18
        agent_dim = 18
        agent_box = spaces.Box(-np.inf, np.inf, (agent_dim,), dtype=np.float32)
        env_box = spaces.Box(-np.inf, np.inf, (3,), dtype=np.float32)

        if self.image_obs:
            self.observation_space = spaces.Dict(
                {
                    "pixels": spaces.Dict(
                        {
                            "front": spaces.Box(
                                0,
                                255,
                                (self.image_height, self.image_width, 3),
                                dtype=np.uint8,
                            ),
                            "wrist": spaces.Box(
                                0,
                                255,
                                (self.image_height, self.image_width, 3),
                                dtype=np.uint8,
                            ),
                        }
                    ),
                    "agent_pos": agent_box,
                }
            )
        else:
            self.observation_space = spaces.Dict(
                {
                    "agent_pos": agent_box,
                    "environment_state": env_box,
                }
            )

    def _setup_action_space(self):
        """设置动作空间 - 增量控制模式"""
        # 动作空间：位置增量(x, y, z) + 姿态增量(rx, ry, rz) + 夹爪命令(grasp_command)
        # 所有值都是归一化的 [-1, 1]，表示相对当前位置/姿态的增量
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

    @abstractmethod
    def _get_camera_images(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        获取相机图像的虚函数，需要子类实现

        Returns:
            Tuple[np.ndarray, np.ndarray]: (front_view, wrist_view) 两个相机的图像
        """
        pass

    @abstractmethod
    def _send_control_command(self, action: np.ndarray) -> None:
        """
        发送控制指令的虚函数，需要子类实现

        Args:
            action: 控制动作 [x, y, z, rx, ry, rz, grasp_command]
        """
        pass

    @abstractmethod
    def _get_robot_state(self) -> np.ndarray:
        """
        获取机器人状态的虚函数，需要子类实现

        Returns:
            np.ndarray: 机器人状态向量
        """
        pass

    @abstractmethod
    def _get_block_position(self) -> np.ndarray:
        """
        获取方块位置的虚函数，需要子类实现

        Returns:
            np.ndarray: 方块位置 [x, y, z]
        """
        pass

    @abstractmethod
    def _reset_robot_to_home(self) -> None:
        """
        将机器人重置到初始位置的虚函数，需要子类实现
        """
        pass

    def reset(self, seed=None, **kwargs) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """重置环境"""
        if seed is not None:
            self.np_random = np.random.RandomState(seed)

        # 重置机器人到初始位置
        self._reset_robot_to_home()

        # 如果启用随机方块位置，设置新的方块位置
        if self.random_block_position:
            block_xy = self.np_random.uniform(*_SAMPLING_BOUNDS)
            # 这里需要调用设置方块位置的具体实现
            # self._set_block_position(block_xy[0], block_xy[1], self._block_z)
        else:
            block_xy = np.array([0.5, 0.0])
            # self._set_block_position(block_xy[0], block_xy[1], self._block_z)

        # 缓存初始方块高度
        self._current_block_position = self._get_block_position()
        self._z_init = self._current_block_position[2]
        self._z_success = self._z_init + 0.1

        # 获取初始观察
        obs = self._compute_observation()
        return obs, {}

    def step(self, action: np.ndarray) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        """执行一步"""
        # 发送控制指令到真实机器人
        self._send_control_command(action)

        # 等待控制周期
        import time
        time.sleep(self.control_dt)

        # 计算观察、奖励和终止条件
        obs = self._compute_observation()
        rew = self._compute_reward()
        success = self._is_success()

        if self.reward_type == "sparse":
            success = rew == 1.0

        # 检查方块是否超出边界
        block_pos = self._get_block_position()
        exceeded_bounds = np.any(block_pos[:2] < (_SAMPLING_BOUNDS[0] - 0.05)) or np.any(
            block_pos[:2] > (_SAMPLING_BOUNDS[1] + 0.05)
        )

        terminated = bool(success or exceeded_bounds)

        return obs, rew, terminated, False, {"succeed": success}

    def _compute_observation(self) -> Dict[str, np.ndarray]:
        """计算当前观察"""
        observation = {}

        # 获取机器人状态
        robot_state = self._get_robot_state().astype(np.float32)

        if self.image_obs:
            # 图像观察
            front_view, wrist_view = self._get_camera_images()
            observation = {
                "pixels": {"front": front_view, "wrist": wrist_view},
                "agent_pos": robot_state,
            }
        else:
            # 仅状态观察
            block_pos = self._get_block_position().astype(np.float32)
            observation = {
                "agent_pos": robot_state,
                "environment_state": block_pos,
            }

        return observation

    def _compute_reward(self) -> float:
        """计算奖励"""
        block_pos = self._get_block_position()

        if self.reward_type == "dense":
            # 需要获取TCP位置来计算距离奖励
            # tcp_pos = self._get_tcp_position()  # 需要在子类中实现
            # dist = np.linalg.norm(block_pos - tcp_pos)
            # r_close = np.exp(-20 * dist)
            # r_lift = (block_pos[2] - self._z_init) / (self._z_success - self._z_init)
            # r_lift = np.clip(r_lift, 0.0, 1.0)
            # return 0.3 * r_close + 0.7 * r_lift
            return 0.0  # 临时返回值
        else:
            lift = block_pos[2] - self._z_init
            return float(lift > 0.1)

    def _is_success(self) -> bool:
        """检查任务是否成功完成"""
        block_pos = self._get_block_position()
        # 需要获取TCP位置来计算距离
        # tcp_pos = self._get_tcp_position()  # 需要在子类中实现
        # dist = np.linalg.norm(block_pos - tcp_pos)
        lift = block_pos[2] - self._z_init
        # return dist < 0.05 and lift > 0.1
        return lift > 0.1  # 临时简化版本

    def render(self):
        """渲染环境"""
        if self.render_mode == "rgb_array":
            if self.image_obs:
                front_view, wrist_view = self._get_camera_images()
                return {"front": front_view, "wrist": wrist_view}
            else:
                # 如果没有图像观察，返回空字典或状态信息
                return {}
        elif self.render_mode == "human":
            # 在人类模式下，可以显示图像或状态信息
            pass

    def close(self) -> None:
        """关闭环境，释放资源"""
        # 子类可以重写此方法来清理资源
        pass


class RealPandaPickCubeGymEnv(RealRobotGymEnv):
    """真实 Panda 机器人抓取方块的环境"""

    def __init__(
        self,
        seed: int = 0,
        control_dt: float = 0.1,
        render_mode: Literal["rgb_array", "human"] = "rgb_array",
        image_obs: bool = False,
        reward_type: str = "sparse",
        random_block_position: bool = False,
        image_height: int = 128,
        image_width: int = 128,
    ):
        super().__init__(
            seed=seed,
            control_dt=control_dt,
            render_mode=render_mode,
            image_obs=image_obs,
            reward_type=reward_type,
            random_block_position=random_block_position,
            image_height=image_height,
            image_width=image_width,
        )

    def _get_camera_images(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        获取相机图像 - 需要根据实际硬件实现

        Returns:
            Tuple[np.ndarray, np.ndarray]: (front_view, wrist_view)
        """
        # TODO: 实现从真实相机获取图像的代码
        # 这里返回占位符图像
        front_view = np.zeros((self.image_height, self.image_width, 3), dtype=np.uint8)
        wrist_view = np.zeros((self.image_height, self.image_width, 3), dtype=np.uint8)
        return front_view, wrist_view

    def _send_control_command(self, action: np.ndarray) -> None:
        """
        发送控制指令到真实机器人 - 需要根据实际机器人接口实现

        Args:
            action: 控制动作 [x, y, z, rx, ry, rz, grasp_command]
        """
        # TODO: 实现发送控制指令到真实机器人的代码
        # 这里可以调用机器人控制库，如：
        # - panda_python (Franka Emika)
        # - ur_rtde (Universal Robots)
        # - 或其他机器人控制库
        pass

    def get_gripper_pose(self):
        return self._get_robot_state()


    def _get_robot_state(self) -> np.ndarray:
        """
        获取机器人状态 - 需要根据实际硬件实现

        Returns:
            np.ndarray: 机器人状态向量 [qpos, qvel, gripper_pose, tcp_pos]
        """
        # TODO: 实现从真实机器人获取状态的代码
        # 这里返回占位符状态
        qpos = np.zeros(7)  # 关节位置
        qvel = np.zeros(7)  # 关节速度
        gripper_pose = np.zeros(1)  # 夹爪位置
        tcp_pos = np.zeros(3)  # TCP位置

        return np.concatenate([qpos, qvel, gripper_pose, tcp_pos])

    def _get_block_position(self) -> np.ndarray:
        """
        获取方块位置 - 需要根据实际感知系统实现

        Returns:
            np.ndarray: 方块位置 [x, y, z]
        """
        # TODO: 实现获取方块位置的代码
        # 这里可以：
        # - 使用计算机视觉检测方块
        # - 使用外部跟踪系统
        # - 使用其他感知方法
        return np.array([0.5, 0.0, self._block_z])

    def _reset_robot_to_home(self) -> None:
        """
        将机器人重置到初始位置 - 需要根据实际机器人实现
        """
        # TODO: 实现机器人重置到初始位置的代码
        pass

    def _get_tcp_position(self) -> np.ndarray:
        """
        获取TCP位置 - 用于计算奖励和成功条件

        Returns:
            np.ndarray: TCP位置 [x, y, z]
        """
        # TODO: 实现获取TCP位置的代码
        return np.zeros(3)


# 使用示例
if __name__ == "__main__":
    # 创建环境实例
    if ROS2_AVAILABLE:
        print("Creating RealCR5PickCubeGymEnv...")
        env = RealCR5PickCubeGymEnv(
            render_mode="rgb_array",
            image_obs=False,  # 不使用图像观察
            reward_type="sparse"
        )

        # 重置环境
        obs, info = env.reset()
        print("初始观察:", obs.keys())
        print("观察空间:", env.observation_space)
        print("动作空间:", env.action_space)

        # 运行几步
        for step in range(5):
            action = env.action_space.sample()  # 随机动作
            obs, reward, terminated, truncated, info = env.step(action)
            print(f"步骤 {step}: 奖励={reward}, 成功={info.get('succeed', False)}")

            if terminated:
                break

        env.close()
    else:
        print("ROS2 modules not available. Please install lerobot_robot_ros2.")
        print("For a complete demo, run: python examples/test_cr5_movement.py")


class RealCR5PickCubeGymEnv(RealRobotGymEnv):
    """基于 ROS2 Robot Config 的真实 CR5 机器人抓取环境"""

    def __init__(
        self,
        seed: int = 0,
        control_dt: float = 0.1,
        render_mode: Literal["rgb_array", "human"] = "rgb_array",
        image_obs: bool = False,
        reward_type: str = "sparse",
        random_block_position: bool = False,
        image_height: int = 128,
        image_width: int = 128,
        ros2_config: Optional[ROS2RobotConfig] = None,
    ):
        # 检查 ROS2 模块是否可用
        if not ROS2_AVAILABLE:
            raise ImportError("ROS2 Robot modules not available. Please install lerobot_robot_ros2.")
        
        # 设置默认 ROS2 配置
        if ros2_config is None:
            ros2_config = self._create_default_cr5_config()
        
        self.robot_config = ros2_config
        self.robot = None  # 将在 super().__init__ 后初始化
        self.wrist_camera = Camera_wrist()
        self.ex_camera = Camera_ex()
        # 调用父类初始化
        super().__init__(
            seed=seed,
            control_dt=control_dt,
            render_mode=render_mode,
            image_obs=image_obs,
            reward_type=reward_type,
            random_block_position=random_block_position,
            image_height=image_height,
            image_width=image_width,
        )
        
        # 初始化机器人连接
        self._initialize_robot()

    @staticmethod
    def _create_default_cr5_config() -> ROS2RobotConfig:
        """创建默认的 CR5 机器人配置"""
        return ROS2RobotConfig(
            id="cr5_robot",
            ros2_interface=ROS2RobotInterfaceConfig(
                joint_states_topic="/joint_states",
                end_effector_pose_topic="/left_current_pose",
                end_effector_target_topic="/left_target",
                control_type=ControlType.CARTESIAN_POSE,
                joint_names=["left_joint1", "left_joint2", "left_joint3", "left_joint4", "left_joint5", "left_joint6"],
                gripper_enabled=True,
                gripper_joint_name="left_gripper_joint",
                gripper_command_topic="left_gripper_joint/position_command",  # 夹爪控制话题
                gripper_min_position=0.105,  # 关闭位置
                gripper_max_position=1.0,   # 打开位置
                max_linear_velocity=0.1,
                max_angular_velocity=0.5,
                joint_state_timeout=1.0,
                end_effector_pose_timeout=1.0,
            ),
            cameras={},  # 不使用相机
        )

    def _initialize_robot(self):
        """初始化机器人连接"""
        try:
            self.robot = ROS2Robot(self.robot_config)
            self.robot.connect()
            logging.info("CR5 robot connected successfully")
        except Exception as e:
            logging.error(f"Failed to connect to CR5 robot: {e}")
            raise

    def _setup_observation_space(self):
        """设置观察空间 - 适配 CR5 机器人"""
        # CR5 机器人状态维度：关节位置(6) + 关节速度(6) + TCP位置(3) + 夹爪位置(1)  = 16
        agent_dim = 16
        agent_box = spaces.Box(-np.inf, np.inf, (agent_dim,), dtype=np.float32)
        env_box = spaces.Box(-np.inf, np.inf, (3,), dtype=np.float32)

        if self.image_obs:
            # 图像观察（暂时使用占位符）
            self.observation_space = spaces.Dict(
                {
                    "pixels": spaces.Dict(
                        {
                            "front": spaces.Box(
                                0,
                                255,
                                (self.image_height, self.image_width, 3),
                                dtype=np.uint8,
                            ),
                            "wrist": spaces.Box(
                                0,
                                255,
                                (self.image_height, self.image_width, 3),
                                dtype=np.uint8,
                            ),
                        }
                    ),
                    "agent_pos": agent_box,
                }
            )
        else:
            self.observation_space = spaces.Dict(
                {
                    "agent_pos": agent_box,
                    "environment_state": env_box,
                }
            )

    def get_gripper_pose(self):
        obs = self.robot.get_observation()

        gripper_pos = obs[f"{self.robot.config.ros2_interface.gripper_joint_name}.pos"]
        
        return gripper_pos

    def _get_robot_state(self) -> np.ndarray:
        """从 ROS2 Robot 获取机器人状态"""
        if self.robot is None:
            raise RuntimeError("Robot not initialized")
        
        try:
            obs = self.robot.get_observation()
            
            # 提取关节状态
            joint_positions = []
            joint_velocities = []
            
            for joint_name in self.robot.config.ros2_interface.joint_names:
                joint_positions.append(obs[f"{joint_name}.pos"])
                joint_velocities.append(obs[f"{joint_name}.vel"])
            
            # 提取夹爪状态
            gripper_pos = obs[f"{self.robot.config.ros2_interface.gripper_joint_name}.pos"]
            
            # 提取 TCP 位置
            tcp_pos = np.array([
                obs["end_effector.position.x"],
                obs["end_effector.position.y"],
                obs["end_effector.position.z"]
            ])
            
            # 组合状态向量：关节位置(6) + 关节速度(6) + 夹爪位置(1) + TCP位置(3) = 16
            return np.concatenate([joint_positions, joint_velocities, tcp_pos, [gripper_pos]])
            
        except Exception as e:
            logging.error(f"Failed to get robot state: {e}")
            # 返回零状态作为后备
            return np.zeros(16, dtype=np.float32)

    def _send_control_command(self, action: np.ndarray) -> None:
        """通过 ROS2 Robot 发送控制指令 - 增量控制模式"""
        if self.robot is None:
            raise RuntimeError("Robot not initialized")
        
        try:
            # 获取当前末端执行器位置
            current_obs = self.robot.get_observation()
            current_pos = np.array([
                current_obs["end_effector.position.x"],
                current_obs["end_effector.position.y"],
                current_obs["end_effector.position.z"]
            ])
            current_ori = np.array([
                current_obs["end_effector.orientation.x"],
                current_obs["end_effector.orientation.y"],
                current_obs["end_effector.orientation.z"],
                current_obs["end_effector.orientation.w"]
            ])
            
            # 定义动作缩放因子
            position_scale = 0.5  # 5cm 最大位置增量
            orientation_scale = 0.5  # 0.1弧度最大姿态增量
            
            # 计算目标位置（当前位置 + 增量）
            target_pos = current_pos + action[:3] * position_scale
            
            # 计算目标姿态（当前位置 + 增量）
            # 注意：这里简化处理，实际应该使用四元数运算
            target_ori = current_ori + np.concatenate([action[3:6] * orientation_scale, [0.0]])
            
            # 处理夹爪控制命令
            # action[6] 是夹爪命令：-1 表示关闭，1 表示打开
            gripper_command = action[6]
            # 将 [-1, 1] 映射到 [0, 1] 范围
            gripper_position = (gripper_command + 1.0) / 2.0
            
            # 动作格式：[x, y, z, rx, ry, rz, grasp_command]
            target_action = {
                "end_effector.position.x": float(target_pos[0]),
                "end_effector.position.y": float(target_pos[1]),
                "end_effector.position.z": float(target_pos[2]),
                "end_effector.orientation.x": float(target_ori[0]),
                "end_effector.orientation.y": float(target_ori[1]),
                "end_effector.orientation.z": float(target_ori[2]),
                "end_effector.orientation.w": float(target_ori[3]),
                "gripper.position": float(gripper_position),  # 添加夹爪控制
            }
            
            self.robot.send_action(target_action)
            
        except Exception as e:
            logging.error(f"Failed to send control command: {e}")
            raise

    def _get_camera_images(self) -> Tuple[np.ndarray, np.ndarray]:
        """获取相机图像并调整到指定尺寸"""
        import cv2
        
        # 获取原始相机帧
        front_view, _ = self.ex_camera.get_frame()
        wrist_view, _ = self.wrist_camera.get_frame()
        
        # 将图像调整到128x128尺寸
        if front_view is not None and front_view.shape[:2] != (self.image_height, self.image_width):
            front_view = cv2.resize(front_view, (self.image_width, self.image_height))
        
        if wrist_view is not None and wrist_view.shape[:2] != (self.image_height, self.image_width):
            wrist_view = cv2.resize(wrist_view, (self.image_width, self.image_height))
        
        # 如果相机返回None，创建占位符图像
        if front_view is None:
            front_view = np.zeros((self.image_height, self.image_width, 3), dtype=np.uint8)
        if wrist_view is None:
            wrist_view = np.zeros((self.image_height, self.image_width, 3), dtype=np.uint8)
        
        return front_view, wrist_view

    def _get_block_position(self) -> np.ndarray:
        """获取方块位置 - 暂时返回固定位置"""
        # TODO: 实现计算机视觉检测方块位置
        # 暂时返回固定位置
        return np.array([0.5, 0.0, self._block_z])

    def _reset_robot_to_home(self) -> None:
        """重置机器人到初始位置"""
        if self.robot is None:
            raise RuntimeError("Robot not initialized")
        
        try:
            # CR5 机器人的初始位置 - 基于实际参数设定
            home_action = {
                "end_effector.position.x": _CR5_HOME_POSE["position"]["x"],
                "end_effector.position.y": _CR5_HOME_POSE["position"]["y"],
                "end_effector.position.z": _CR5_HOME_POSE["position"]["z"],
                "end_effector.orientation.x": _CR5_HOME_POSE["orientation"]["x"],
                "end_effector.orientation.y": _CR5_HOME_POSE["orientation"]["y"],
                "end_effector.orientation.z": _CR5_HOME_POSE["orientation"]["z"],
                "end_effector.orientation.w": _CR5_HOME_POSE["orientation"]["w"],
            }
            
            self.robot.send_action(home_action)
            time.sleep(2.0)  # 等待机器人到达位置
            
        except Exception as e:
            logging.error(f"Failed to reset robot to home: {e}")
            raise

    def _get_tcp_position(self) -> np.ndarray:
        """获取TCP位置 - 用于计算奖励和成功条件"""
        if self.robot is None:
            raise RuntimeError("Robot not initialized")
        
        try:
            obs = self.robot.get_observation()
            return np.array([
                obs["end_effector.position.x"],
                obs["end_effector.position.y"],
                obs["end_effector.position.z"]
            ])
        except Exception as e:
            logging.error(f"Failed to get TCP position: {e}")
            return np.zeros(3)

    def close(self) -> None:
        """关闭环境，释放资源"""
        if self.robot is not None:
            try:
                self.robot.disconnect()
                logging.info("CR5 robot disconnected")
            except Exception as e:
                logging.error(f"Error disconnecting robot: {e}")
            finally:
                self.robot = None
        super().close()