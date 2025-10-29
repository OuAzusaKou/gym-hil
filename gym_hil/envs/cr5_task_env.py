from typing import Any, Dict, Tuple
import numpy as np
import logging
import time
from gym_hil.envs.real_cr5_env import RealCR5PickCubeGymEnv



TARGET_POS =   {
    'x': 0.669927695343881,
    'y': 0.3580578291818187,
    'z': 0.011135972386946473,
    'r_x': -0.6883915623825985,
    'r_y': 0.7248515465813707,
    'r_z': 0.02585817468371632,
    'r_w': 0.006216676046442958,
}

class CR5TaskGymEnv(RealCR5PickCubeGymEnv):
    """CR5 任务环境 - 继承自 RealCR5PickCubeGymEnv 并扩展功能"""

    def __init__(self, *args, target_pos: np.ndarray = None, target_orientation: Dict = None, **kwargs):
        """初始化任务环境
        
        Args:
            target_pos: 目标位置 [x, y, z]，默认为 TARGET_POS 中的位置
            target_orientation: 目标旋转字典 {'x', 'y', 'z', 'w'}，默认为 TARGET_POS 中的旋转
            *args, **kwargs: 传递给父类的参数
        """
        # 调用父类初始化
        super().__init__(*args, **kwargs)
        
        # 设置预定义的目标位置和旋转
        if target_pos is None:
            target_pos = np.array([
                TARGET_POS['x'],
                TARGET_POS['y'],
                TARGET_POS['z']
            ])
        self.target_pos = np.array(target_pos, dtype=np.float32)
        
        # 设置目标旋转
        if target_orientation is None:
            self.target_orientation = {
                'x': TARGET_POS['r_x'],
                'y': TARGET_POS['r_y'],
                'z': TARGET_POS['r_z'],
                'w': TARGET_POS['r_w']
            }
        else:
            self.target_orientation = target_orientation
        
        # 添加任务相关的自定义变量
        self.task_step_count = 0
        self.custom_info = {}
        logging.info(f"Task environment initialized with target_pos: {self.target_pos}, target_orientation: {self.target_orientation}")

    def _move_to_position(self, position: np.ndarray, orientation: Dict = None, gripper_position: float = None, wait_time: float = 2.0):
        """移动机器人到指定位置
        
        Args:
            position: 目标位置 [x, y, z]
            orientation: 目标旋转字典 {'x', 'y', 'z', 'w'}，None 表示使用目标旋转
            gripper_position: 夹爪位置 (0.0-1.0)，None 表示保持当前状态
            wait_time: 等待时间（秒）
        """
        if self.robot is None:
            raise RuntimeError("Robot not initialized")
        
        # 确定使用的旋转
        if orientation is None:
            # 如果没有提供旋转，使用目标旋转
            target_ori = self.target_orientation
        else:
            target_ori = orientation
        
        # 构建动作字典
        action = {
            "end_effector.position.x": float(position[0]),
            "end_effector.position.y": float(position[1]),
            "end_effector.position.z": float(position[2]),
            "end_effector.orientation.x": target_ori['x'],
            "end_effector.orientation.y": target_ori['y'],
            "end_effector.orientation.z": target_ori['z'],
            "end_effector.orientation.w": target_ori['w'],
        }
        
        # 如果需要控制夹爪
        if gripper_position is not None:
            action["gripper.position"] = float(np.clip(gripper_position, 0.0, 1.0))
        
        # 发送动作
        self.robot.send_action(action)
        time.sleep(wait_time)  # 等待机器人到达位置

    def reset(self, seed=None, **kwargs) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """重置环境 - 执行预定义动作序列后调用父类reset"""
        
        if self.robot is None:
            raise RuntimeError("Robot not initialized")
        
        logging.info("Starting custom reset sequence...")
        
        # 1. 打开夹爪
        logging.info("Opening gripper...")
        current_pos = self._get_tcp_position()
        self._move_to_position(current_pos, gripper_position=1.0, wait_time=3.0)
        
        # 2. 移动到 target pos 正上方 5cm
        logging.info(f"Moving to 5cm above target position {self.target_pos}...")
        above_target = self.target_pos.copy()
        above_target[2] += 0.05  # 向上5cm
        self._move_to_position(above_target, gripper_position=1.0, wait_time=3.0)
        
        # 3. 下降到 target pos
        logging.info(f"Moving down to target position {self.target_pos}...")
        self._move_to_position(self.target_pos, gripper_position=1.0, wait_time=3.0)
        
        # 4. 关闭夹爪
        logging.info("Closing gripper...")
        self._move_to_position(self.target_pos, gripper_position=0.0, wait_time=3.0)
        
        # 5. 后退10cm（沿Z轴正方向）
        logging.info("Moving back 10cm...")
        back_pos = self.target_pos.copy()
        back_pos[0] -= 0.05  # 沿Z轴后退10cm
        self._move_to_position(back_pos, gripper_position=0.0, wait_time=2.0)
        
        # 6. 打开夹爪
        logging.info("Opening gripper...")
        self._move_to_position(back_pos, gripper_position=1.0, wait_time=1.0)
        
        logging.info("Custom reset sequence completed, calling parent reset...")
        
        # 7. 最后调用父类的reset函数
        obs, info = super().reset(seed=seed, **kwargs)
        
        # 添加自定义信息
        self.task_step_count = 0
        self.custom_info = {
            "task_reset_completed": True,
            "initial_position": self._get_tcp_position().copy(),
            "target_position": self.target_pos.copy(),
        }
        
        # 合并自定义信息到 info
        info.update(self.custom_info)
        
        logging.info("Task environment reset completed")
        return obs, info

    def step(self, action: np.ndarray) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        """执行一步 - 继承父类功能并添加自定义逻辑"""
        # 1. 先调用父类的 step 方法
        obs, reward, terminated, truncated, info = super().step(action)
        
        # 2. 添加自定义步骤逻辑
        self.task_step_count += 1
        
        return obs, reward, terminated, truncated, info
    
    def _compute_custom_metric(self) -> float:
        """计算自定义指标"""
        # 在这里实现你的自定义指标计算
        return 0.0
    
    # 可选：如果需要，可以重写其他方法来扩展功能
    # def _compute_reward(self) -> float:
    #     """可以重写奖励计算"""
    #     base_reward = super()._compute_reward()
    #     # 添加自定义奖励
    #     custom_reward = 0.0
    #     return base_reward + custom_reward
    
    # def _is_success(self) -> bool:
    #     """可以重写成功判断"""
    #     base_success = super()._is_success()
    #     # 添加自定义成功条件
    #     return base_success
