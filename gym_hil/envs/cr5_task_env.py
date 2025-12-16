from typing import Any, Dict, Tuple
import cv2
import numpy as np
import logging
import time
from gym_hil.envs.real_cr5_env import RealCR5PickCubeGymEnv



# TARGET_POS =   {
#     'x': 0.669927695343881,
#     'y': 0.3580578291818187,
#     'z': 0.011135972386946473,
#     'r_x': -0.6883915623825985,
#     'r_y': 0.7248515465813707,
#     'r_z': 0.02585817468371632,
#     'r_w': 0.006216676046442958,
# }
TARGET_POS =   {
        "x": -0.4287862558930951,
        "y": 0.211441424559421934,
        "z": 0.06131800404118503,
        "r_x": 0.0,
        "r_y": 1.0,
        "r_z": 0.0,
        "r_w": 0.0
}
READY_POS = {
    "x": 0.6117862558930951,
    "y": -0.008441424559421934,
    "z": 0.12131800404118503,
    "y_neg": -0.002,
    "z_neg": 0.113
}

class CR5TaskGymEnv(RealCR5PickCubeGymEnv):
    """CR5 任务环境 - 继承自 RealCR5PickCubeGymEnv 并扩展功能"""

    def __init__(self, *args, target_pos: np.ndarray = None, target_orientation: Dict = None,
                 crop: Tuple[int, int, int, int] = [350,80,550,390],
                 out_image_size: int = 128,
                 image_obs: bool = True,
                 **kwargs):
        """初始化任务环境
        
        Args:
            target_pos: 目标位置 [x, y, z]，默认为 TARGET_POS 中的位置
            target_orientation: 目标旋转字典 {'x', 'y', 'z', 'w'}，默认为 TARGET_POS 中的旋转
            crop: 图像裁剪区域 (x, y, w, h)，先裁剪后再缩放到 out_image_size×out_image_size
            out_image_size: 输出图像边长，默认 128
            image_obs: 是否启用图像观测，默认 True
            *args, **kwargs: 传递给父类的参数
        """
        # 调用父类初始化，固定输出分辨率为 out_image_size × out_image_size
        super().__init__(
            *args,
            image_obs=image_obs,
            image_height=out_image_size,
            image_width=out_image_size,
            **kwargs
        )
        
        # 裁剪配置
        self.crop = crop  # (x, y, w, h)
        self.out_image_size = int(out_image_size)

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

        logging.info(f"Task environment initialized with target_pos: {self.target_pos}, target_orientation: {self.target_orientation}, crop: {self.crop}, out_image_size: {self.out_image_size}")
    
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

    def _get_camera_images(self) -> Tuple[np.ndarray, np.ndarray]:
        """获取相机图像，先按 crop 裁剪，再 resize 到 out_image_size×out_image_size"""
        # 直接使用父类里初始化的相机句柄


        front_view, _ = self.ex_camera.get_frame()
        wrist_view, _ = self.wrist_camera.get_frame()



        # 执行裁剪 (x, y, w, h)
        if self.crop is not None:
            x, y, cw, ch = self.crop
            def safe_crop(img):
                H, W = img.shape[:2]
                x0 = max(0, int(x))
                y0 = max(0, int(y))
                x1 = min(W, x0 + int(cw))
                y1 = min(H, y0 + int(ch))
                if x1 <= x0 or y1 <= y0:
                    return img  # 无效裁剪则返回原图
                return img[y0:y1, x0:x1]

            front_view = safe_crop(front_view)
            # wrist_view = safe_crop(wrist_view)
            x, y, cw, ch = [100,200,300,200]

            wrist_view = safe_crop(wrist_view)

        # 裁剪后再缩放到 out_image_size × out_image_size
        out_sz = (self.out_image_size, self.out_image_size)
        if front_view.shape[1] != self.out_image_size or front_view.shape[0] != self.out_image_size:
            front_view = cv2.resize(front_view, out_sz)
        if wrist_view.shape[1] != self.out_image_size or wrist_view.shape[0] != self.out_image_size:
            wrist_view = cv2.resize(wrist_view, out_sz)


        cv2.imshow("Front View", front_view)
        cv2.waitKey(1)
        cv2.imshow("Wrist View", wrist_view)
        cv2.waitKey(1)
        


        # cv2.waitKey(1)
        return front_view, wrist_view

    def _get_tcp_orientation(self) -> Dict[str, float]:
        """获取TCP姿态（四元数）
        
        Returns:
            Dict[str, float]: 姿态字典 {'x', 'y', 'z', 'w'}
        """
        if self.robot is None:
            raise RuntimeError("Robot not initialized")
        
        try:
            obs = self.robot.get_observation()
            return {
                'x': float(obs["end_effector.orientation.x"]),
                'y': float(obs["end_effector.orientation.y"]),
                'z': float(obs["end_effector.orientation.z"]),
                'w': float(obs["end_effector.orientation.w"])
            }
        except Exception as e:
            logging.error(f"Failed to get TCP orientation: {e}")
            # 返回默认姿态作为后备
            return {'x': 0.0, 'y': 0.0, 'z': 0.0, 'w': 1.0}

    def reset(self, seed=None, **kwargs) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """重置环境 - 执行预定义动作序列后调用父类reset"""
        
        if self.robot is None:
            raise RuntimeError("Robot not initialized")
        
        logging.info("Starting custom reset sequence...")
        
        # # 1. 打开夹爪#
        logging.info("Opening gripper...")
        current_pos = self._get_tcp_position()
        current_ori = self._get_tcp_orientation()  # 获取当前姿态
        self._move_to_position(current_pos, orientation=current_ori, gripper_position=1.0, wait_time=3.0)
        
        # # 2. 移动到 target pos 正上方 5cm
        logging.info(f"Moving to 5cm above target position {self.target_pos}...")
        above_target = self.target_pos.copy()
        above_target[2] += 0.05  # 向上5cm
        self._move_to_position(above_target, gripper_position=1.0, wait_time=3.0)
        
        # # 3. 下降到 target pos
        logging.info(f"Moving down to target position {self.target_pos}...")
        self._move_to_position(self.target_pos, gripper_position=1.0, wait_time=3.0)
        
        # # 4. 关闭夹爪
        logging.info("Closing gripper...")
        self._move_to_position(self.target_pos, gripper_position=0.0, wait_time=3.0)
        
        # 5. 后退10cm（沿Z轴正方向）
        logging.info("Moving back 10cm...")
        back_pos = self.target_pos.copy()
        back_pos[0] += 0.030  # 沿x轴后退10cm
        self._move_to_position(back_pos, orientation=current_ori, gripper_position=0.0, wait_time=3.0)
        
        # # 6. 打开夹爪
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
        # action[-1] = -1.0  # 打开夹爪
        action[3:6] = 0.0  # 旋转
        obs, reward, terminated, truncated, info = super().step(action)
        
        # # 2. 添加基于 READY_POS 的势场奖励
        # current_pos = self._get_tcp_position()
        # current_x = current_pos[0]
        # current_y = current_pos[1]
        # current_z = current_pos[2]
        # ready_x = READY_POS['x']
        # ready_y = READY_POS['y']
        # ready_z = READY_POS['z']
        # ready_z_neg = READY_POS['z_neg']
        # ready_y_neg = READY_POS['y_neg']
        # # X 坐标势场奖励计算
        # if current_x >= ready_x:
        #     # 进入或超过 READY_pos.X 后无奖励
        #     potential_reward_x = 0.0
        # else:
        #     # 小于 READY_pos.X 时，计算距离并给予负奖励
        #     distance_x = ready_x - current_x  # 距离 READY_pos.X 的距离
        #     potential_reward_x = -distance_x * 0.10
        
        # # Y 坐标势场奖励计算
        # if current_y > ready_y:
        #     # 大于 READY_pos.Y 后无奖励
        #     potential_reward_y = 0.0
        # else:
        #     # 小于等于 READY_pos.Y 时，计算距离并给予负奖励
        #     distance_y = ready_y - current_y  # 距离 READY_pos.Y 的距离
        #     potential_reward_y = -distance_y * 0.10

        # if current_y < ready_y_neg:
        #     # 大于 READY_pos.Y 后无奖励
        #     potential_reward_y_neg = 0.0
        # else:
        #     # 小于等于 READY_pos.Y 时，计算距离并给予负奖励
        #     distance_y_neg =  current_y - ready_y_neg  # 距离 READY_pos.Y 的距离
        #     potential_reward_y_neg = -distance_y_neg * 0.10
        
        # # Z 坐标势场奖励计算
        # if current_z < ready_z:
        #     # 小于 READY_pos.Z 后无奖励
        #     potential_reward_z = 0.0
        # else:
        #     # 大于等于 READY_pos.Z 时，计算距离并给予负奖励
        #     distance_z = current_z - ready_z  # 距离 READY_pos.Z 的距离
        #     potential_reward_z = -distance_z * 0.10

        # if current_z >= ready_z_neg:
        #     # 小于 READY_pos.Z 后无奖励
        #     potential_reward_z_neg = 0.0
        # else:
        #     # 大于等于 READY_pos.Z 时，计算距离并给予负奖励
        #     distance_z_neg = ready_z_neg - current_z  # 距离 READY_pos.Z 的距离
        #     potential_reward_z_neg = -distance_z_neg * 0.10
        
        # # 总势场奖励
        # potential_reward = potential_reward_x + potential_reward_y + potential_reward_z + potential_reward_z_neg + potential_reward_y_neg
        
        # # 将势场奖励加到总奖励中
        # reward += (potential_reward*100)
        
        # 3. 添加自定义步骤逻辑
        self.task_step_count += 1
        
        # 可选：在 info 中记录势场奖励信息
        # info['potential_reward'] = potential_reward
        # info['potential_reward_x'] = potential_reward_x
        # info['potential_reward_y'] = potential_reward_y
        # info['potential_reward_z'] = potential_reward_z
        # info['current_x'] = current_x
        # info['current_y'] = current_y
        # info['current_z'] = current_z
        # info['ready_x'] = ready_x
        # info['ready_y'] = ready_y
        # info['ready_z'] = ready_z
        # info['distance_to_ready_x'] = ready_x - current_x if current_x < ready_x else 0.0
        # info['distance_to_ready_y'] = ready_y - current_y if current_y <= ready_y else 0.0
        # info['distance_to_ready_z'] = current_z - ready_z if current_z >= ready_z else 0.0
        
        return obs, reward, terminated, truncated, info
    
    def _compute_custom_metric(self) -> float:
        """计算自定义指标"""
        # 在这里实现你的自定义指标计算
        return 0.0



def main():
    """测试函数：创建环境并保存 observation 图片"""
    import os
    from datetime import datetime
    
    # 创建输出目录
    output_dir = "test_observation_images"
    os.makedirs(output_dir, exist_ok=True)
    
    print("正在创建环境...")
    # 创建环境实例
    env = CR5TaskGymEnv(
        image_obs=True,
        out_image_size=128,
        # crop=[500, 200, 750, 400],
    )
    
    print("正在重置环境...")
    # 重置环境
    obs, info = env.reset()
    
    print(f"Observation keys: {list(obs.keys())}")
    if "pixels" in obs:
        print(f"Pixels keys: {list(obs['pixels'].keys())}")
        print(f"Front image shape: {obs['pixels']['front'].shape}")
        print(f"Wrist image shape: {obs['pixels']['wrist'].shape}")
    
    # 保存初始 observation 的图片
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if "pixels" in obs:
        front_img = obs["pixels"]["front"]
        wrist_img = obs["pixels"]["wrist"]
        
        # 保存 front 视图
        front_path = os.path.join(output_dir, f"front_reset_{timestamp}.png")
        cv2.imwrite(front_path, cv2.cvtColor(front_img, cv2.COLOR_RGB2BGR))
        print(f"已保存 front 图片: {front_path}")
        
        # 保存 wrist 视图
        wrist_path = os.path.join(output_dir, f"wrist_reset_{timestamp}.png")
        cv2.imwrite(wrist_path, cv2.cvtColor(wrist_img, cv2.COLOR_RGB2BGR))
        print(f"已保存 wrist 图片: {wrist_path}")
    
    # 执行几步测试
    print("\n执行几步测试...")
    for step in range(5):
        # 创建随机动作（或者零动作）
        action = np.zeros(7, dtype=np.float32)
        action[-1] = -1.0  # 打开夹爪
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        if "pixels" in obs:
            front_img = obs["pixels"]["front"]
            wrist_img = obs["pixels"]["wrist"]
            
            # 保存每步的图片
            front_path = os.path.join(output_dir, f"front_step_{step:03d}_{timestamp}.png")
            wrist_path = os.path.join(output_dir, f"wrist_step_{step:03d}_{timestamp}.png")
            
            cv2.imwrite(front_path, cv2.cvtColor(front_img, cv2.COLOR_RGB2BGR))
            cv2.imwrite(wrist_path, cv2.cvtColor(wrist_img, cv2.COLOR_RGB2BGR))
            
            print(f"Step {step}: reward={reward:.4f}, saved images")
        
        if terminated or truncated:
            print(f"Episode ended at step {step}")
            break
    
    print(f"\n所有图片已保存到: {output_dir}")
    env.close()


if __name__ == "__main__":
    main()
    
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
