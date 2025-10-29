# Gym-HIL 手柄输入到 Gym Action 转换流程分析

## 整体架构

手柄输入转换为 gym action 的流程分为5层：

```
手柄硬件 → 原始输入读取 → 增量计算 → Action数组构建 → Action缩放 → 机器人执行
```

---

## 第一层：手柄输入读取（GamepadController）

**文件位置**：`gym_hil/wrappers/intervention_utils.py`

### 1.1 初始化阶段

```python
# GamepadController.__init__()
- 初始化步长参数：x_step_size, y_step_size, z_step_size (默认0.01m)
- 设置死区 (deadzone, 默认0.1)
- 加载手柄配置 (controller_config.json)
```

```370:449:gym_hil/wrappers/intervention_utils.py
    def get_deltas(self):
        """Get the current movement deltas from gamepad state."""
        import pygame

        try:
            # Get axis mappings from config
            axes = self.controller_config.get("axes", {})
            axis_inversion = self.controller_config.get("axis_inversion", {})

            # Get axis indices from config (with defaults if not found)
            left_x_axis = axes.get("left_x", 0)
            left_y_axis = axes.get("left_y", 1)
            right_x_axis = axes.get("right_x", 3)  # For yaw rotation
            right_y_axis = axes.get("right_y", 4)  # For Z movement

            # Get axis inversion settings (with defaults if not found)
            invert_left_x = axis_inversion.get("left_x", False)
            invert_left_y = axis_inversion.get("left_y", True)
            invert_right_x = axis_inversion.get("right_x", True)  # For yaw
            invert_right_y = axis_inversion.get("right_y", True)
            
            # D-pad inversion settings (for get_hat values)
            invert_dpad_x = axis_inversion.get("dpad_x", False)
            invert_dpad_y = axis_inversion.get("dpad_y", False)

            # Read joystick axes for translation
            x_input = self.joystick.get_axis(left_x_axis)  # Left/Right -> Y axis
            y_input = self.joystick.get_axis(left_y_axis)  # Up/Down -> X axis
            z_input = self.joystick.get_axis(right_y_axis)  # Z axis
            
            # Apply deadzone to avoid drift
            x_input = 0 if abs(x_input) < self.deadzone else x_input
            y_input = 0 if abs(y_input) < self.deadzone else y_input
            z_input = 0 if abs(z_input) < self.deadzone else z_input

            # Apply inversion if configured
            if invert_left_x:
                x_input = -x_input
            if invert_left_y:
                y_input = -y_input
            if invert_right_y:
                z_input = -z_input

            # Calculate translation deltas
            delta_x = y_input * self.y_step_size  # Forward/backward
            delta_y = x_input * self.x_step_size  # Left/right
            delta_z = z_input * self.z_step_size  # Up/down
            
            # Calculate rotation deltas if enabled
            if self.enable_rotation:
                # Read rotation axes - right stick X controls Yaw (rz)
                yaw_input = self.joystick.get_axis(right_x_axis)
                
                # Read dpad for pitch and roll (dpad uses get_hat, not get_axis)
                # hat[0] is x-axis (-1, 0, 1), hat[1] is y-axis (-1, 0, 1)
                dpad_x, dpad_y = self.joystick.get_hat(0)  # Get the hat value for hat 0 (the dpad)
                
                # Convert dpad to input (already in range -1, 0, 1)
                roll_input = float(dpad_y)  # dpad_y -> roll (rx) - up/down on dpad
                pitch_input = float(dpad_x)  # dpad_x -> pitch (ry) - left/right on dpad
                
                # Apply deadzone for analog stick (dpad doesn't need deadzone as it's digital)
                yaw_input = 0 if abs(yaw_input) < self.deadzone else yaw_input
                
                # Apply inversion
                if invert_right_x:
                    yaw_input = -yaw_input
                if invert_dpad_x:
                    pitch_input = -pitch_input
                if invert_dpad_y:
                    roll_input = -roll_input
                
                # Calculate rotation deltas
                delta_rx = roll_input  # Roll
                delta_ry = pitch_input  # Pitch
                delta_rz = yaw_input  # Yaw rotation
                
                return delta_x, delta_y, delta_z, delta_rx, delta_ry, delta_rz
            else:
                return delta_x, delta_y, delta_z, 0.0, 0.0, 0.0

        except pygame.error:
            print("Error reading gamepad. Is it still connected?")
            if self.enable_rotation:
                return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
            else:
                return 0.0, 0.0, 0.0
```

### 1.2 输入映射关系

**摇杆轴映射**：
- **左摇杆 X 轴** (left_x_axis) → `delta_y` (左右移动)
- **左摇杆 Y 轴** (left_y_axis) → `delta_x` (前后移动，默认反转)
- **右摇杆 Y 轴** (right_y_axis) → `delta_z` (上下移动)
- **右摇杆 X 轴** (right_x_axis) → `delta_rz` (偏航角，如果启用旋转)

**D-pad 映射** (如果启用旋转控制)：
- **D-pad 左右** → `delta_ry` (俯仰角)
- **D-pad 上下** → `delta_rx` (翻滚角)

### 1.3 处理流程

1. **读取原始轴值**：从 pygame joystick 读取 [-1, 1] 范围的轴值
2. **应用死区**：小于 `deadzone` (0.1) 的值设为 0，防止漂移
3. **应用反转**：根据配置翻转轴方向
4. **计算增量**：
   - 位置增量 = 轴值 × step_size（0.01m）
   - 旋转增量 = 轴值（单位：无单位比例值，后续会缩放）

**输出**：`(delta_x, delta_y, delta_z, delta_rx, delta_ry, delta_rz)` 或 `(delta_x, delta_y, delta_z)`

---

## 第二层：Action数组构建（InputsControlWrapper）

**文件位置**：`gym_hil/wrappers/hil_wrappers.py`

```184:231:gym_hil/wrappers/hil_wrappers.py
    def get_gamepad_action(self):
        """
        Get the current action from the gamepad if any input is active.

        Returns:
            Tuple of (is_active, action, terminate_episode, success)
        """
        # Update the controller to get fresh inputs
        self.controller.update()

        # Get movement deltas from the controller (may include rotation)
        deltas = self.controller.get_deltas()
        
        # Handle both 3D and 6D deltas
        if len(deltas) == 6:
            delta_x, delta_y, delta_z, delta_rx, delta_ry, delta_rz = deltas
            # Create 6D action including rotation
            gamepad_action = np.array([delta_x, delta_y, delta_z, delta_rx, delta_ry, delta_rz], dtype=np.float32)
        else:
            delta_x, delta_y, delta_z = deltas[:3]
            # Create 3D action for translation only
            gamepad_action = np.array([delta_x, delta_y, delta_z], dtype=np.float32)

        intervention_is_active = self.controller.should_intervene()

        if self.use_gripper:
            gripper_command = self.controller.gripper_command()
            if gripper_command == "open":
                gamepad_action = np.concatenate([gamepad_action, [2.0]])
            elif gripper_command == "close":
                gamepad_action = np.concatenate([gamepad_action, [0.0]])
            else:
                gamepad_action = np.concatenate([gamepad_action, [1.0]])

        # Check episode ending buttons
        # We'll rely on controller.get_episode_end_status() which returns "success", "failure", or None
        episode_end_status = self.controller.get_episode_end_status()
        terminate_episode = episode_end_status is not None
        success = episode_end_status == "success"
        rerecord_episode = episode_end_status == "rerecord_episode"

        return (
            intervention_is_active,
            gamepad_action,
            terminate_episode,
            success,
            rerecord_episode,
        )
```

### 2.1 转换逻辑

1. **获取增量**：调用 `controller.get_deltas()` 获取增量值
2. **构建 Action 数组**：
   - 如果返回6个值（包含旋转）：`[delta_x, delta_y, delta_z, delta_rx, delta_ry, delta_rz]`
   - 如果返回3个值（仅平移）：`[delta_x, delta_y, delta_z]`
3. **添加夹爪命令**（如果启用）：
   - "open" → `2.0`
   - "close" → `0.0`
   - "no-op" → `1.0`

**输出**：numpy array，形状为 `(3,)`, `(6,)`, `(4,)`, 或 `(7,)`，取决于是否包含旋转和夹爪

---

## 第三层：Action缩放（EEActionWrapper）

**文件位置**：`gym_hil/wrappers/hil_wrappers.py`

```88:113:gym_hil/wrappers/hil_wrappers.py
    def action(self, action):
        """
        Mujoco env is expecting a 7D action space
        [x, y, z, rx, ry, rz, gripper_open]
        For the moment we only control the x, y, z, gripper
        Now supports rotation control if action has 6 dimensions
        """

        # action between -1 and 1, scale to step_size
        action_xyz = action[:3] * self._ee_step_size
        
        # Check if action includes rotation (6D or 7D with gripper)
        if len(action) >= 6:
            # Rotation step size (in radians)
            rotation_step_size = 0.1
            actions_orn = action[3:6] * rotation_step_size
        else:
            # No rotation control
            actions_orn = np.zeros(3)
        gripper_open_command = [0.0]
        if self.use_gripper:
            # NOTE: Normalize gripper action from [0, 2] -> [-1, 1]
            gripper_open_command = [action[-1] - 1.0]

        action = np.concatenate([action_xyz, actions_orn, gripper_open_command])
        return action
```

### 3.1 缩放逻辑

1. **位置缩放**：
   - `action_xyz = action[:3] * ee_step_size`
   - `ee_step_size` 默认：`{"x": 0.025, "y": 0.025, "z": 0.025}` (0.025m = 2.5cm)
   
2. **旋转缩放**（如果action包含旋转）：
   - `actions_orn = action[3:6] * 0.1` （0.1 弧度 ≈ 5.7°）

3. **夹爪归一化**（如果启用）：
   - 输入范围：`[0, 2]`
   - 输出范围：`[-1, 1]`
   - 公式：`gripper = action[-1] - 1.0`

**输出**：7维 numpy array `[x, y, z, rx, ry, rz, gripper]`，单位为米和弧度

---

## 第四层：环境应用（RealCR5PickCubeGymEnv）

**文件位置**：`gym_hil/envs/real_cr5_env.py`

```588:700:gym_hil/envs/real_cr5_env.py
    def _send_control_command(self, action: np.ndarray) -> None:
        """通过 ROS2 Robot 发送控制指令 - 增量控制模式"""
        if self.robot is None:
            raise RuntimeError("Robot not initialized")

        # 获取当前末端执行器位置
        current_obs = self.robot.get_observation()
        current_pos = np.array([
            current_obs["end_effector.position.x"],
            current_obs["end_effector.position.y"],
            current_obs["end_effector.position.z"]
        ])
        # ROS2格式: [x, y, z, w]
        current_ori = np.array([
            current_obs["end_effector.orientation.x"],
            current_obs["end_effector.orientation.y"],
            current_obs["end_effector.orientation.z"],
            current_obs["end_effector.orientation.w"]
        ])

        # 定义动作缩放因子
        position_scale = 0.03  # 5cm 最大位置增量
        orientation_scale = 0.1  # 0.1弧度最大姿态增量

        # 计算目标位置：如果位置动作为零，使用上一次的位置指令
        position_action = action[:3]
        if np.allclose(position_action, 0, atol=1e-6):
            # 位置动作为零，使用上一次的位置指令
            if self._last_position_command is not None:
                target_pos = self._last_position_command.copy()
            else:
                target_pos = current_pos.copy()
                self._last_position_command = target_pos.copy()
        else:
            # 位置动作非零，基于上一次指令计算新位置
            if self._last_position_command is not None:
                target_pos = self._last_position_command + position_action * position_scale
            else:
                target_pos = current_pos + position_action * position_scale
            self._last_position_command = target_pos.copy()

        # 计算目标姿态：如果角度动作为零，使用上一次的角度指令
        orientation_action = action[3:6]
        if np.allclose(orientation_action, 0, atol=1e-6):
            # 角度动作为零，使用上一次的角度指令
            if self._last_orientation_command is not None:
                target_ori = self._last_orientation_command.copy()
            else:
                target_ori = current_ori.copy()
                self._last_orientation_command = target_ori.copy()
        else:
            # 角度动作非零，基于上一次指令计算新姿态
            rx, ry, rz = orientation_action * orientation_scale
            
            # 将欧拉角增量转换为四元数
            # 半角度
            half_angles = np.array([rx, ry, rz]) / 2.0
            cx, cy, cz = np.cos(half_angles)
            sx, sy, sz = np.sin(half_angles)
            
            # 增量四元数（绕x, y, z轴旋转）
            # ROS格式: [x, y, z, w]
            q_inc_x = np.array([sx, 0, 0, cx])  # 绕x轴旋转（roll）
            q_inc_y = np.array([0, sy, 0, cy])  # 绕y轴旋转（pitch）
            q_inc_z = np.array([0, 0, sz, cz])  # 绕z轴旋转（yaw）
            
            # 四元数乘法函数
            def quat_mult_ros(q1, q2):  # [x, y, z, w]格式
                x1, y1, z1, w1 = q1
                x2, y2, z2, w2 = q2
                return np.array([
                    w1*x2 + x1*w2 + y1*z2 - z1*y2,  # x
                    w1*y2 - x1*z2 + y1*w2 + z1*x2,  # y
                    w1*z2 + x1*y2 - y1*x2 + z1*w2,  # z
                    w1*w2 - x1*x2 - y1*y2 - z1*z2   # w
                ])
            
            # 组合增量旋转（按照ZYX顺序）
            q_inc = quat_mult_ros(quat_mult_ros(q_inc_z, q_inc_y), q_inc_x)
            
            # 将增量旋转应用到上一次的指令姿态（不是当前state）
            if self._last_orientation_command is not None:
                base_ori = self._last_orientation_command
            else:
                base_ori = current_ori
            
            target_ori = quat_mult_ros(q_inc, base_ori)
            
            # 归一化四元数
            target_ori = target_ori / np.linalg.norm(target_ori)
            
            # 缓存新的姿态指令
            self._last_orientation_command = target_ori.copy()

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
```

### 4.1 最终转换逻辑

1. **位置计算**（增量累积）：
   - `target_pos = last_position + action[:3] * position_scale`
   - `position_scale = 0.03` (3cm)

2. **姿态计算**（欧拉角增量 → 四元数）：
   - `orientation_scale = 0.1` (0.1 弧度)
   - 欧拉角增量转换为增量四元数
   - 增量四元数 × 上一次指令姿态 = 新的目标姿态

3. **夹爪映射**：
   - 输入：`[-1, 1]` (从 EEActionWrapper)
   - 输出：`[0, 1]` (ROS2 格式)
   - 公式：`gripper_position = (gripper_command + 1.0) / 2.0`

**最终输出**：发送给 ROS2 机器人的目标位置、姿态和夹爪状态

---

## 完整转换链示例

假设左摇杆向前推到底（Y轴 = -1.0，假设已反转）：

```
1. 手柄输入：left_y = -1.0
   ↓
2. GamepadController.get_deltas():
   - y_input = -1.0 (已应用反转和死区)
   - delta_x = -1.0 * 0.01 = -0.01m
   ↓
3. InputsControlWrapper.get_gamepad_action():
   - gamepad_action = [-0.01, 0.0, 0.0, 0.0, 0.0, 0.0]
   ↓
4. EEActionWrapper.action():
   - action_xyz = [-0.01, 0.0, 0.0] * 0.025 = [-0.00025, 0.0, 0.0]
   - 注意：这里有问题！delta已经是米为单位，不应该再乘以step_size
   ↓
5. RealCR5PickCubeGymEnv._send_control_command():
   - position_scale = 0.03
   - target_pos = last_pos + [-0.00025, 0.0, 0.0] * 0.03
   - = last_pos + [-0.0000075, 0.0, 0.0] (7.5微米！)
```

**注意**：当前实现中存在双重缩放问题，可能导致动作过小。

---

## 关键参数总结

| 层级 | 参数 | 默认值 | 说明 |
|------|------|--------|------|
| GamepadController | x_step_size | 0.01m | 左摇杆X轴步长 |
| GamepadController | y_step_size | 0.01m | 左摇杆Y轴步长 |
| GamepadController | z_step_size | 0.01m | 右摇杆Y轴步长 |
| GamepadController | deadzone | 0.1 | 摇杆死区 |
| EEActionWrapper | ee_step_size | 0.025m | 动作空间步长缩放 |
| EEActionWrapper | rotation_step_size | 0.1 rad | 旋转步长缩放 |
| RealCR5Env | position_scale | 0.03m | 最终位置缩放 |
| RealCR5Env | orientation_scale | 0.1 rad | 最终姿态缩放 |

---

## 总结

整个转换流程的核心特点：

1. **增量控制模式**：所有动作都是相对于上一次指令的增量
2. **多级缩放**：从手柄输入到机器人执行经过多次缩放
3. **配置驱动**：手柄映射和反转通过 JSON 配置文件管理
4. **支持旋转**：可选择启用/禁用旋转控制（需要4轴+1 hat）
5. **死区处理**：防止摇杆漂移
6. **姿态处理**：使用四元数进行增量旋转计算

