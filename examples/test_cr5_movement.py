#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
CR5 Robot X-axis Movement Demo with Gripper Control

This script demonstrates CR5 robot control using the RealCR5PickCubeGymEnv environment.
The robot performs x-axis back-and-forth movements with gripper open/close control:
- When moving to target position: gripper opens
- When returning to original position: gripper closes

Usage:
    python test_cr5_movement.py [options]

Options:
    --move-distance: Movement distance in meters (default: 0.1)
    --num-cycles: Number of movement cycles (default: 4)
    --move-duration: Duration for each movement in seconds (default: 3.0)
    --pause-duration: Pause duration at each position in seconds (default: 1.0)
    --control-dt: Control frequency in seconds (default: 0.1)

Requirements:
    - ROS2 environment must be sourced
    - lerobot_robot_ros2 package must be installed
    - CR5 robot must be running and publishing joint states
    - Gripper control topic "left_gripper_joint/position_command" must be available
"""

import argparse
import signal
import sys
import time
from typing import Any, Dict

import gymnasium as gym
import numpy as np

import gym_hil  # noqa: F401


def main():
    """CR5 机械臂 x 方向往返运动演示（带夹爪控制）"""
    parser = argparse.ArgumentParser(description="CR5 Robot X-axis Movement Demo")
    parser.add_argument(
        "--render-mode", 
        type=str, 
        default="rgb_array", 
        choices=["rgb_array", "human"], 
        help="Rendering mode"
    )
    parser.add_argument(
        "--image-obs", 
        action="store_true", 
        help="Enable image observations"
    )
    parser.add_argument(
        "--control-dt", 
        type=float, 
        default=0.1, 
        help="Control frequency (seconds)"
    )
    parser.add_argument(
        "--move-distance", 
        type=float, 
        default=0.1, 
        help="Movement distance in meters"
    )
    parser.add_argument(
        "--num-cycles", 
        type=int, 
        default=4, 
        help="Number of movement cycles"
    )
    parser.add_argument(
        "--move-duration", 
        type=float, 
        default=3.0, 
        help="Duration for each movement (seconds)"
    )
    parser.add_argument(
        "--pause-duration", 
        type=float, 
        default=1.0, 
        help="Pause duration at each position (seconds)"
    )
    args = parser.parse_args()

    print("CR5 Robot X-axis Movement Demo with Gripper Control")
    print("=" * 60)
    print("This demo will:")
    print("- Move robot back and forth in x-direction")
    print("- Open gripper when moving to target position")
    print("- Close gripper when returning to original position")
    print(f"Movement distance: {args.move_distance}m")
    print(f"Number of cycles: {args.num_cycles}")
    print(f"Move duration: {args.move_duration}s")
    print(f"Pause duration: {args.pause_duration}s")
    print("Press Ctrl+C to stop")
    print("-" * 60)

    # 创建 CR5 环境

    env = gym.make(
        "gym_hil/RealCR5PickCube-v0",
        render_mode=args.render_mode,
        image_obs=args.image_obs,
        control_dt=args.control_dt,
        reward_type="sparse"
    )
    print("✓ CR5 environment created successfully")


    # 重置环境
    try:
        obs, info = env.reset()
        print("✓ Environment reset successfully")
        print(f"Observation keys: {list(obs.keys())}")
        print(f"Observation space: {env.observation_space}")
        print(f"Action space: {env.action_space}")
    except Exception as e:
        print(f"❌ Failed to reset environment: {e}")
        env.close()
        return

    # 获取初始位置
    try:
        print(f"✓ Initial observation obtained")
    except Exception as e:
        print(f"❌ Failed to get initial observation: {e}")
        env.close()
        return

    # 信号处理
    env_closed = False

    def signal_handler(sig, frame):
        nonlocal env_closed
        print("\n\nShutting down...")
        if not env_closed:
            env.close()
            env_closed = True
            print("✓ Environment closed successfully")
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    # 运动参数
    cycle_count = 0
    move_to_target = True
    move_distance = args.move_distance
    num_cycles = args.num_cycles
    move_duration = args.move_duration
    pause_duration = args.pause_duration

    print(f"\nStarting {num_cycles} movement cycles...")
    print(f"Moving {move_distance}m in x-direction")
    print("-" * 50)

    # try:
    while cycle_count < num_cycles:
        cycle_count += 1
        
        if move_to_target:
            # 向右移动并打开夹爪
            print(f"Cycle {cycle_count}/{num_cycles}: Moving RIGHT (+{move_distance}m) with gripper OPEN")
            action = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)  # 最后一位是夹爪控制
            # 计算动作值：move_distance / position_scale = move_distance / 0.05
        else:
            # 向左移动（回到原位）并关闭夹爪
            print(f"Cycle {cycle_count}/{num_cycles}: Moving LEFT (-{move_distance}m) with gripper CLOSED")
            action = np.array([-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0], dtype=np.float32)  # 最后一位是夹爪控制
            # 计算动作值：move_distance / position_scale = move_distance / 0.05
        # 执行动作
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"✓ Action executed successfully")

        # 显示当前状态
        # 尝试从观察中获取位置信息
        if "agent_pos" in obs and len(obs["agent_pos"]) >= 16:
            # agent_pos 包含: 关节位置(6) + 关节速度(6) + TCP位置(3) + 夹爪位置(1)
            tcp_pos = obs["agent_pos"][-4:-1]  # TCP位置 (倒数第4到倒数第2个)
            gripper_pos = obs["agent_pos"][-1]  # 夹爪位置 (最后一个)
            print(f"  Current TCP position: x={tcp_pos[0]:.3f}, y={tcp_pos[1]:.3f}, z={tcp_pos[2]:.3f}")
            print(f"  Current gripper position: {gripper_pos:.3f} ({'OPEN' if gripper_pos > 0.5 else 'CLOSED'})")
        else:
            print(f"  Observation shape: {obs.get('agent_pos', 'N/A').shape if hasattr(obs.get('agent_pos', None), 'shape') else 'N/A'}")

        # 等待移动完成
        print(f"  Waiting {move_duration}s for movement to complete...")
        time.sleep(move_duration)

        # 切换方向
        move_to_target = not move_to_target

        # 在位置暂停
        if cycle_count < num_cycles:  # 不在最后一个周期后暂停
            print(f"  Pausing for {pause_duration}s...")
            time.sleep(pause_duration)

    print(f"\n✓ Completed {num_cycles} movement cycles successfully!")
    print("Demo finished.")


if __name__ == "__main__":
    main()
