#!/usr/bin/env python3
"""
简单的pygame测试脚本
用于显示当前按下的按键和各个axis的数值
"""

import pygame
import sys
import os

# 初始化pygame
pygame.init()
pygame.joystick.init()

# 检查是否有游戏手柄连接
joystick_count = pygame.joystick.get_count()
joysticks = []

if joystick_count > 0:
    print(f"检测到 {joystick_count} 个游戏手柄:")
    for i in range(joystick_count):
        joystick = pygame.joystick.Joystick(i)
        joystick.init()
        joysticks.append(joystick)
        print(f"  [{i}] {joystick.get_name()}")
else:
    print("未检测到游戏手柄，仅支持键盘输入")

# 初始化时钟
clock = pygame.time.Clock()

print("\n" + "="*60)
print("按 ESC 或 Ctrl+C 退出")
print("="*60 + "\n")

running = True
pressed_keys = set()
last_output = ""

try:
    while running:
        # 处理事件
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                pressed_keys.add(event.key)
            elif event.type == pygame.KEYUP:
                pressed_keys.discard(event.key)

        # 获取当前按下的按键名称
        key_names = []
        for key in sorted(pressed_keys):
            key_name = pygame.key.name(key)
            key_names.append(key_name)

        # 构建输出信息
        output_lines = []
        
        # 键盘按键信息
        if key_names:
            output_lines.append(f"按键: {', '.join(key_names)}")
        else:
            output_lines.append("按键: (无)")

        # 游戏手柄信息
        if joysticks:
            for idx, joystick in enumerate(joysticks):
                output_lines.append(f"\n游戏手柄 [{idx}] ({joystick.get_name()}):")
                
                # 显示各个axis的数值
                num_axes = joystick.get_numaxes()
                if num_axes > 0:
                    axes_values = []
                    for i in range(num_axes):
                        axis_value = joystick.get_axis(i)
                        # 格式化显示，保留3位小数
                        axes_values.append(f"Axis[{i}]: {axis_value:7.3f}")
                    output_lines.append("  " + " | ".join(axes_values))
                else:
                    output_lines.append("  无摇杆轴")
                
                # 显示按钮状态
                num_buttons = joystick.get_numbuttons()
                if num_buttons > 0:
                    pressed_buttons = []
                    for i in range(num_buttons):
                        if joystick.get_button(i):
                            pressed_buttons.append(f"Button[{i}]")
                    if pressed_buttons:
                        output_lines.append("  按钮: " + ", ".join(pressed_buttons))
                
                # 显示hat（方向键）状态
                num_hats = joystick.get_numhats()
                if num_hats > 0:
                    hat_values = []
                    for i in range(num_hats):
                        hat = joystick.get_hat(i)
                        if hat != (0, 0):  # 只显示非零值
                            hat_values.append(f"Hat[{i}]: {hat}")
                    if hat_values:
                        output_lines.append("  方向键: " + ", ".join(hat_values))

        # 只在输出变化时打印（避免刷屏）
        current_output = "\n".join(output_lines)
        if current_output != last_output:
            # 清屏（在支持的终端中）
            os.system('clear' if os.name != 'nt' else 'cls')
            print("="*60)
            print(f"时间: {pygame.time.get_ticks() // 1000}秒")
            print("="*60)
            print("\n".join(output_lines))
            print("\n" + "="*60)
            print("按 ESC 或 Ctrl+C 退出")
            print("="*60)
            last_output = current_output

        # 控制更新频率
        clock.tick(30)  # 30 FPS

except KeyboardInterrupt:
    print("\n\n程序被用户中断")

finally:
    # 清理
    for joystick in joysticks:
        joystick.quit()
    pygame.quit()
    print("\n程序已退出")

