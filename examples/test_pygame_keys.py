#!/usr/bin/env python3
"""
Pygame 按键测试脚本
显示按下的按键信息，包括键盘按键、鼠标按键等
"""

import pygame
import sys

# 初始化 Pygame
pygame.init()

# 设置窗口
WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600
screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption("Pygame 按键测试")

# 设置字体
font_large = pygame.font.Font(None, 48)
font_medium = pygame.font.Font(None, 36)
font_small = pygame.font.Font(None, 24)

# 颜色定义
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BLUE = (0, 100, 255)
GREEN = (0, 200, 0)
RED = (255, 0, 0)
GRAY = (128, 128, 128)

# 状态变量
pressed_keys = []
current_key_name = ""
last_key_event = None
mouse_pos = (0, 0)
mouse_buttons = {"left": False, "middle": False, "right": False}
clock = pygame.time.Clock()

def get_key_name(key):
    """获取按键名称"""
    try:
        return pygame.key.name(key)
    except:
        return f"Unknown({key})"

def draw_text(text, font, color, x, y, center=False):
    """绘制文本"""
    surface = font.render(str(text), True, color)
    if center:
        rect = surface.get_rect()
        x = x - rect.width // 2
    screen.blit(surface, (x, y))
    return surface.get_height()

def main():
    global pressed_keys, current_key_name, last_key_event, mouse_pos, mouse_buttons
    
    running = True
    
    print("=" * 60)
    print("Pygame 按键测试程序")
    print("=" * 60)
    print("按 ESC 键退出")
    print("按任意键查看按键信息")
    print("-" * 60)
    
    while running:
        screen.fill(BLACK)
        
        # 处理事件
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            
            elif event.type == pygame.KEYDOWN:
                key_name = get_key_name(event.key)
                current_key_name = key_name
                last_key_event = "KEYDOWN"
                
                if event.key not in pressed_keys:
                    pressed_keys.append(event.key)
                
                # 打印到控制台
                print(f"按键按下: {key_name} (键码: {event.key}, Unicode: {event.unicode})")
                
                # ESC 键退出
                if event.key == pygame.K_ESCAPE:
                    running = False
            
            elif event.type == pygame.KEYUP:
                key_name = get_key_name(event.key)
                current_key_name = key_name
                last_key_event = "KEYUP"
                
                if event.key in pressed_keys:
                    pressed_keys.remove(event.key)
                
                print(f"按键释放: {key_name} (键码: {event.key})")
            
            elif event.type == pygame.MOUSEBUTTONDOWN:
                button_names = {1: "左键", 2: "中键", 3: "右键", 4: "滚轮上", 5: "滚轮下"}
                button_name = button_names.get(event.button, f"按钮{event.button}")
                current_key_name = f"鼠标{button_name}"
                last_key_event = "MOUSEBUTTONDOWN"
                
                if event.button == 1:
                    mouse_buttons["left"] = True
                elif event.button == 2:
                    mouse_buttons["middle"] = True
                elif event.button == 3:
                    mouse_buttons["right"] = True
                
                print(f"鼠标按下: {button_name} 位置: {event.pos}")
            
            elif event.type == pygame.MOUSEBUTTONUP:
                button_names = {1: "左键", 2: "中键", 3: "右键", 4: "滚轮上", 5: "滚轮下"}
                button_name = button_names.get(event.button, f"按钮{event.button}")
                current_key_name = f"鼠标{button_name}"
                last_key_event = "MOUSEBUTTONUP"
                
                if event.button == 1:
                    mouse_buttons["left"] = False
                elif event.button == 2:
                    mouse_buttons["middle"] = False
                elif event.button == 3:
                    mouse_buttons["right"] = False
                
                print(f"鼠标释放: {button_name} 位置: {event.pos}")
            
            elif event.type == pygame.MOUSEMOTION:
                mouse_pos = event.pos
        
        # 获取当前按下的所有按键（用于显示持续按下的按键）
        keys_currently_pressed = pygame.key.get_pressed()
        currently_pressed = [get_key_name(i) for i, pressed in enumerate(keys_currently_pressed) if pressed]
        
        # 绘制界面
        y_offset = 30
        
        # 标题
        draw_text("Pygame 按键测试", font_large, WHITE, WINDOW_WIDTH // 2, y_offset, center=True)
        y_offset += 80
        
        # 最后按键事件
        if last_key_event:
            event_color = GREEN if last_key_event == "KEYDOWN" or last_key_event == "MOUSEBUTTONDOWN" else RED
            draw_text(f"最后事件: {last_key_event}", font_medium, event_color, 50, y_offset)
            y_offset += 40
            
            if current_key_name:
                draw_text(f"按键名称: {current_key_name}", font_medium, BLUE, 50, y_offset)
                y_offset += 50
        
        # 当前按下的按键
        draw_text("当前按下的按键:", font_medium, WHITE, 50, y_offset)
        y_offset += 40
        
        if currently_pressed:
            for key in currently_pressed[:10]:  # 最多显示10个
                draw_text(f"  • {key}", font_small, GREEN, 70, y_offset)
                y_offset += 30
        else:
            draw_text("  无", font_small, GRAY, 70, y_offset)
            y_offset += 30
        
        y_offset += 20
        
        # 鼠标信息
        draw_text("鼠标信息:", font_medium, WHITE, 50, y_offset)
        y_offset += 40
        
        mouse_button_text = []
        if mouse_buttons["left"]:
            mouse_button_text.append("左键")
        if mouse_buttons["middle"]:
            mouse_button_text.append("中键")
        if mouse_buttons["right"]:
            mouse_button_text.append("右键")
        
        mouse_status = " | ".join(mouse_button_text) if mouse_button_text else "无"
        draw_text(f"位置: {mouse_pos}", font_small, WHITE, 70, y_offset)
        y_offset += 30
        draw_text(f"按键: {mouse_status}", font_small, WHITE, 70, y_offset)
        y_offset += 40
        
        # 提示信息
        draw_text("提示: 按 ESC 退出 | 按任意键查看按键信息", font_small, GRAY, 50, WINDOW_HEIGHT - 40)
        
        pygame.display.flip()
        clock.tick(60)
    
    pygame.quit()
    print("-" * 60)
    print("程序已退出")
    sys.exit()

if __name__ == "__main__":
    main()

