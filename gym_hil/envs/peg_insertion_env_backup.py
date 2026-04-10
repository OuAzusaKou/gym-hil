import cv2
import mujoco
import numpy as np
from gymnasium import spaces
from typing import Any, Dict, Tuple
from gym_hil.mujoco_gym_env import FrankaGymEnv, GymRenderingSpec
from pathlib import Path

# 修改home position，让joint7绕z轴旋转90度（从0.785变为0.785 + π/2）
_PANDA_HOME = np.asarray([-1.56927488e-05,  7.87927241e-02,  1.55829031e-05, -2.41910629e+00,
 -2.03847233e-06,  2.49289901e+00, -7.85794880e-01])
# _PANDA_HOME = np.asarray((0, 0.195, 0, -2.43, 0, 2.62, 0.785 - np.pi/2))
_CARTESIAN_BOUNDS = np.asarray([[0.45, -0.085, 0], [0.52, 0.235, 0.132]])

class PegInsertionEnv(FrankaGymEnv):
    """Peg Insertion environment with updated peg_head/tail geoms."""

    def __init__(
        self,
        seed: int = 0,
        control_dt: float = 0.1,
        physics_dt: float = 0.002,
        render_spec: GymRenderingSpec = GymRenderingSpec(),
        render_mode: str = "rgb_array",
        image_obs: bool = False,
        # reward_type: str = "dense",
        reward_type: str = "sparse",
    ):
        self.reward_type = reward_type
        super().__init__(
            seed=seed,
            control_dt=control_dt,
            physics_dt=physics_dt,
            render_spec=render_spec,
            render_mode=render_mode,
            image_obs=image_obs,
            home_position=_PANDA_HOME,
            cartesian_bounds=_CARTESIAN_BOUNDS,
            xml_path=Path(__file__).parent.parent / "assets" / "PegInsertionSide_scene.xml",
        )

        # geom/body IDs
        self._peg_head_geom_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_GEOM, "peg_head_geom")
        self._peg_tail_geom_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_GEOM, "peg_tail_geom")
        self._peg_collision_geom_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_GEOM, "peg_collision")
        self._peg_body_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_BODY, "peg")
        self._hole_body_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_BODY, "box_with_hole")


                # 初始化 crop 参数
        self.crop = [10, 50, 100, 50]  # [x, y, width, height]
        self.crop_step = 5  # 每次调整的步长
        # Observation space应该是Dict类型，匹配_compute_observation返回值
        if self.image_obs:
            self.observation_space = spaces.Dict({
                "pixels": spaces.Dict({
                    "front": spaces.Box(0, 255, shape=(render_spec.height, render_spec.width, 3), dtype=np.uint8),
                    "wrist": spaces.Box(0, 255, shape=(render_spec.height, render_spec.width, 3), dtype=np.uint8),
                }),
                "agent_pos": spaces.Box(-np.inf, np.inf, (3,), dtype=np.float32),
                # "environment_state": spaces.Box(-np.inf, np.inf, (43,), dtype=np.float32),
            })
        else:
            self.observation_space = spaces.Dict({
                "agent_pos": spaces.Box(-np.inf, np.inf, (3,), dtype=np.float32),
                "environment_state": spaces.Box(-np.inf, np.inf, (43,), dtype=np.float32),
            })
        self.reward = 0.0

    def reset(self, seed=None, **kwargs):
        super().reset(seed=seed)
        self.reward = 0.0
        mujoco.mj_resetData(self._model, self._data)

        # 先 reset_robot()，保持 robot home 和 mocap 在 home TCP
        self.reset_robot()
        
        # IMPORTANT: 更新mocap姿态以保持夹爪旋转90度
        # 读取当前TCP姿态（已经是旋转后的）
        # tcp_quat = self._data.sensor("2f85/pinch_quat").data.copy()
        tcp_quat = self._data.sensor("changingtek/pinch_quat").data.copy()
        self._data.mocap_quat[0] = tcp_quat

        # 随机 peg 尺寸
        # peg_length = np.random.uniform(0.18, 0.24)
        # peg_radius = np.random.uniform(0.015, 0.025)
        peg_length = 0.2  # 固定长度
        peg_radius = 0.02  # 固定半径
        
        # 更新头尾的尺寸和位置，确保它们紧密相接
        half_length = peg_length / 2
        self._model.geom_size[self._peg_head_geom_id] = np.array([half_length/2, peg_radius, peg_radius])
        self._model.geom_size[self._peg_tail_geom_id] = np.array([half_length/2, peg_radius, peg_radius])
        self._model.geom_size[self._peg_collision_geom_id] = np.array([half_length, peg_radius, peg_radius])
        
        # 更新头尾的位置，使它们紧密相接
        self._model.geom_pos[self._peg_head_geom_id] = np.array([half_length/2, 0, 0])
        self._model.geom_pos[self._peg_tail_geom_id] = np.array([-half_length/2, 0, 0])

        # 随机 peg 位置，但在 home TCP 可达范围内
        peg_pos = np.array([0.5, 0, 0.1])  # 可以稍微随机 ±0.05 m
        self._data.xpos[self._peg_body_id] = peg_pos

        # 孔的位置保持静态
        self._hole_pos = self._data.xpos[self._hole_body_id].copy()

        # 更新 simulation
        mujoco.mj_forward(self._model, self._data)

        # 缓存初始高度
        self._z_init = self._data.xpos[self._peg_body_id, 2]
        self._z_success = self._z_init + 0.1

        obs = self._compute_observation()
        return obs, {}


    
    def step(self, action: np.ndarray) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        """Take a step in the environment."""
        # Apply the action to the robot
        action[3:6] = 0.0
        self.apply_action(action)

        # Compute observation, reward and termination
        obs = self._compute_observation()
        rew = self._compute_reward()
        success = self._is_success()

        terminated = bool(success)
        success = bool(success)
        return obs, rew, terminated, False, {"succeed": success}

    def _compute_observation(self) -> dict:
        tcp_pos = self._data.sensordata[:3].astype(np.float32)
        peg_pos = self._data.xpos[self._peg_body_id].astype(np.float32)
        hole_pos = self._hole_pos.astype(np.float32)

        env_state = np.concatenate([tcp_pos, peg_pos, hole_pos, np.zeros(34, dtype=np.float32)])

        if self.image_obs:
            front_view, wrist_view = self.render()
            # print("front_view.shape", front_view.shape)
            # print("wrist_view.shape", wrist_view.shape)


            # cv2.imshow("Front View", front_view)
            # cv2.waitKey(1)
            # cv2.imshow("Wrist View", wrist_view)
            # cv2.waitKey(1)
            # cv2.imwrite("front_view.png", front_view)
            # cv2.imwrite("wrist_view.png", wrist_view)
            # key = cv2.waitKey(1) & 0xFF
            # if key != 255:  # 有按键被按下
            #     if key == ord('q'):  # q: 增加 x
            #         self.crop[0] += self.crop_step
            #         print(f"Crop x: {self.crop[0]}")
            #     elif key == ord('a'):  # a: 减少 x
            #         self.crop[0] = max(0, self.crop[0] - self.crop_step)
            #         print(f"Crop x: {self.crop[0]}")
            #     elif key == ord('w'):  # w: 增加 y
            #         self.crop[1] += self.crop_step
            #         print(f"Crop y: {self.crop[1]}")
            #     elif key == ord('s'):  # s: 减少 y
            #         self.crop[1] = max(0, self.crop[1] - self.crop_step)
            #         print(f"Crop y: {self.crop[1]}")
            #     elif key == ord('e'):  # e: 增加 width
            #         self.crop[2] += self.crop_step
            #         print(f"Crop width: {self.crop[2]}")
            #     elif key == ord('d'):  # d: 减少 width
            #         self.crop[2] = max(self.crop_step, self.crop[2] - self.crop_step)
            #         print(f"Crop width: {self.crop[2]}")
            #     elif key == ord('r'):  # r: 增加 height
            #         self.crop[3] += self.crop_step
            #         print(f"Crop height: {self.crop[3]}")
            #     elif key == ord('f'):  # f: 减少 height
            #         self.crop[3] = max(self.crop_step, self.crop[3] - self.crop_step)
            #         print(f"Crop height: {self.crop[3]}")
            #     elif key == ord('z'):  # z: 重置为默认值
            #         self.crop = [10, 50, 100, 50]
            #         print(f"Crop reset to: {self.crop}")
            #     elif key == ord('p'):  # p: 打印当前 crop 值
            #         print(f"Current crop: {self.crop}")



                    # 执行裁剪 (x, y, w, h)
            self.crop = [45,45,50,50]
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
                # x, y, cw, ch = [100,200,300,200]

                # wrist_view = safe_crop(wrist_view)

            # 裁剪后再缩放到 out_image_size × out_image_size
            out_sz = (self._render_specs.height, self._render_specs.height)
            if front_view.shape[1] != self._render_specs.height or front_view.shape[0] != self._render_specs.height:
                front_view = cv2.resize(front_view, out_sz)
            if wrist_view.shape[1] != self._render_specs.height or wrist_view.shape[0] != self._render_specs.height:
                wrist_view = cv2.resize(wrist_view, out_sz)

            front_show =front_view.copy()
            front_show = cv2.cvtColor(front_show, cv2.COLOR_RGB2BGR)
            wrist_show = wrist_view.copy()
            wrist_show = cv2.cvtColor(wrist_show, cv2.COLOR_RGB2BGR)

            cv2.imshow("Front View", front_show)
            cv2.waitKey(1)
            cv2.imshow("Wrist View", wrist_show)
            cv2.waitKey(1)





            return {
                "pixels": {"front": front_view, "wrist": wrist_view},
                "agent_pos": tcp_pos,
                # "environment_state": env_state,
            }
        else:
            return {"agent_pos": tcp_pos, "environment_state": env_state}

    def _compute_reward(self) -> float:
        tcp_pos = self._data.sensordata[:3]  # 末端执行器位置
        peg_pos = self._data.xpos[self._peg_body_id]

        # reach reward
        dist_to_peg = np.linalg.norm(tcp_pos - peg_pos)
        r_reach = 1 - np.tanh(4.0 * dist_to_peg)

        # grasp判定：lift-based
        lift_height = 0.05  # 提升高度阈值，可根据实际调整
        peg_base_height = 0.02  # peg初始高度
        is_grasped = (peg_pos[2] - peg_base_height) > lift_height

        # pre-insertion reward（Y-Z平面对齐）
        peg_to_hole_yz = np.linalg.norm((peg_pos - self._hole_pos)[1:])
        pre_insertion_reward = 3 * (1 - np.tanh(0.5 * peg_to_hole_yz + 4.5 * peg_to_hole_yz))
        pre_insertion_reward *= is_grasped  # 只有抓住才计算

        # insertion reward（整体对齐）
        insertion_dist = np.linalg.norm(peg_pos - self._hole_pos)
        inserted = insertion_dist < 0.15 and is_grasped
        insertion_reward = 5 * (1 - np.tanh(5.0 * insertion_dist)) * inserted

        # 总 reward
        reward = r_reach + float(is_grasped) + pre_insertion_reward + insertion_reward

        # sparse reward
        if self.reward_type == "sparse":
            if inserted:
                self.reward = 1.0
                reward = 1.0
            else:
                self.reward = 0.0
                reward = 0.0

        return float(reward)


    def _is_success(self) -> bool:
        # peg_pos = self._data.xpos[self._peg_body_id]
        # peg_to_hole_yz = np.linalg.norm((peg_pos - self._hole_pos)[1:])
        # return peg_to_hole_yz < 0.01
        return True if self.reward == 1.0 else False
