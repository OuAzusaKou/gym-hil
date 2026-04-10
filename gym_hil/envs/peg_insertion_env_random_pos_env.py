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
        reward_type: str = "sparse",
        # ── 新增：场景切换参数 ──────────────────────────────────────
        scene_mode: str = "random",  # "cycle" | "random" | "fixed"
        scene_index: int = 0,       # 仅 scene_mode="fixed" 时有效
        # ────────────────────────────────────────────────────────────
    ):
        self.reward_type = reward_type

        # ── 新增：多场景配置 ────────────────────────────────────────
        _assets_dir = Path(__file__).parent.parent / "assets"
        self._xml_paths = [
            _assets_dir / "PegInsertionSide_scene.xml",   # Scene0 原始
            # _assets_dir / "PegInsertionSide_scene_grid_black.xml",  # Scene1 暗黑
            # _assets_dir / "PegInsertionSide_scene_grid_brown.xml",  # Scene2 棕色地板
            # _assets_dir / "PegInsertionSide_scene_light_dark.xml",  # Scene3 暗光
            _assets_dir / "PegInsertionSide_scene_skybox_yellow.xml",  # Scene4 暖光
        ]
        self._scene_mode    = scene_mode
        self._fixed_index   = scene_index
        self._episode_count = 0
        self._render_spec   = render_spec  # 保存供 _reload_model 使用
        # ────────────────────────────────────────────────────────────

        super().__init__(
            seed=seed,
            control_dt=control_dt,
            physics_dt=physics_dt,
            render_spec=render_spec,
            render_mode=render_mode,
            image_obs=image_obs,
            home_position=_PANDA_HOME,
            cartesian_bounds=_CARTESIAN_BOUNDS,
            xml_path=self._xml_paths[0],  # 初始化用第0个场景
        )

        # geom/body IDs
        self._bind_peg_ids()

        # 初始化 crop 参数
        self.crop = [10, 50, 100, 50]
        self.crop_step = 5

        if self.image_obs:
            self.observation_space = spaces.Dict({
                "pixels": spaces.Dict({
                    "front": spaces.Box(0, 255, shape=(render_spec.height, render_spec.width, 3), dtype=np.uint8),
                    "wrist": spaces.Box(0, 255, shape=(render_spec.height, render_spec.width, 3), dtype=np.uint8),
                }),
                "agent_pos": spaces.Box(-np.inf, np.inf, (3,), dtype=np.float32),
            })
        else:
            self.observation_space = spaces.Dict({
                "agent_pos": spaces.Box(-np.inf, np.inf, (3,), dtype=np.float32),
                "environment_state": spaces.Box(-np.inf, np.inf, (43,), dtype=np.float32),
            })
        self.reward = 0.0

    def _bind_peg_ids(self):
        """绑定 peg/hole 相关的 geom/body ID（每次重载模型后调用）。"""
        self._peg_head_geom_id      = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_GEOM, "peg_head_geom")
        self._peg_tail_geom_id      = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_GEOM, "peg_tail_geom")
        self._peg_collision_geom_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_GEOM, "peg_collision")
        self._peg_body_id           = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_BODY, "peg")
        self._hole_body_id          = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_BODY, "box_with_hole")

    def _pick_xml(self) -> Path:
        """根据 scene_mode 选出本 episode 的 XML 路径。"""
        n = len(self._xml_paths)
        if self._scene_mode == "fixed":
            idx = self._fixed_index % n
        elif self._scene_mode == "random":
            idx = np.random.randint(0, n)
        else:  # "cycle"
            idx = self._episode_count % n
        xml = self._xml_paths[idx]
        print(f"[PegInsertionEnv] Episode {self._episode_count} → Scene {idx} ({xml.name})")
        return xml

    def _reload_model(self, xml_path: Path):
        """
        重新加载 MuJoCo 模型，并同步更新父类 FrankaGymEnv 中所有
        依赖 model/data 的缓存引用，确保物理仿真和渲染都使用新模型。
        """
        # 1. 重建 model / data
        self._model = mujoco.MjModel.from_xml_path(str(xml_path))
        self._model.vis.global_.offwidth  = self._render_spec.width
        self._model.vis.global_.offheight = self._render_spec.height
        self._data = mujoco.MjData(self._model)

        # 2. 同步父类 FrankaGymEnv 中缓存的 robot ID
        #    （这些在 FrankaGymEnv.__init__ 里绑定，换模型后必须重新绑定）
        self._panda_dof_ids  = np.asarray([self._model.joint(f"joint{i}").id for i in range(1, 8)])
        self._panda_ctrl_ids = np.asarray([self._model.actuator(f"actuator{i}").id for i in range(1, 8)])
        self._gripper_ctrl_id = self._model.actuator("gripper_actuator").id
        self._pinch_site_id   = self._model.site("pinch").id

        # 3. 同步 camera ID
        camera_id_1 = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_CAMERA, "front")
        camera_id_2 = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_CAMERA, "handcam_rgb")
        self.camera_id = (camera_id_1, camera_id_2)

        # 4. 重建离屏渲染器（_viewer 绑定了旧 model，必须重建）
        if self._viewer is not None:
            try:
                self._viewer.close()
            except Exception:
                pass
        self._viewer = mujoco.Renderer(
            self._model,
            height=self._render_spec.height,
            width=self._render_spec.width,
        )

        # 5. 重新绑定 peg/hole ID
        self._bind_peg_ids()

    # def _reload_model(self, xml_path: Path):
    #     # 加载新模型
    #     new_model = mujoco.MjModel.from_xml_path(str(xml_path))
        
    #     # 检查结构是否兼容（你的5个XML结构完全相同，所以一定兼容）
    #     assert new_model.nq == self._model.nq
    #     assert new_model.nbody == self._model.nbody
        
    #     # 原地把新模型的数据复制进旧模型（指针不变，内容变）
    #     mujoco.mj_copyModel(self._model, new_model)
        
    #     # data 不需要重建，resetData 即可
    #     mujoco.mj_resetData(self._model, self._data)
        
    #     # peg/hole ID 在结构相同的模型里不会变，但重新绑定一次更安全
    #     self._bind_peg_ids()

    def reset(self, seed=None, **kwargs):
        if seed is not None:
            np.random.seed(seed)

        # ── 切换场景 ────────────────────────────────────────────────
        xml_path = self._pick_xml()
        self._reload_model(xml_path)   # 必须在 super().reset() 之前
        self._episode_count += 1
        # ────────────────────────────────────────────────────────────

        super().reset(seed=seed)
        self.reward = 0.0
        mujoco.mj_resetData(self._model, self._data)

        # 先 reset_robot()，保持 robot home 和 mocap 在 home TCP
        self.reset_robot()

        tcp_quat = self._data.sensor("changingtek/pinch_quat").data.copy()
        self._data.mocap_quat[0] = tcp_quat

        peg_length = 0.2
        peg_radius = 0.02
        half_length = peg_length / 2
        self._model.geom_size[self._peg_head_geom_id] = np.array([half_length/2, peg_radius, peg_radius])
        self._model.geom_size[self._peg_tail_geom_id] = np.array([half_length/2, peg_radius, peg_radius])
        self._model.geom_size[self._peg_collision_geom_id] = np.array([half_length, peg_radius, peg_radius])
        self._model.geom_pos[self._peg_head_geom_id] = np.array([half_length/2, 0, 0])
        self._model.geom_pos[self._peg_tail_geom_id] = np.array([-half_length/2, 0, 0])

        rand_peg_pos = self.get_random_peg_position()
        self._data.jnt("peg_free").qpos[:3] = rand_peg_pos

        self._hole_pos = self._data.xpos[self._hole_body_id].copy()

        mujoco.mj_forward(self._model, self._data)

        self._z_init    = self._data.xpos[self._peg_body_id, 2]
        self._z_success = self._z_init + 0.1

        current_peg_pos = self._data.xpos[self._peg_body_id]
        print(f"Peg随机位置：x={current_peg_pos[0]:.3f}, y={current_peg_pos[1]:.3f}, z={current_peg_pos[2]:.3f}")

        obs = self._compute_observation()
        return obs, {}

    def get_random_peg_position(self):
        rand_x = np.random.uniform(0.45, 0.52)
        rand_y = np.random.uniform(-0.01, 0.06)
        fixed_z = 0.03
        return np.array([rand_x, rand_y, fixed_z])

    def step(self, action: np.ndarray) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        """Take a step in the environment."""
        action[3:6] = 0.0
        self.apply_action(action)
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
                        return img
                    return img[y0:y1, x0:x1]
                front_view = safe_crop(front_view)

            out_sz = (self._render_specs.height, self._render_specs.height)
            if front_view.shape[1] != self._render_specs.height or front_view.shape[0] != self._render_specs.height:
                front_view = cv2.resize(front_view, out_sz)
            if wrist_view.shape[1] != self._render_specs.height or wrist_view.shape[0] != self._render_specs.height:
                wrist_view = cv2.resize(wrist_view, out_sz)

            front_show = cv2.cvtColor(front_view.copy(), cv2.COLOR_RGB2BGR)
            wrist_show = cv2.cvtColor(wrist_view.copy(), cv2.COLOR_RGB2BGR)
            cv2.imshow("Front View", front_show)
            cv2.waitKey(1)
            cv2.imshow("Wrist View", wrist_show)
            cv2.waitKey(1)

            return {
                "pixels": {"front": front_view, "wrist": wrist_view},
                "agent_pos": tcp_pos,
            }
        else:
            return {"agent_pos": tcp_pos, "environment_state": env_state}

    def _compute_reward(self) -> float:
        tcp_pos = self._data.sensordata[:3]
        peg_pos = self._data.xpos[self._peg_body_id]

        dist_to_peg = np.linalg.norm(tcp_pos - peg_pos)
        r_reach = 1 - np.tanh(4.0 * dist_to_peg)

        lift_height = 0.05
        peg_base_height = 0.02
        is_grasped = (peg_pos[2] - peg_base_height) > lift_height

        peg_to_hole_yz = np.linalg.norm((peg_pos - self._hole_pos)[1:])
        pre_insertion_reward = 3 * (1 - np.tanh(0.5 * peg_to_hole_yz + 4.5 * peg_to_hole_yz))
        pre_insertion_reward *= is_grasped

        insertion_dist = np.linalg.norm(peg_pos - self._hole_pos)
        inserted = insertion_dist < 0.15 and is_grasped
        insertion_reward = 5 * (1 - np.tanh(5.0 * insertion_dist)) * inserted

        reward = r_reach + float(is_grasped) + pre_insertion_reward + insertion_reward

        if self.reward_type == "sparse":
            if inserted:
                self.reward = 1.0
                reward = 1.0
            else:
                self.reward = 0.0
                reward = 0.0

        return float(reward)

    def _is_success(self) -> bool:
        return True if self.reward == 1.0 else False