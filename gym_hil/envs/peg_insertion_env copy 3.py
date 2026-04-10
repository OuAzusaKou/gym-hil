import cv2
import mujoco
import numpy as np
from gymnasium import spaces
from typing import Any, Dict, Tuple
from gym_hil.mujoco_gym_env import FrankaGymEnv, GymRenderingSpec
from pathlib import Path

_PANDA_HOME = np.asarray([-1.56927488e-05,  7.87927241e-02,  1.55829031e-05, -2.41910629e+00,
 -2.03847233e-06,  2.49289901e+00, -7.85794880e-01])
_CARTESIAN_BOUNDS = np.asarray([[0.45, -0.085, 0], [0.52, 0.235, 0.132]])

# ──────────────────────────────────────────────────────────────────────────────
# 5个XML文件的差异对比（其余结构完全相同）：
#
#              | headlight diffuse | headlight ambient | light0 diffuse | light1 diffuse | floor rgb
# -------------|-------------------|-------------------|----------------|----------------|-------------------
# 原始 scene0  | .4 .4 .4          | .5 .5 .5          | 1 1 1          | .8 .8 .8       | .1 .2 .3 / .2 .3 .4
# xml1 scene1  | .8 .8 .8          | .2 .2 .2          | 1 1 1          | .8 .8 .8       | .1 .1 .12/ .15 .15 .18
# xml2 scene2  | .4 .4 .4          | .5 .5 .5          | 1 1 1          | .8 .8 .8       | .4 .3 .2 / .6 .5 .4
# xml3 scene3  | .4 .4 .4          | .5 .5 .5          | .3 .3 .3       | .3 .3 .3       | .1 .2 .3 / .2 .3 .4
# xml4 scene4  | .6 .55 .5         | .6 .55 .5         | 1 1 1          | .8 .8 .8       | .1 .2 .3 / .2 .3 .4
#
# floor_rgba 取两色均值，用 geom_rgba 覆盖（纹理格会被乘以该颜色）
# ──────────────────────────────────────────────────────────────────────────────

_SCENE_CONFIGS = [
    {
        "name": "Scene0_default",
        "headlight_diffuse": np.array([0.4,  0.4,  0.4 ]),
        "headlight_ambient": np.array([0.5,  0.5,  0.5 ]),
        "light0_diffuse":    np.array([1.0,  1.0,  1.0 ]),
        "light1_diffuse":    np.array([0.8,  0.8,  0.8 ]),
        "floor_rgb1":        np.array([0.1, 0.2, 0.3]),   
        "floor_rgb2":        np.array([0.2, 0.3, 0.4]),
    },
    {
        "name": "Scene1_dark",
        "headlight_diffuse": np.array([0.8,   0.8,   0.8  ]),
        "headlight_ambient": np.array([0.2,   0.2,   0.2  ]),
        "light0_diffuse":    np.array([1.0,   1.0,   1.0  ]),
        "light1_diffuse":    np.array([0.8,   0.8,   0.8  ]),
        "floor_rgb1":        np.array([0.1, 0.1, 0.12]),  
        "floor_rgb2":        np.array([0.15, 0.15, 0.18]),
    },
    {
        "name": "Scene2_brown_floor",
        "headlight_diffuse": np.array([0.4,  0.4,  0.4 ]),
        "headlight_ambient": np.array([0.5,  0.5,  0.5 ]),
        "light0_diffuse":    np.array([1.0,  1.0,  1.0 ]),
        "light1_diffuse":    np.array([0.8,  0.8,  0.8 ]),
        "floor_rgb1":        np.array([0.4, 0.3, 0.2]), 
        "floor_rgb2":        np.array([0.6, 0.5, 0.4]),
    },
    {
        "name": "Scene3_dim_light",
        "headlight_diffuse": np.array([0.4,  0.4,  0.4 ]),
        "headlight_ambient": np.array([0.5,  0.5,  0.5 ]),
        "light0_diffuse":    np.array([0.3,  0.3,  0.3 ]),         # ← 主光调暗
        "light1_diffuse":    np.array([0.3,  0.3,  0.3 ]),         # ← 辅光调暗
        "floor_rgb1":        np.array([0.1, 0.2, 0.3]),   
        "floor_rgb2":        np.array([0.2, 0.3, 0.4]),
    },
    {
        "name": "Scene4_warm",
        "headlight_diffuse": np.array([0.6,  0.55, 0.5 ]),
        "headlight_ambient": np.array([0.6,  0.55, 0.5 ]),
        "light0_diffuse":    np.array([1.0,  1.0,  1.0 ]),
        "light1_diffuse":    np.array([0.8,  0.8,  0.8 ]),
        "floor_rgb1":        np.array([0.1, 0.2, 0.3]),   
        "floor_rgb2":        np.array([0.2, 0.3, 0.4]),
    },
]


class PegInsertionEnv(FrankaGymEnv):
    """
    Peg Insertion environment.
    使用单一XML文件，每次 reset() 时通过直接修改 self._model 的视觉属性
    来切换场景外观，无需重新加载模型。

    scene_mode:
        "cycle"  — 按顺序循环：episode 0→Scene0, 1→Scene1, ..., 5→Scene0, ...
        "random" — 每次 reset 随机选一个场景
        "fixed"  — 始终使用 scene_index 指定的场景
    """

    def __init__(
        self,
        seed: int = 0,
        control_dt: float = 0.1,
        physics_dt: float = 0.002,
        render_spec: GymRenderingSpec = GymRenderingSpec(),
        render_mode: str = "rgb_array",
        image_obs: bool = False,
        reward_type: str = "sparse",
        scene_mode: str = "cycle",
        scene_index: int = 0,
    ):
        self.reward_type    = reward_type
        self._scene_mode    = scene_mode
        self._fixed_index   = scene_index
        self._episode_count = 0
        self._scenes        = _SCENE_CONFIGS

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

        self._bind_ids()

        self.crop      = [10, 50, 100, 50]
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
                "agent_pos":        spaces.Box(-np.inf, np.inf, (3,),  dtype=np.float32),
                "environment_state": spaces.Box(-np.inf, np.inf, (43,), dtype=np.float32),
            })

        self.reward = 0.0

    # ── ID 绑定 ───────────────────────────────────────────────────────────────

    def _bind_ids(self):
        G = mujoco.mjtObj.mjOBJ_GEOM
        B = mujoco.mjtObj.mjOBJ_BODY
        self._peg_head_geom_id      = mujoco.mj_name2id(self._model, G, "peg_head_geom")
        self._peg_tail_geom_id      = mujoco.mj_name2id(self._model, G, "peg_tail_geom")
        self._peg_collision_geom_id = mujoco.mj_name2id(self._model, G, "peg_collision")
        self._floor_geom_id         = mujoco.mj_name2id(self._model, G, "floor")
        self._peg_body_id           = mujoco.mj_name2id(self._model, B, "peg")
        self._hole_body_id          = mujoco.mj_name2id(self._model, B, "box_with_hole")
        # light 按 XML 顺序：0 = "0 0 3"（主光），1 = "0 -.5 .4"（辅光）
        self._light0_id = 0
        self._light1_id = 1

    # ── 场景选择与应用 ────────────────────────────────────────────────────────

    def _pick_scene(self) -> dict:
        n = len(self._scenes)
        if self._scene_mode == "fixed":
            idx = self._fixed_index % n
        elif self._scene_mode == "random":
            idx = np.random.randint(0, n)
        else:  # "cycle"
            idx = self._episode_count % n
        scene = self._scenes[idx]
        print(f"[PegInsertionEnv] Episode {self._episode_count} → {scene['name']} ({idx+1}/{n})")
        return scene

    def _apply_scene(self, scene: dict):
        """
        把场景配置写入 self._model 的视觉属性。
        在 mj_forward() 之前调用，下一帧渲染即生效，不需要重载模型。
        """
        # headlight（vis 结构体，直接切片赋值）
        self._model.vis.headlight.diffuse[:] = scene["headlight_diffuse"]
        self._model.vis.headlight.ambient[:] = scene["headlight_ambient"]

        # 场景灯光 diffuse（shape: [n_lights, 3]）
        self._model.light_diffuse[self._light0_id] = scene["light0_diffuse"]
        self._model.light_diffuse[self._light1_id] = scene["light1_diffuse"]

        # 地板 geom rgba（纹理格颜色会与 rgba 相乘，从而产生色调变化）
        # self._model.geom_rgba[self._floor_geom_id] = scene["floor_rgba"]
        self._set_floor_texture(scene["floor_rgb1"], scene["floor_rgb2"])

    # def _upload_texture(self, tex_id: int):
    #     ctx = self._viewer._mjr_context
    #     mujoco.mjr_uploadTexture(self._model, ctx, tex_id)
    def _upload_texture(self, tex_id: int):
        # 离屏渲染器（cv2窗口用的）
        if hasattr(self, '_viewer') and self._viewer is not None:
            mujoco.mjr_uploadTexture(self._model, self._viewer._mjr_context, tex_id)

        # 交互式 viewer（MuJoCo 弹出的可交互窗口）
        # 常见路径，逐一排查
        for attr in ['_passive_viewer', '_interactive_viewer', '_mujoco_viewer', 'viewer']:
            v = getattr(self, attr, None)
            if v is not None:
                print(f"[Texture] found interactive viewer at self.{attr}, type={type(v)}")
                # 打印它的属性找 context
                ctx_attrs = [a for a in dir(v) if 'ctx' in a.lower() or 'context' in a.lower() or 'mjr' in a.lower()]
                print(f"[Texture] context-related attrs: {ctx_attrs}")
                break

    def _set_floor_texture(self, rgb1: np.ndarray, rgb2: np.ndarray):
        tex_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_TEXTURE, "grid")
        tex_h  = self._model.tex_height[tex_id]
        tex_w  = self._model.tex_width[tex_id]
        adr    = self._model.tex_adr[tex_id]

        c1 = (np.array(rgb1) * 255).astype(np.uint8)
        c2 = (np.array(rgb2) * 255).astype(np.uint8)

        tex_flat = self._model.tex_data[adr : adr + tex_h * tex_w * 3].reshape(tex_h, tex_w, 3)

        cell = tex_h // 2
        ii, jj = np.meshgrid(np.arange(tex_h), np.arange(tex_w), indexing='ij')
        mask = ((ii // cell) + (jj // cell)) % 2 == 0
        tex_flat[mask]  = c1
        tex_flat[~mask] = c2

        # 同步到 GPU
        self._upload_texture(tex_id)

    # ── reset ─────────────────────────────────────────────────────────────────

    def reset(self, seed=None, **kwargs):
        if seed is not None:
            np.random.seed(seed)

        scene = self._pick_scene()          # 选场景
        self._episode_count += 1

        super().reset(seed=seed)
        self.reward = 0.0
        mujoco.mj_resetData(self._model, self._data)

        self._apply_scene(scene)            # ← 应用视觉属性（在 mj_forward 之前）

        self.reset_robot()

        tcp_quat = self._data.sensor("changingtek/pinch_quat").data.copy()
        self._data.mocap_quat[0] = tcp_quat

        peg_length  = 0.2
        peg_radius  = 0.02
        half_length = peg_length / 2
        self._model.geom_size[self._peg_head_geom_id]      = np.array([half_length / 2, peg_radius, peg_radius])
        self._model.geom_size[self._peg_tail_geom_id]      = np.array([half_length / 2, peg_radius, peg_radius])
        self._model.geom_size[self._peg_collision_geom_id] = np.array([half_length,     peg_radius, peg_radius])
        self._model.geom_pos[self._peg_head_geom_id]       = np.array([ half_length / 2, 0, 0])
        self._model.geom_pos[self._peg_tail_geom_id]       = np.array([-half_length / 2, 0, 0])

        rand_peg_pos = self.get_random_peg_position()
        self._data.jnt("peg_free").qpos[:3] = rand_peg_pos

        self._hole_pos = self._data.xpos[self._hole_body_id].copy()

        mujoco.mj_forward(self._model, self._data)

        self._z_init    = self._data.xpos[self._peg_body_id, 2]
        self._z_success = self._z_init + 0.1

        current_peg_pos = self._data.xpos[self._peg_body_id]
        print(f"Peg随机位置：x={current_peg_pos[0]:.3f}, y={current_peg_pos[1]:.3f}, z={current_peg_pos[2]:.3f}")
        # render() 之后再上传纹理，确保 context 存在
        if self.image_obs:
            tex_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_TEXTURE, "grid")
            self._upload_texture(tex_id)

        obs = self._compute_observation()
        return obs, {}

    # ── 其余方法 ──────────────────────────────────────────────────────────────

    def get_random_peg_position(self):
        return np.array([
            np.random.uniform(0.45, 0.52),
            np.random.uniform(-0.01, 0.06),
            0.03,
        ])

    def step(self, action: np.ndarray) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        action[3:6] = 0.0
        self.apply_action(action)
        obs     = self._compute_observation()
        rew     = self._compute_reward()
        success = self._is_success()
        return obs, rew, bool(success), False, {"succeed": bool(success)}

    def _compute_observation(self) -> dict:
        tcp_pos  = self._data.sensordata[:3].astype(np.float32)
        peg_pos  = self._data.xpos[self._peg_body_id].astype(np.float32)
        hole_pos = self._hole_pos.astype(np.float32)
        env_state = np.concatenate([tcp_pos, peg_pos, hole_pos, np.zeros(34, dtype=np.float32)])

        if self.image_obs:
            front_view, wrist_view = self.render()

            self.crop = [45, 45, 50, 50]
            if self.crop is not None:
                x, y, cw, ch = self.crop
                def safe_crop(img):
                    H, W = img.shape[:2]
                    x0, y0 = max(0, int(x)), max(0, int(y))
                    x1, y1 = min(W, x0 + int(cw)), min(H, y0 + int(ch))
                    return img[y0:y1, x0:x1] if x1 > x0 and y1 > y0 else img
                front_view = safe_crop(front_view)

            out_sz = (self._render_specs.height, self._render_specs.height)
            if front_view.shape[:2] != (out_sz[1], out_sz[0]):
                front_view = cv2.resize(front_view, out_sz)
            if wrist_view.shape[:2] != (out_sz[1], out_sz[0]):
                wrist_view = cv2.resize(wrist_view, out_sz)

            cv2.imshow("Front View", cv2.cvtColor(front_view.copy(), cv2.COLOR_RGB2BGR)); cv2.waitKey(1)
            cv2.imshow("Wrist View", cv2.cvtColor(wrist_view.copy(), cv2.COLOR_RGB2BGR)); cv2.waitKey(1)

            return {"pixels": {"front": front_view, "wrist": wrist_view}, "agent_pos": tcp_pos}
        else:
            return {"agent_pos": tcp_pos, "environment_state": env_state}

    def _compute_reward(self) -> float:
        tcp_pos = self._data.sensordata[:3]
        peg_pos = self._data.xpos[self._peg_body_id]

        dist_to_peg    = np.linalg.norm(tcp_pos - peg_pos)
        r_reach        = 1 - np.tanh(4.0 * dist_to_peg)
        is_grasped     = (peg_pos[2] - 0.02) > 0.05
        peg_to_hole_yz = np.linalg.norm((peg_pos - self._hole_pos)[1:])
        pre_ins        = 3 * (1 - np.tanh(0.5 * peg_to_hole_yz + 4.5 * peg_to_hole_yz)) * is_grasped
        insertion_dist = np.linalg.norm(peg_pos - self._hole_pos)
        inserted       = insertion_dist < 0.15 and is_grasped
        ins_rew        = 5 * (1 - np.tanh(5.0 * insertion_dist)) * inserted
        reward         = r_reach + float(is_grasped) + pre_ins + ins_rew

        if self.reward_type == "sparse":
            self.reward = 1.0 if inserted else 0.0
            reward = self.reward

        return float(reward)

    def _is_success(self) -> bool:
        return self.reward == 1.0