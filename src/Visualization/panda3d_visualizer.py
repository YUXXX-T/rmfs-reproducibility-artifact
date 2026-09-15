"""
Panda3D Warehouse Visualizer
==============================
Real-time warehouse visualization using Panda3D.

Supports two camera modes controlled by ``view_mode``:

* ``"2d"`` — orthographic top-down.  Flat quads, no depth.
* ``"3d"`` — perspective camera at an isometric angle.
  Obstacles and stations are raised boxes; pods and robots
  are taller quads.  Mouse-orbit is enabled so the user can
  rotate, pan and zoom.

The window runs in the same process — the simulation engine
calls ``render()`` each tick, which updates node positions
and pumps the Panda3D event loop once.

Panda3D 仓库可视化工具
==============================
基于 Panda3D 实现的实时仓库可视化。

支持两种摄像机模式，由参数 ``view_mode`` 控制：

* ``"2d"`` —— 正交俯视模式。物体显示为扁平的四边形，无深度感。
* ``"3d"`` —— 采用等轴测视角的透视摄像机模式。 
障碍物和工作站显示为立体方块；货架（Pods）和机器人则显示为较高的四边形。 
该模式启用了鼠标环绕控制，用户可以通过鼠标进行旋转、平移和缩放操作。

可视化窗口与仿真引擎运行在同一进程中 ——
仿真引擎在每个时间步（tick）调用一次 ``render()`` 方法，
该方法负责更新场景节点的位置，并驱动 Panda3D 的事件循环执行一次。

"""

import math
import os
from typing import TYPE_CHECKING

from direct.showbase.ShowBase import ShowBase
from panda3d.core import (
    OrthographicLens,
    PerspectiveLens,
    CardMaker,
    TextNode,
    LVecBase4f,
    LMatrix4f,
    TransparencyAttrib,
    AntialiasAttrib,
    WindowProperties,
    NodePath,
    LineSegs,
    LPoint3f,
    RigidBodyCombiner,
    Plane,
    Point3,
    Vec3,
    Point2,
)

if TYPE_CHECKING:
    from WorldState.world import WorldState

from Visualization.visualizer import BaseVisualizer


# ─── colour palettes ────────────────────────────────────────────────────────
_DARK_PALETTE = {
    "bg":       LVecBase4f(0.08, 0.08, 0.14, 1),
    "free":     LVecBase4f(0.12, 0.12, 0.20, 1),
    "obstacle": LVecBase4f(0.28, 0.28, 0.34, 1),
    "station":  LVecBase4f(0.85, 0.22, 0.28, 1),
    "pod_home": LVecBase4f(0.12, 0.22, 0.26, 1),
    "pod":      LVecBase4f(0.18, 0.77, 0.71, 1),
    "grid":     LVecBase4f(0.20, 0.20, 0.33, 0.6),
    "text":     LVecBase4f(0.90, 0.90, 0.90, 1),
    "gizmo_bg": LVecBase4f(0.06, 0.06, 0.10, 1),
    "grid_line": (0.25, 0.25, 0.40, 0.4),
    "zone_queue":   LVecBase4f(0.55, 0.35, 0.12, 1),
    "zone_buffer":  LVecBase4f(0.45, 0.25, 0.50, 1),
    "zone_service": LVecBase4f(0.70, 0.18, 0.22, 1),
    "zone_exit":    LVecBase4f(0.20, 0.55, 0.35, 1),
    "zone_entry":   LVecBase4f(0.18, 0.35, 0.60, 1),
}

_LIGHT_PALETTE = {
    "bg":       LVecBase4f(1.0, 1.0, 1.0, 1),
    "free":     LVecBase4f(0.92, 0.92, 0.95, 1),
    "obstacle": LVecBase4f(0.55, 0.55, 0.60, 1),
    "station":  LVecBase4f(0.90, 0.25, 0.30, 1),
    "pod_home": LVecBase4f(0.82, 0.90, 0.92, 1),
    "pod":      LVecBase4f(0.10, 0.62, 0.56, 1),
    "grid":     LVecBase4f(0.70, 0.70, 0.78, 0.6),
    "text":     LVecBase4f(0.15, 0.15, 0.15, 1),
    "gizmo_bg": LVecBase4f(0.88, 0.88, 0.92, 1),
    "grid_line": (0.60, 0.60, 0.70, 0.5),
    "zone_queue":   LVecBase4f(0.85, 0.65, 0.30, 1),
    "zone_buffer":  LVecBase4f(0.72, 0.45, 0.75, 1),
    "zone_service": LVecBase4f(0.90, 0.30, 0.35, 1),
    "zone_exit":    LVecBase4f(0.30, 0.75, 0.45, 1),
    "zone_entry":   LVecBase4f(0.30, 0.55, 0.85, 1),
}

# Robot colours (tab10-like) — shared by both themes
_ROBOT_COLOURS = [
    LVecBase4f(0.12, 0.47, 0.71, 1),   # blue
    LVecBase4f(1.00, 0.50, 0.05, 1),   # orange
    LVecBase4f(0.17, 0.63, 0.17, 1),   # green
    LVecBase4f(0.84, 0.15, 0.16, 1),   # red
    LVecBase4f(0.58, 0.40, 0.74, 1),   # purple
    LVecBase4f(0.55, 0.34, 0.29, 1),   # brown
    LVecBase4f(0.89, 0.47, 0.76, 1),   # pink
    LVecBase4f(0.50, 0.50, 0.50, 1),   # grey
    LVecBase4f(0.74, 0.74, 0.13, 1),   # olive
    LVecBase4f(0.09, 0.75, 0.81, 1),   # cyan
]

_POD_TYPE_COLOURS = {
    "A": LVecBase4f(0.90, 0.30, 0.30, 1),   # red
    "B": LVecBase4f(0.30, 0.70, 0.90, 1),   # blue
    "C": LVecBase4f(0.30, 0.85, 0.40, 1),   # green
}

CELL = 1.0        # world-space size of one grid cell
PAD  = 0.02       # gap between cell quads


def _make_box(name: str, sx: float, sy: float, sz: float) -> NodePath:
    """Build a simple 6-face box centred at origin.

    sx = size along X, sy = size along Y, sz = size along Z (up).
    """
    root = NodePath(name)
    root.setTwoSided(True)  # render both sides so the box looks solid
    hx, hy, hz = sx / 2, sy / 2, sz / 2

    # top (+Z) — card lies flat facing up
    cm_top = CardMaker(name + "_top")
    cm_top.setFrame(-hx, hx, -hy, hy)
    top = root.attachNewNode(cm_top.generate())
    top.setP(-90)
    top.setPos(0, 0, hz)

    # bottom (-Z)
    bot = root.attachNewNode(cm_top.generate())
    bot.setP(90)
    bot.setPos(0, 0, -hz)

    # front (-Y) — card faces camera
    cm_fwd = CardMaker(name + "_fwd")
    cm_fwd.setFrame(-hx, hx, -hz, hz)
    front = root.attachNewNode(cm_fwd.generate())
    front.setPos(0, -hy, 0)

    # back (+Y)
    back = root.attachNewNode(cm_fwd.generate())
    back.setH(180)
    back.setPos(0, hy, 0)

    # left (-X)
    cm_side = CardMaker(name + "_side")
    cm_side.setFrame(-hy, hy, -hz, hz)
    left = root.attachNewNode(cm_side.generate())
    left.setH(90)
    left.setPos(-hx, 0, 0)

    # right (+X)
    right = root.attachNewNode(cm_side.generate())
    right.setH(-90)
    right.setPos(hx, 0, 0)

    return root


_WHEEL_CLR = LVecBase4f(0.25, 0.25, 0.28, 1)
_JACKAL_BASE_CLR = LVecBase4f(0.2, 0.2, 0.2, 1)


def _make_robot(name: str, base_model=None, fenders_model=None,
                wheel_model=None, rm_cfg=None) -> NodePath:
    """Build a robot model: Jackal STL body if available, else box fallback."""
    root = NodePath(name)

    if base_model is not None and fenders_model is not None:
        offset_z = rm_cfg.body_offset_z if rm_cfg else 0.008
        hpr = rm_cfg.body_hpr if rm_cfg else (90, 90, 90)
        wheel_r = rm_cfg.wheel_radius if rm_cfg else 0.08
        wheel_pos = rm_cfg.wheel_positions if rm_cfg else [
            (0.18, 0.14, 0.08), (0.18, -0.14, 0.08),
            (-0.18, 0.14, 0.08), (-0.18, -0.14, 0.08)]

        base_pivot = root.attachNewNode(name + "_base")
        base_pivot.setPos(0, 0, offset_z)
        base_pivot.setHpr(*hpr)
        b = base_model.copyTo(base_pivot)
        b.setColor(_JACKAL_BASE_CLR)

        fenders_pivot = root.attachNewNode(name + "_body")
        fenders_pivot.setPos(0, 0, offset_z)
        fenders_pivot.setHpr(*hpr)
        fenders_model.copyTo(fenders_pivot)

        for i, (wx, wy, wz) in enumerate(wheel_pos):
            if wheel_model is not None:
                pivot = root.attachNewNode(f"{name}_wp{i}")
                pivot.setPos(wx, wy, wz)
                pivot.setScale(wheel_r / 217.0)
                pivot.setR(90)
                w = wheel_model.copyTo(pivot)
                w.setPos(0, 0, -80.5)
            else:
                w = _make_box(f"{name}_w{i}", 0.04, 0.04, 0.02)
                w.reparentTo(root)
                w.setPos(wx, wy, wz)
                w.setColor(_WHEEL_CLR)
    else:
        body_w = CELL * 0.50
        body_d = CELL * 0.50
        body_h = CELL * 0.30
        wheel_r = CELL * 0.07
        body = _make_box(name + "_body", body_w, body_d, body_h)
        body.reparentTo(root)
        body.setPos(0, 0, wheel_r + body_h / 2)
        wx = body_w * 0.42
        wy = body_d * 0.42
        for i, (dx, dy) in enumerate([(-wx, -wy), (wx, -wy), (-wx, wy), (wx, wy)]):
            if wheel_model is not None:
                pivot = root.attachNewNode(f"{name}_wp{i}")
                pivot.setScale(wheel_r / 217.0)
                pivot.setR(90)
                pivot.setPos(dx, dy, wheel_r)
                w = wheel_model.copyTo(pivot)
                w.setPos(0, 0, -80.5)
            else:
                w = _make_box(f"{name}_w{i}", 0.05, 0.05, 0.03)
                w.reparentTo(root)
                w.setPos(dx, dy, wheel_r)
                w.setColor(_WHEEL_CLR)

    return root


class Panda3DVisualizer(BaseVisualizer):
    """
    Warehouse visualizer with switchable 2D / 3D camera.
    支持 2D / 3D 视角切换的仓库可视化工具

    参数
    ----------
    view_mode : str
        ``"2d"`` for orthographic top-down (default),
        ``"3d"`` for isometric perspective with mouse orbit.
    """

    def __init__(self, view_mode: str = "2d", use_gpu: bool = False,
                 night_mode: bool = True, robot_label_scale: float = 0.25,
                 robot_model_cfg=None):
        self._view_mode = view_mode.lower()
        self._use_gpu = use_gpu
        self._night_mode = night_mode
        self._robot_label_scale = robot_label_scale
        self._robot_model_cfg = robot_model_cfg
        self._pal = _DARK_PALETTE if night_mode else _LIGHT_PALETTE
        self._app: ShowBase | None = None
        self._initialised = False

        # Scene-graph references kept across frames
        self._pod_nodes: dict[int, NodePath] = {}
        self._pod_labels: dict[int, NodePath] = {}
        self._agent_nodes: dict[int, NodePath] = {}
        self._agent_labels: dict[int, NodePath] = {}
        self._glow_nodes: dict[int, NodePath] = {}
        self._station_labels: dict[int, TextNode] = {}
        self._station_label_nps: dict[int, NodePath] = {}
        self._hud_text: TextNode | None = None
        self._hud_np: NodePath | None = None
        self._parent_window_handle: int | None = None

        # Toggle states
        self._show_pod_labels: bool = False
        self._color_by_type: bool = False

        # Robot orientation tracking
        self._agent_prev_pos: dict[int, tuple] = {}
        self._agent_headings: dict[int, float] = {}

        # Selection state
        self._selected_agent_id: int | None = None
        self._selected_pod_id: int | None = None
        self._world_state_ref: "WorldState | None" = None
        self._click_start_pos: tuple[float, float] | None = None

    # ── 公共 API ────────────────────────────────────────────────────

    def render(self, world_state: "WorldState"):
        """每个 tick 由仿真引擎调用一次。"""
        self._world_state_ref = world_state
        if not self._initialised:
            self._setup(world_state)

        self._update_pods(world_state)
        self._update_agents(world_state)
        self._update_station_labels(world_state)
        self._update_hud(world_state)

        # Pump one frame so the window stays responsive
        self._app.taskMgr.step()

    # ── initialisation ────────────────────────────────────────────────

    def _setup(self, world_state: "WorldState"):
        from WorldState.map_state import CellType

        ms = world_state.map_state
        rows, cols = ms.rows, ms.cols
        is_3d = self._view_mode == "3d"
        self._map_size = (rows, cols)

        # ---- ShowBase ----
        if self._parent_window_handle is not None:
            from panda3d.core import loadPrcFileData
            loadPrcFileData("", "window-type none")
            loadPrcFileData("", "win-foreground-window 0")

        self._app = ShowBase()

        wp = WindowProperties()
        mode_label = "3D" if is_3d else "2D"
        wp.setTitle(f"MAS-RMFS  —  Panda3D {mode_label}")
        if self._parent_window_handle is not None:
            wp.setParentWindow(self._parent_window_handle)
            wp.setOrigin(0, 0)
            # 使用 Qt 容器的实际尺寸
            init_sz = getattr(self, "_parent_initial_size", None)
            if init_sz and init_sz[0] > 0 and init_sz[1] > 0:
                wp.setSize(init_sz[0], init_sz[1])
            else:
                wp.setSize(1024, 768)
        else:
            wp.setSize(1024, 768) if is_3d else wp.setSize(900, 900)

        if self._parent_window_handle is not None:
            self._app.openMainWindow(props=wp)
        else:
            self._app.win.requestProperties(wp)

        # 在窗口存在之后设置背景颜色
        self._app.setBackgroundColor(self._pal["bg"])
        self._app.render.setAntialias(AntialiasAttrib.MMultisample)

        # 追踪窗口大小以修正宽高比
        self._win_aspect = (wp.getXSize() / wp.getYSize()
                            if wp.getYSize() > 0 else 1.0)
        self._app.accept("window-event", self._on_window_event)

        # 网格在世界坐标中的中心
        # 3D：X = 列, Y = -行（深度）, Z = 向上
        # 2D：X = 列, Y = 深度（相机轴）, Z = -行
        cx = (cols - 1) * CELL / 2
        cy_3d = -((rows - 1) * CELL / 2)   # centre Y for 3D
        cz_2d = -((rows - 1) * CELL / 2)   # centre Z for 2D

        if is_3d:
            # ---- 3D perspective with custom orbit camera ----
            self._app.disableMouse()

            lens = PerspectiveLens()
            lens.setFov(45)
            lens.setNearFar(0.5, 500)
            self._app.cam.node().setLens(lens)

            # 轨道相机状态（围绕枢轴的球坐标）
            self._cam_pivot = LPoint3f(cx, cy_3d, 0)
            self._cam_heading = -135.0  # degrees
            self._cam_pitch = 35.0      # degrees above horizon
            self._cam_dist = max(rows, cols) * CELL * 1.5
            self._mouse_prev = None
            self._update_orbit_camera()

            # 绑定鼠标事件用于旋转 / 平移 / 缩放
            self._app.accept("mouse1",    self._on_mouse_down, [1])
            self._app.accept("mouse1-up", self._on_mouse_up, [1])
            self._app.accept("mouse3",    self._on_mouse_down, [3])
            self._app.accept("mouse3-up", self._on_mouse_up, [3])
            self._app.accept("wheel_up",  self._on_zoom, [-1])
            self._app.accept("wheel_down", self._on_zoom, [1])
            self._mouse_btn = 0
            self._app.taskMgr.add(self._orbit_task, "orbit_camera")

            # 键盘视角预设
            self._app.accept("1", self._set_view_preset, ["top"])
            self._app.accept("2", self._set_view_preset, ["front"])
            self._app.accept("3", self._set_view_preset, ["right"])
            self._app.accept("4", self._set_view_preset, ["iso"])
            self._app.accept("r", self._set_view_preset, ["reset"])

            # 构建坐标轴指示器
            self._setup_axis_gizmo()
        else:
            # ---- 2D 正交 ----
            self._app.disableMouse()
            lens = OrthographicLens()
            half_w = cols * CELL / 2 + 1.0
            half_h = rows * CELL / 2 + 1.0
            self._ortho_film_w = half_w * 2
            self._ortho_film_h = half_h * 2
            lens.setFilmSize(self._ortho_film_w, self._ortho_film_h)
            lens.setNearFar(-100, 100)
            self._app.cam.node().setLens(lens)

            self._app.cam.setPos(cx, -10, cz_2d)
            self._app.cam.lookAt(cx, 0, cz_2d)

            # 2D 鼠标控制：滚轮 = 缩放，右键拖动 = 平移，左键点击 = 选取
            self._app.accept("wheel_up",   self._on_zoom_2d, [-1])
            self._app.accept("wheel_down",  self._on_zoom_2d, [1])
            self._app.accept("mouse1",      self._on_pick_click)
            self._app.accept("mouse3",      self._on_mouse_down, [3])
            self._app.accept("mouse3-up",   self._on_mouse_up, [3])
            self._mouse_btn = 0
            self._mouse_prev = None
            self._app.taskMgr.add(self._pan_task_2d, "pan_2d")

        # ---- 静态网格单元 ----
        if self._use_gpu:
            # 使用 RigidBodyCombiner 将静态地砖批量合并为一次绘制调用
            rbc = RigidBodyCombiner("static_grid")
            static_root = self._app.render.attachNewNode(rbc)
        else:
            static_root = self._app.render.attachNewNode("static_grid")

        if is_3d:
            self._build_grid_3d(static_root, ms, rows, cols)
        else:
            self._build_grid_2d(static_root, ms, rows, cols)

        # ---- Station labels (dynamic — updated each tick for Req 5) ----
        station_label_root = self._app.render.attachNewNode("station_labels")
        for sid, (sr, sc) in ms.station_positions.items():
            tn = TextNode(f"station_{sid}")
            tn.setText(f"S{sid}")
            tn.setTextColor(1, 0.42, 0.50, 1)
            tn.setAlign(TextNode.ACenter)
            tn.setCardColor(0, 0, 0, 0.5)
            tn.setCardAsMargin(0.05, 0.05, 0.05, 0.05)
            tn.setCardDecal(True)
            tnp = station_label_root.attachNewNode(tn)
            if is_3d:
                tnp.setPos(sc * CELL, -sr * CELL, 0.6)
                tnp.setScale(0.25)
                tnp.setBillboardPointEye()
            else:
                tnp.setPos(sc * CELL, -0.1, -sr * CELL + CELL * 0.35)
                tnp.setScale(0.25)
            self._station_labels[sid] = tn
            self._station_label_nps[sid] = tnp

        # ---- 网格线 (3D floor lines) ----
        if is_3d:
            self._draw_grid_lines(static_root, rows, cols)

        # GPU: batch static geometry
        if self._use_gpu:
            rbc.collect()
            static_root.flattenStrong()

        # ---- Pod nodes (dynamic) ----
        pod_root = self._app.render.attachNewNode("pods")
        # GPU: create one template and instance it
        if self._use_gpu and is_3d:
            pod_template = _make_box("pod_tpl", CELL * 0.6, CELL * 0.6, CELL * 0.35)
            pod_template.setColor(self._pal["pod"])
            pod_template.setTransparency(TransparencyAttrib.MAlpha)
            pod_template.flattenStrong()
        else:
            pod_template = None

        for pod in world_state.pod_state.pods.values():
            pr, pc = pod.current_position
            if is_3d:
                if pod_template is not None:
                    # Instance the template
                    np = pod_root.attachNewNode(f"pod_{pod.pod_id}")
                    pod_template.instanceTo(np)
                    np.setPos(pc * CELL, -pr * CELL, CELL * 0.175)
                else:
                    np = _make_box("pod", CELL * 0.6, CELL * 0.6, CELL * 0.35)
                    np.reparentTo(pod_root)
                    np.setPos(pc * CELL, -pr * CELL, CELL * 0.175)
                    np.setColor(self._pal["pod"])
                    np.setTransparency(TransparencyAttrib.MAlpha)
            else:
                pod_cm = CardMaker("pod")
                s = CELL * 0.35
                pod_cm.setFrame(-s, s, -s, s)
                np = pod_root.attachNewNode(pod_cm.generate())
                np.setPos(pc * CELL, -0.2, -pr * CELL)
                np.setColor(self._pal["pod"])
                np.setTransparency(TransparencyAttrib.MAlpha)
            self._pod_nodes[pod.pod_id] = np

            # Pod label (initially hidden)
            ptn = TextNode(f"pod_lbl_{pod.pod_id}")
            ptn.setText(str(pod.pod_id))
            ptn.setTextColor(1, 1, 1, 1)
            ptn.setAlign(TextNode.ACenter)
            ptn.setCardColor(0, 0, 0, 0.5)
            ptn.setCardAsMargin(0.02, 0.02, 0.02, 0.02)
            ptn.setCardDecal(True)
            plnp = pod_root.attachNewNode(ptn)
            if is_3d:
                plnp.setPos(pc * CELL, -pr * CELL, CELL * 0.45)
                plnp.setScale(0.18)
                plnp.setBillboardPointEye()
            else:
                plnp.setPos(pc * CELL, -0.3, -pr * CELL + CELL * 0.02)
                plnp.setScale(0.15)
            plnp.hide()
            self._pod_labels[pod.pod_id] = plnp

        # ---- Agent nodes (dynamic) ----
        agent_root = self._app.render.attachNewNode("agents")

        wheel_model = None
        base_model = None
        fenders_model = None
        rm_cfg = self._robot_model_cfg
        use_model = rm_cfg.use_model if rm_cfg else True
        if is_3d and use_model:
            from panda3d.core import Filename
            from Visualization.stl_loader import load_stl
            vis_dir = os.path.dirname(__file__)

            base_stl = os.path.join(vis_dir, "models", "jackal", "jackal-base.stl")
            fenders_stl = os.path.join(vis_dir, "models", "jackal", "jackal-fenders.stl")
            if os.path.isfile(base_stl) and os.path.isfile(fenders_stl):
                base_model = load_stl(base_stl, "jackal_base")
                fenders_model = load_stl(fenders_stl, "jackal_fenders")

            egg_path = os.path.join(
                vis_dir, "models", "car_wheel", "meshes", "car_wheel.egg",
            )
            tex_path = os.path.join(
                vis_dir, "models", "car_wheel",
                "materials", "textures", "car_wheel.png",
            )
            if os.path.isfile(egg_path):
                wheel_model = self._app.loader.loadModel(
                    Filename.fromOsSpecific(egg_path)
                )
                if os.path.isfile(tex_path):
                    tex = self._app.loader.loadTexture(
                        Filename.fromOsSpecific(tex_path)
                    )
                    wheel_model.setTexture(tex, 1)
                    wheel_model.setMaterialOff()

        for agent in world_state.agents:
            clr = _ROBOT_COLOURS[agent.agent_id % len(_ROBOT_COLOURS)]

            if is_3d:
                # Glow (larger translucent box)
                gnp = _make_box("glow", CELL * 0.75, CELL * 0.75, CELL * 0.55)
                gnp.reparentTo(agent_root)
                gnp.setColor(clr[0], clr[1], clr[2], 0.2)
                gnp.setTransparency(TransparencyAttrib.MAlpha)
                gnp.hide()
                self._glow_nodes[agent.agent_id] = gnp

                # Agent model (body + wheels)
                anp = _make_robot("agent", base_model=base_model,
                                  fenders_model=fenders_model,
                                  wheel_model=wheel_model,
                                  rm_cfg=rm_cfg)
                anp.reparentTo(agent_root)
                body_np = anp.find("**/agent_body")
                if body_np:
                    body_np.setColor(clr)
                self._agent_nodes[agent.agent_id] = anp

                # Billboard label
                tn = TextNode(f"agent_{agent.agent_id}")
                tn.setText(str(agent.agent_id))
                tn.setTextColor(1, 1, 1, 1)
                tn.setAlign(TextNode.ACenter)
                lnp = agent_root.attachNewNode(tn)
                lnp.setScale(self._robot_label_scale)
                lnp.setBillboardPointEye()
                self._agent_labels[agent.agent_id] = lnp
            else:
                # 2D — flat cards
                glow_cm = CardMaker("glow")
                g = CELL * 0.55
                glow_cm.setFrame(-g, g, -g, g)
                gnp = agent_root.attachNewNode(glow_cm.generate())
                gnp.setColor(clr[0], clr[1], clr[2], 0.25)
                gnp.setTransparency(TransparencyAttrib.MAlpha)
                gnp.hide()
                self._glow_nodes[agent.agent_id] = gnp

                agent_cm = CardMaker("agent")
                a = CELL * 0.4
                agent_cm.setFrame(-a, a, -a, a)
                anp = agent_root.attachNewNode(agent_cm.generate())
                anp.setColor(clr)
                self._agent_nodes[agent.agent_id] = anp

                tn = TextNode(f"agent_{agent.agent_id}")
                tn.setText(str(agent.agent_id))
                tn.setTextColor(1, 1, 1, 1)
                tn.setAlign(TextNode.ACenter)
                lnp = agent_root.attachNewNode(tn)
                lnp.setScale(self._robot_label_scale * 0.88)
                self._agent_labels[agent.agent_id] = lnp

        # ---- HUD text ----
        self._hud_text = TextNode("hud")
        self._hud_text.setTextColor(self._pal["text"])
        self._hud_text.setAlign(TextNode.ALeft)
        self._hud_text.setShadow(0.05, 0.05)
        self._hud_text.setShadowColor(0, 0, 0, 0.8)
        self._hud_np = self._app.aspect2d.attachNewNode(self._hud_text)
        self._hud_np.setScale(0.05)
        self._hud_np.setPos(-1.3, 0, 0.92)

        self._initialised = True
        self._is_3d = is_3d

    # ── grid builders ─────────────────────────────────────────────────

    def _build_grid_2d(self, parent, ms, rows, cols):
        from WorldState.map_state import CellType
        cm = CardMaker("cell")
        cm.setFrame(-CELL / 2 + PAD, CELL / 2 - PAD,
                     -CELL / 2 + PAD, CELL / 2 - PAD)
        for r in range(rows):
            for c in range(cols):
                cell = ms.grid[r][c]
                if cell == CellType.OBSTACLE:
                    clr = self._pal["obstacle"]
                elif cell == CellType.STATION:
                    clr = self._pal["station"]
                elif cell == CellType.POD_HOME:
                    clr = self._pal["pod_home"]
                elif cell == CellType.STATION_QUEUE:
                    clr = self._pal["zone_queue"]
                elif cell == CellType.STATION_BUFFER:
                    clr = self._pal["zone_buffer"]
                elif cell == CellType.STATION_SERVICE:
                    clr = self._pal["zone_service"]
                elif cell == CellType.STATION_EXIT:
                    clr = self._pal["zone_exit"]
                elif cell == CellType.STATION_ENTRY:
                    clr = self._pal["zone_entry"]
                else:
                    clr = self._pal["free"]
                np = parent.attachNewNode(cm.generate())
                np.setPos(c * CELL, 0, -r * CELL)
                np.setColor(clr)

    def _build_grid_3d(self, parent, ms, rows, cols):
        """3D grid: XY ground plane, Z = up."""
        from WorldState.map_state import CellType

        # Floor tiles — lie flat on XY plane (face +Z)
        floor_cm = CardMaker("floor")
        floor_cm.setFrame(-CELL / 2 + PAD, CELL / 2 - PAD,
                          -CELL / 2 + PAD, CELL / 2 - PAD)

        for r in range(rows):
            for c in range(cols):
                cell = ms.grid[r][c]
                x = c * CELL
                y = -r * CELL

                if cell == CellType.OBSTACLE:
                    # Raised box sitting on the floor
                    box = _make_box("obs", CELL - PAD * 2, CELL - PAD * 2, CELL * 0.5)
                    box.reparentTo(parent)
                    box.setPos(x, y, CELL * 0.25)
                    box.setColor(self._pal["obstacle"])
                else:
                    # Flat tile on the ground
                    np = parent.attachNewNode(floor_cm.generate())
                    np.setP(-90)   # rotate card to face +Z (up)
                    np.setPos(x, y, 0)
                    if cell == CellType.STATION:
                        np.setColor(self._pal["station"])
                    elif cell == CellType.POD_HOME:
                        np.setColor(self._pal["pod_home"])
                    elif cell == CellType.STATION_QUEUE:
                        np.setColor(self._pal["zone_queue"])
                    elif cell == CellType.STATION_BUFFER:
                        np.setColor(self._pal["zone_buffer"])
                    elif cell == CellType.STATION_SERVICE:
                        np.setColor(self._pal["zone_service"])
                    elif cell == CellType.STATION_EXIT:
                        np.setColor(self._pal["zone_exit"])
                    elif cell == CellType.STATION_ENTRY:
                        np.setColor(self._pal["zone_entry"])
                    else:
                        np.setColor(self._pal["free"])

    def _draw_grid_lines(self, parent, rows, cols):
        """Draw thin grid lines on the XY ground plane (Z=0.01)."""
        ls = LineSegs("grid_lines")
        ls.setColor(*self._pal["grid_line"])
        ls.setThickness(1.0)

        min_x = -CELL / 2
        max_x = (cols - 1) * CELL + CELL / 2
        min_y = -((rows - 1) * CELL + CELL / 2)
        max_y = CELL / 2
        z = 0.01  # slightly above floor to avoid z-fighting

        # Lines along X (one per row boundary)
        for r in range(rows + 1):
            y = CELL / 2 - r * CELL
            ls.moveTo(min_x, y, z)
            ls.drawTo(max_x, y, z)

        # Lines along Y (one per column boundary)
        for c in range(cols + 1):
            x = -CELL / 2 + c * CELL
            ls.moveTo(x, max_y, z)
            ls.drawTo(x, min_y, z)

        parent.attachNewNode(ls.create())

    # ── per-frame updates ────────────────────────────────────────────

    def _update_pods(self, world_state):
        for pod in world_state.pod_state.pods.values():
            np = self._pod_nodes.get(pod.pod_id)
            lnp = self._pod_labels.get(pod.pod_id)
            if np is None:
                continue
            if pod.is_carried:
                np.hide()
                if lnp:
                    lnp.hide()
            else:
                np.show()
                pr, pc = pod.current_position
                if self._is_3d:
                    np.setPos(pc * CELL, -pr * CELL, CELL * 0.175)
                    if lnp:
                        lnp.setPos(pc * CELL, -pr * CELL, CELL * 0.45)
                else:
                    np.setPos(pc * CELL, -0.2, -pr * CELL)
                    if lnp:
                        lnp.setPos(pc * CELL, -0.3, -pr * CELL + CELL * 0.02)

                if self._color_by_type:
                    clr = _POD_TYPE_COLOURS.get(pod.pod_type, self._pal["pod"])
                    np.setColor(clr)
                else:
                    np.setColor(self._pal["pod"])

                if lnp:
                    if self._show_pod_labels:
                        lnp.show()
                    else:
                        lnp.hide()

    def _update_agents(self, world_state):
        rm_cfg = self._robot_model_cfg
        use_model = rm_cfg.use_model if rm_cfg else False

        for agent in world_state.agents:
            aid = agent.agent_id
            r, c = agent.position
            x = c * CELL
            z = -r * CELL

            anp = self._agent_nodes[aid]
            gnp = self._glow_nodes[aid]
            lnp = self._agent_labels[aid]

            if self._is_3d:
                y = -r * CELL
                anp.setPos(x, y, 0)
                gnp.setPos(x, y, CELL * 0.275)
                lnp.setPos(x, y, CELL * 0.55)

                if use_model:
                    prev = self._agent_prev_pos.get(aid)
                    if prev is not None and prev != (r, c):
                        dr = r - prev[0]
                        dc = c - prev[1]
                        heading = math.degrees(math.atan2(dc, dr))
                        self._agent_headings[aid] = heading
                    if aid in self._agent_headings:
                        anp.setH(self._agent_headings[aid])
                self._agent_prev_pos[aid] = (r, c)
            else:
                anp.setPos(x, -0.5, z)
                gnp.setPos(x, -0.4, z)
                lnp.setPos(x, -0.6, z + CELL * 0.02)

            if agent.carried_pod_id is not None or aid == self._selected_agent_id:
                gnp.show()
            else:
                gnp.hide()

    def _update_hud(self, world_state):
        os = world_state.order_state
        completed = os.total_completed
        total = os.total_orders
        rate = completed / max(world_state.tick, 1)
        self._hud_text.setText(
            f"Tick: {world_state.tick}   "
            f"Orders: {completed}/{total}   "
            f"Rate: {rate:.2f}/tick"
        )
        if self._is_3d and not hasattr(self, "_help_np"):
            ht = TextNode("help")
            ht.setTextColor(0.6, 0.6, 0.7, 0.8)
            ht.setAlign(TextNode.ARight)
            ht.setShadow(0.04, 0.04)
            ht.setShadowColor(0, 0, 0, 0.6)
            ht.setText(
                "[1] Top  [2] Front  [3] Right  [4] Iso  [R] Reset\n"
                "Left-drag: Orbit  Right-drag: Pan  Scroll: Zoom"
            )
            self._help_np = self._app.aspect2d.attachNewNode(ht)
            self._help_np.setScale(0.04)
            self._help_np.setPos(1.3, 0, -0.92)

    def _update_station_labels(self, world_state):
        """Update station labels: show order ID when a robot is delivering."""
        from WorldState.task_state import TaskType, TaskStatus
        ms = world_state.map_state

        station_order_map: dict[int, int] = {}
        for agent in world_state.agents:
            active_task = world_state.task_state.get_active_task_for_agent(
                agent.agent_id
            )
            if (active_task is not None
                    and active_task.task_type == TaskType.DELIVER
                    and active_task.status == TaskStatus.IN_PROGRESS):
                for sid, pos in ms.station_positions.items():
                    if agent.position == pos:
                        station_order_map[sid] = active_task.order_id
                        break

        for sid, tn in self._station_labels.items():
            if sid in station_order_map:
                tn.setText(f"O{station_order_map[sid]}")
            else:
                tn.setText(f"S{sid}")

    # ── 窗口大小改变时的宽高比修正 ─────────────────────────

    def _on_window_event(self, window):
        """窗口大小改变时修正镜头宽高比。"""
        if window is None or window != self._app.win:
            return
        w = window.getXSize()
        h = window.getYSize()
        if h == 0:
            return
        aspect = w / h
        if abs(aspect - self._win_aspect) < 0.001:
            return  # 无明显变化
        self._win_aspect = aspect

        lens = self._app.cam.node().getLens()
        if self._is_3d:
            # 透视：直接设置宽高比
            lens.setAspectRatio(aspect)
        else:
            # 正交：调整胶片宽度同时保持高度不变
            # 使垂直范围不变，水平范围
            # 按比例扩展 → 无变形
            self._ortho_film_w = self._ortho_film_h * aspect
            lens.setFilmSize(self._ortho_film_w, self._ortho_film_h)

    # ── 自定义轨道相机（仅 3D） ────────────────────────────────

    def _update_orbit_camera(self):
        """从围绕枢轴的球坐标重新定位相机。"""
        h_rad = math.radians(self._cam_heading)
        p_rad = math.radians(self._cam_pitch)
        d = self._cam_dist

        # 球坐标 → 笛卡尔坐标（Panda3D：Y = 前方，Z = 向上）
        cos_p = math.cos(p_rad)
        cam_x = self._cam_pivot.x + d * cos_p * math.sin(h_rad)
        cam_y = self._cam_pivot.y - d * cos_p * math.cos(h_rad)
        cam_z = self._cam_pivot.z + d * math.sin(p_rad)

        self._app.camera.setPos(cam_x, cam_y, cam_z)
        self._app.camera.lookAt(self._cam_pivot)

    def _on_mouse_down(self, btn):
        self._mouse_btn = btn
        self._mouse_prev = None
        if btn == 1 and self._app.mouseWatcherNode.hasMouse():
            self._click_start_pos = (
                self._app.mouseWatcherNode.getMouseX(),
                self._app.mouseWatcherNode.getMouseY(),
            )

    def _on_mouse_up(self, btn):
        if self._mouse_btn == btn:
            if btn == 1 and self._is_3d and self._click_start_pos is not None:
                if self._app.mouseWatcherNode.hasMouse():
                    mx = self._app.mouseWatcherNode.getMouseX()
                    my = self._app.mouseWatcherNode.getMouseY()
                    dx = mx - self._click_start_pos[0]
                    dy = my - self._click_start_pos[1]
                    if (dx * dx + dy * dy) < 0.0004:
                        self._on_pick_click()
                self._click_start_pos = None
            self._mouse_btn = 0
            self._mouse_prev = None

    def _on_zoom(self, direction):
        """滚轮缩放（3D）：方向 -1 = 放大，+1 = 缩小。"""
        factor = 1.15
        if direction < 0:
            self._cam_dist /= factor
        else:
            self._cam_dist *= factor
        self._cam_dist = max(2.0, min(200.0, self._cam_dist))
        self._update_orbit_camera()

    def _on_zoom_2d(self, direction):
        """滚轮缩放（2D）：调整正交胶片大小实现等比缩放。"""
        factor = 1.12
        if direction < 0:
            self._ortho_film_w /= factor
            self._ortho_film_h /= factor
        else:
            self._ortho_film_w *= factor
            self._ortho_film_h *= factor
        # 限制在合理范围内
        self._ortho_film_w = max(2.0, min(200.0, self._ortho_film_w))
        self._ortho_film_h = max(2.0, min(200.0, self._ortho_film_h))
        lens = self._app.cam.node().getLens()
        lens.setFilmSize(self._ortho_film_w, self._ortho_film_h)

    def _pan_task_2d(self, task):
        """右键拖动平移 2D 正交视图。"""
        if not self._app.mouseWatcherNode.hasMouse():
            return task.cont
        mx = self._app.mouseWatcherNode.getMouseX()
        mz = self._app.mouseWatcherNode.getMouseY()
        if self._mouse_btn == 3:
            if self._mouse_prev is not None:
                dx = mx - self._mouse_prev[0]
                dz = mz - self._mouse_prev[1]
                # 根据当前胶片大小缩放平移速度以保持一致手感
                scale = self._ortho_film_w * 0.5
                pos = self._app.cam.getPos()
                self._app.cam.setPos(
                    pos.x - dx * scale,
                    pos.y,
                    pos.z - dz * scale,
                )
            self._mouse_prev = (mx, mz)
        return task.cont

    def _set_view_preset(self, name):
        """将相机切换到预设视角。"""
        presets = {
            "top":   (-90.0, 89.0),   # looking straight down
            "front": (-90.0, 0.1),    # looking from front (-Y)
            "right": (0.0,   0.1),    # looking from right (+X)
            "iso":   (-135.0, 35.0),  # isometric
            "reset": (-135.0, 35.0),  # same as iso + reset pivot
        }
        if name in presets:
            self._cam_heading, self._cam_pitch = presets[name]
            if name == "reset":
                ms = getattr(self, "_map_size", None)
                if ms:
                    self._cam_pivot = LPoint3f(
                        (ms[1] - 1) * CELL / 2,
                        -((ms[0] - 1) * CELL / 2),
                        0,
                    )
            self._update_orbit_camera()

    # ── axis gizmo ────────────────────────────────────────────────────

    def _setup_axis_gizmo(self):
        """
        Create a small 3D axis indicator in the lower-left corner.
        在左下角创建一个小型的三维坐标轴指示器。
        """
        # Separate scene for the gizmo
        self._gizmo_root = NodePath("gizmo_root")
        self._gizmo_pivot = self._gizmo_root.attachNewNode("gizmo_pivot")

        axis_len = 1.0
        axes = [
            ("X", (axis_len, 0, 0), (1.0, 0.3, 0.3, 1)),    # red
            ("Y", (0, axis_len, 0), (0.3, 1.0, 0.3, 1)),    # green
            ("Z", (0, 0, axis_len), (0.4, 0.5, 1.0, 1)),    # blue
        ]

        for label, end, colour in axes:
            # Axis line
            ls = LineSegs(f"axis_{label}")
            ls.setColor(*colour)
            ls.setThickness(2.5)
            ls.moveTo(0, 0, 0)
            ls.drawTo(*end)
            self._gizmo_pivot.attachNewNode(ls.create())

            # Label
            tn = TextNode(f"lbl_{label}")
            tn.setText(label)
            tn.setTextColor(*colour)
            tn.setAlign(TextNode.ACenter)
            tnp = self._gizmo_pivot.attachNewNode(tn)
            tnp.setPos(end[0] * 1.25, end[1] * 1.25, end[2] * 1.25)
            tnp.setScale(0.35)
            tnp.setBillboardPointEye()

        # Create a small DisplayRegion in the lower-left
        dr = self._app.win.makeDisplayRegion(0, 0.18, 0, 0.24)
        dr.setSort(20)
        dr.setClearColorActive(True)
        dr.setClearColor(self._pal["gizmo_bg"])
        dr.setClearDepthActive(True)

        # Gizmo camera — create manually (not via makeCamera)
        from panda3d.core import Camera
        lens = PerspectiveLens()
        lens.setFov(40)
        lens.setNearFar(0.1, 100)

        cam_node = Camera("gizmo_cam")
        cam_node.setLens(lens)
        gizmo_cam_np = self._gizmo_root.attachNewNode(cam_node)
        gizmo_cam_np.setPos(0, -5, 0)
        gizmo_cam_np.lookAt(0, 0, 0)
        dr.setCamera(gizmo_cam_np)

        self._gizmo_cam = gizmo_cam_np
        self._gizmo_dr = dr

    def _update_gizmo(self):
        """
        Sync gizmo rotation with the main camera orientation.
        将 Gizmo 的旋转与主摄像机的朝向同步。
        """
        if not hasattr(self, "_gizmo_pivot"):
            return
        # The gizmo pivot should rotate to match the main camera's view
        # We set the gizmo camera to the same orientation as the main camera
        # but at a fixed distance from origin
        # 辅助工具的枢轴应旋转，以与主摄像机的视角保持一致。
        # 我们将辅助工具摄像机的朝向设置为与主摄像机相同，
        # 但使其与原点保持固定的距离。
        h_rad = math.radians(self._cam_heading)
        p_rad = math.radians(self._cam_pitch)
        d = 5.0

        cos_p = math.cos(p_rad)
        cx = d * cos_p * math.sin(h_rad)
        cy = -d * cos_p * math.cos(h_rad)
        cz = d * math.sin(p_rad)

        self._gizmo_cam.setPos(cx, cy, cz)
        self._gizmo_cam.lookAt(0, 0, 0)

    def _orbit_task(self, task):
        """
        Per-frame task: reads mouse position delta for orbit/pan.
        逐帧任务：读取鼠标位置增量，用于轨道旋转/平移。
        """
        # Sync gizmo each frame
        self._update_gizmo()

        if not self._app.mouseWatcherNode.hasMouse():
            return task.cont

        mx = self._app.mouseWatcherNode.getMouseX()
        my = self._app.mouseWatcherNode.getMouseY()

        if self._mouse_btn == 0 or self._mouse_prev is None:
            self._mouse_prev = (mx, my)
            return task.cont

        dx = mx - self._mouse_prev[0]
        dy = my - self._mouse_prev[1]
        self._mouse_prev = (mx, my)

        if self._mouse_btn == 1:
            # Left-drag → orbit
            self._cam_heading += dx * 150
            self._cam_pitch += dy * 100
            self._cam_pitch = max(5.0, min(85.0, self._cam_pitch))
            self._update_orbit_camera()

        elif self._mouse_btn == 3:
            # Right-drag → pan (move pivot in camera-local plane)
            h_rad = math.radians(self._cam_heading)
            speed = self._cam_dist * 0.5

            # Camera-right direction (on the XY ground plane)
            rx = math.cos(h_rad)
            ry = math.sin(h_rad)

            self._cam_pivot.x -= dx * speed * rx
            self._cam_pivot.y -= dx * speed * ry
            self._cam_pivot.z += dy * speed
            self._update_orbit_camera()

        return task.cont

    # ── 实体选取（点击选中） ────────────────────────────────────────

    def _on_pick_click(self):
        """鼠标点击选取最近的机器人或 Pod。"""
        if not self._app.mouseWatcherNode.hasMouse():
            return

        mx = self._app.mouseWatcherNode.getMouseX()
        my = self._app.mouseWatcherNode.getMouseY()

        world_pos = self._mouse_to_world(mx, my)
        if world_pos is None:
            return

        selected = self._find_nearest_entity(world_pos)

        if selected is None:
            self._selected_agent_id = None
            self._selected_pod_id = None
        elif selected[0] == "agent":
            self._selected_agent_id = selected[1]
            self._selected_pod_id = None
        elif selected[0] == "pod":
            self._selected_pod_id = selected[1]
            self._selected_agent_id = None

    def _mouse_to_world(self, mx: float, my: float):
        """将归一化鼠标坐标投影到世界坐标系的地面平面上。"""
        lens = self._app.cam.node().getLens()
        cam_np = self._app.camera

        near_point = Point3()
        far_point = Point3()
        if not lens.extrude(Point2(mx, my), near_point, far_point):
            return None

        near_world = self._app.render.getRelativePoint(cam_np, near_point)
        far_world = self._app.render.getRelativePoint(cam_np, far_point)

        if self._is_3d:
            plane = Plane(Vec3(0, 0, 1), Point3(0, 0, 0))
        else:
            plane = Plane(Vec3(0, 1, 0), Point3(0, 0, 0))

        intersection = Point3()
        if not plane.intersectsLine(intersection, near_world, far_world):
            return None

        return intersection

    def _find_nearest_entity(self, world_pos):
        """查找离世界坐标点最近的机器人或 Pod。"""
        threshold = CELL * 0.5
        best_dist = threshold
        best = None

        ws = self._world_state_ref
        if ws is None:
            return None

        for agent in ws.agents:
            r, c = agent.position
            if self._is_3d:
                dx = world_pos.x - c * CELL
                dy = world_pos.y - (-r * CELL)
            else:
                dx = world_pos.x - c * CELL
                dy = world_pos.z - (-r * CELL)
            dist = (dx * dx + dy * dy) ** 0.5
            if dist < best_dist:
                best_dist = dist
                best = ("agent", agent.agent_id)

        for pod in ws.pod_state.pods.values():
            if pod.is_carried:
                continue
            pr, pc = pod.current_position
            if self._is_3d:
                dx = world_pos.x - pc * CELL
                dy = world_pos.y - (-pr * CELL)
            else:
                dx = world_pos.x - pc * CELL
                dy = world_pos.z - (-pr * CELL)
            dist = (dx * dx + dy * dy) ** 0.5
            if dist < best_dist:
                best_dist = dist
                best = ("pod", pod.pod_id)

        return best

    def toggle_pod_labels(self):
        self._show_pod_labels = not self._show_pod_labels

    def toggle_pod_type_colors(self):
        self._color_by_type = not self._color_by_type

    def get_selection_info(self) -> dict | None:
        """返回当前选中实体的信息，供 UI 显示。"""
        ws = self._world_state_ref
        if ws is None:
            return None

        if self._selected_agent_id is not None:
            agent = ws.get_agent(self._selected_agent_id)
            info = {
                "type": "agent",
                "agent_id": agent.agent_id,
                "position": agent.position,
                "status": agent.status.name,
                "carried_pod_id": agent.carried_pod_id,
            }
            if agent.assigned_task_id is not None:
                task = ws.task_state.tasks.get(agent.assigned_task_id)
                if task:
                    info["task_id"] = task.task_id
                    info["task_type"] = task.task_type.name
                    info["pod_id"] = task.pod_id
                    info["source"] = task.source
                    info["destination"] = task.destination
                    order = ws.order_state.orders.get(task.order_id)
                    if order:
                        info["order_id"] = order.order_id
                        info["sku_demands"] = order.sku_demands
                        info["order_status"] = order.status.name
            return info

        if self._selected_pod_id is not None:
            pod = ws.pod_state.get_pod(self._selected_pod_id)
            if pod:
                return {
                    "type": "pod",
                    "pod_id": pod.pod_id,
                    "pod_type": pod.pod_type,
                    "position": pod.current_position,
                    "home_position": pod.home_position,
                    "is_carried": pod.is_carried,
                    "carried_by": pod.carried_by,
                    "sku_inventory": dict(pod.sku_inventory),
                }

        return None
