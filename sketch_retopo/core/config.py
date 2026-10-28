# Port of ConfigTemp.hh / ConfigSaved.hh / ConfigRender.hh / RealParam.hh / AutoSave.hh
# with the default values from SketchRetopo_init.cc.

import time

DBL_MAX = 1.7976931348623157e308


class RealParam:
    """A real-valued parameter adjusted by dragging horizontally with the right mouse button."""

    def __init__(self, value=0.0, value_max=DBL_MAX, value_min=-DBL_MAX, is_relative=False, adjust_rate=1.0):
        self.value = value
        self.value_max = value_max
        self.value_min = value_min
        self.is_relative = is_relative
        self.adjust_rate = adjust_rate
        self.camera_distance = 1.0
        self.adjust_init_condition = None

    def __call__(self):
        return (self.camera_distance if self.is_relative else 1.0) * self.value

    def set_camera_distance(self, d):
        self.camera_distance = d

    def adjust_init(self, mouse_x):
        self.adjust_init_condition = (mouse_x, self.value)

    def adjust_move(self, mouse_x):
        dx = mouse_x - self.adjust_init_condition[0]
        self.value = self.adjust_init_condition[1] + self.adjust_rate * dx
        self.value = min(max(self.value, self.value_min), self.value_max)

    def adjust_done(self):
        self.adjust_init_condition = None

    def is_adjusting(self):
        return self.adjust_init_condition is not None


class AutoSave:
    def __init__(self):
        self.enabled = True
        self.filename = ''
        self.interval = 30
        self.unsaved = False
        self.last_time = time.time()


class ConfigTemp:
    default_quadSize_ratio = 0.03
    default_segmentSize_ratio = 0.002

    def __init__(self):
        self.snapSize = RealParam(0.02, DBL_MAX, 0, True, 0.001)
        self.brushSize_spine = RealParam(0.06, DBL_MAX, 0, True, 0.001)
        self.brushSize_autocmpl = RealParam(0.08, DBL_MAX, 0, True, 0.001)
        self.brushSize_moveVertex = RealParam(0.06, DBL_MAX, 0, True, 0.001)
        self.quadSize = RealParam(0, DBL_MAX, 0, False, 0)          # set when loading the base mesh
        self.segmentSize = RealParam(0, DBL_MAX, 0, False, 0)
        self.autoSave = AutoSave()
        self.autocmpl_preview_interval = 200      # milliseconds
        self.cylinder_num_div = 4
        self.loop_threshold_normal = 0.3
        self.loop_threshold_area = 0.3
        self.snakes_internal1 = 10000000.0
        self.snakes_internal2 = 10000000.0
        self.snakes_external = 1.0
        self.snakes_damping = 0.5
        # extra (not in SketchRetopo2, where Ctrl+Shift+drag smooths like Shift+drag)
        self.snakes_on_ctrl_shift = False


class ConfigSaved:
    def __init__(self):
        self.projOffset = 0.0
        self.symmetric = False


class ConfigRender:
    DEFAULT = 0
    QUADMESH_ONLY = 1
    BASEMESH_ONLY = 2

    def __init__(self):
        self.mode = ConfigRender.DEFAULT
        self.always_show_quads = True
        self.basemesh_render_line = False
        self.show_axis = False
        self.show_bbox = False
        self.use_ortho = False
        self.show_expmap = False
        self.overlay_alpha = 0.0
        self.overlay_v_flipped = True
        self.show_snake_energy = False
        self.snake_energy_scale = 10.0
        self.auto_camera_center = True
        # ConfigRender::Color (SketchRetopo2)
        self.color_background_bottom = (0.25, 0.25, 0.25)
        self.color_background_top = (0.0, 0.0, 0.0)
        self.color_active = (1.0, 0.2, 0.1)
        self.color_patch = (0.7, 0.7, 0.7)
        self.color_patch_alpha = 0.2
        self.corner_radius = 0.005
        self.edge_width_interior = 1.0
        self.edge_width_boundary = 2.0
        self.singularity_radius = 10.0
        self.patch_color_rgb = [
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
            (0.2, 0.1, 0.9),
            (0.5, 0.5, 0.7),
            (0.7, 0.7, 0.7),
            (0.7, 0.6, 0.5),
            (0.2, 0.8, 0.3),
        ]
        self.patch_color_alpha = 0.5
        self.light_pos = (50.0, 100.0, 200.0)
        self.quadmesh_only_show_auxiliary = False
        self.quadmesh_only_show_left_side = False
        self.turntable_mode = False
        self.turntable_speed = 0.02
        self.phong_tessellation = True
        self.phong_subdiv = 3
        self.phong_weight = 0.7
