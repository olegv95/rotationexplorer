"""
Rotation Explorer — PyQt6 desktop application.
Mirrors rotation_guide.html with 4 tabs:
  1. Rotation Matrix   2. Quaternions   3. Gimbal Lock   4. Reference

Convention:  X = nose (roll φ),  Y = right wing (pitch θ),  Z = down (yaw ψ).
ZYX aerospace Euler:  R = Rz(ψ) · Ry(θ) · Rx(φ).
"""

import sys
import math
import numpy as np
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QTabWidget,
    QVBoxLayout, QHBoxLayout, QGridLayout, QLabel,
    QSlider, QPushButton, QGroupBox, QScrollArea, QSplitter,
    QSizePolicy, QFrame,
)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QFont, QSurfaceFormat, QPalette, QColor

from OpenGL.GL import *
from OpenGL.GLU import *
from PyQt6.QtOpenGLWidgets import QOpenGLWidget

import rotation_explorer.dcm as DCM
import rotation_explorer.quaternion as Q
import rotation_explorer.kinematics as K

# ── Colour constants ──────────────────────────────────────────────────────────
BG        = "#0d0e12"
PANEL_BG  = "#13141a"
CARD_BG   = "#1a1c24"
BORDER    = "#2a2d3a"
TEXT      = "#c8ccd8"
DIM       = "#7a7f96"
YAW_C     = "#4f9cf9"
PITCH_C   = "#4fd9a0"
ROLL_C    = "#f97b4f"
ACCENT    = "#a78bfa"
DANGER    = "#f87171"
GOOD      = "#4fd9a0"

GL_BG     = (0.051, 0.055, 0.071)
GL_YAW    = (0.310, 0.612, 0.976)
GL_PITCH  = (0.310, 0.851, 0.627)
GL_ROLL   = (0.976, 0.482, 0.310)
GL_GRID   = (0.18, 0.20, 0.27)
GL_WHITE  = (0.78, 0.80, 0.85)

D2R = math.pi / 180


# ── Dark-theme stylesheet ─────────────────────────────────────────────────────
STYLE = f"""
QMainWindow, QWidget {{ background: {BG}; color: {TEXT}; font-family: 'Segoe UI', sans-serif; font-size: 13px; }}
QTabWidget::pane {{ border: 1px solid {BORDER}; background: {PANEL_BG}; }}
QTabBar::tab {{ background: {CARD_BG}; color: {DIM}; padding: 8px 18px; border: 1px solid {BORDER}; }}
QTabBar::tab:selected {{ background: {PANEL_BG}; color: {TEXT}; border-bottom: 2px solid {ACCENT}; }}
QGroupBox {{ border: 1px solid {BORDER}; border-radius: 6px; margin-top: 10px; padding: 8px;
             background: {CARD_BG}; color: {TEXT}; font-weight: bold; }}
QGroupBox::title {{ subcontrol-origin: margin; left: 8px; padding: 0 4px; }}
QSlider::groove:horizontal {{ height: 4px; background: {BORDER}; border-radius: 2px; }}
QSlider::handle:horizontal {{ background: {ACCENT}; width: 14px; height: 14px;
                               margin: -5px 0; border-radius: 7px; }}
QSlider::sub-page:horizontal {{ background: {ACCENT}; border-radius: 2px; }}
QPushButton {{ background: {CARD_BG}; color: {TEXT}; border: 1px solid {BORDER};
               border-radius: 5px; padding: 5px 12px; }}
QPushButton:hover {{ background: {BORDER}; }}
QPushButton:pressed {{ background: {ACCENT}; color: #fff; }}
QLabel {{ color: {TEXT}; }}
QScrollArea {{ border: none; background: transparent; }}
QScrollBar:vertical {{ background: {CARD_BG}; width: 8px; }}
QScrollBar::handle:vertical {{ background: {BORDER}; border-radius: 4px; }}
"""


# ─────────────────────────────────────────────────────────────────────────────
# 3-D Viewport widget
# ─────────────────────────────────────────────────────────────────────────────
class AircraftViewport(QOpenGLWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.roll  = 0.0   # φ  X-axis (nose)
        self.pitch = 0.0   # θ  Y-axis (right wing)
        self.yaw   = 0.0   # ψ  Z-axis (down)

        self._az   = 0.8   # camera azimuth (rad)
        self._el   = 1.05  # camera elevation from Z (rad)
        self._dist = 5.0   # camera distance
        self._last_pos = None
        self.setMinimumSize(320, 280)

    # ── public setter ─────────────────────────────────────────────────────────
    def set_angles(self, roll=None, pitch=None, yaw=None):
        if roll  is not None: self.roll  = roll
        if pitch is not None: self.pitch = pitch
        if yaw   is not None: self.yaw   = yaw
        self.update()

    # ── OpenGL callbacks ──────────────────────────────────────────────────────
    def initializeGL(self):
        glClearColor(*GL_BG, 1.0)
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_NORMALIZE)
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        glLightfv(GL_LIGHT0, GL_POSITION, [5.0, 8.0, 5.0, 1.0])
        glLightfv(GL_LIGHT0, GL_DIFFUSE,  [0.85, 0.85, 0.85, 1.0])
        glLightfv(GL_LIGHT0, GL_AMBIENT,  [0.30, 0.30, 0.35, 1.0])
        glEnable(GL_COLOR_MATERIAL)
        glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)

    def resizeGL(self, w, h):
        glViewport(0, 0, w, max(h, 1))
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(45.0, w / max(h, 1), 0.1, 100.0)
        glMatrixMode(GL_MODELVIEW)

    def paintGL(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()
        # Z-down spherical camera
        r, az, el = self._dist, self._az, self._el
        cx = r * math.sin(el) * math.cos(az)
        cy = r * math.sin(el) * math.sin(az)
        cz = -r * math.cos(el)
        gluLookAt(cx, cy, cz, 0, 0, 0, 0, 0, -1)

        self._draw_grid()
        self._draw_world_axes()

        # Apply ZYX rotation: Rz(ψ) · Ry(θ) · Rx(φ)
        glPushMatrix()
        glRotatef(self.yaw,   0, 0, 1)
        glRotatef(self.pitch, 0, 1, 0)
        glRotatef(self.roll,  1, 0, 0)
        self._draw_aircraft()
        glPopMatrix()

    # ── mouse drag to orbit ───────────────────────────────────────────────────
    def mousePressEvent(self, ev):
        self._last_pos = ev.position()

    def mouseMoveEvent(self, ev):
        if self._last_pos is None:
            return
        dx = ev.position().x() - self._last_pos.x()
        dy = ev.position().y() - self._last_pos.y()
        self._az  += dx * 0.008
        self._el   = max(0.05, min(math.pi - 0.05, self._el + dy * 0.008))
        self._last_pos = ev.position()
        self.update()

    def wheelEvent(self, ev):
        delta = ev.angleDelta().y()
        self._dist = max(1.5, min(20.0, self._dist - delta * 0.005))
        self.update()

    # ── scene helpers ─────────────────────────────────────────────────────────
    def _draw_grid(self):
        glDisable(GL_LIGHTING)
        glColor3f(*GL_GRID)
        glLineWidth(1.0)
        glBegin(GL_LINES)
        for i in range(-4, 5):
            glVertex3f(i, -4, 0); glVertex3f(i,  4, 0)
            glVertex3f(-4, i, 0); glVertex3f( 4, i, 0)
        glEnd()
        glEnable(GL_LIGHTING)

    def _draw_world_axes(self):
        glDisable(GL_LIGHTING)
        glLineWidth(2.0)
        glBegin(GL_LINES)
        glColor3f(0.6, 0.2, 0.2); glVertex3f(0,0,0); glVertex3f(1.5,0,0)
        glColor3f(0.2, 0.6, 0.2); glVertex3f(0,0,0); glVertex3f(0,1.5,0)
        glColor3f(0.2, 0.2, 0.7); glVertex3f(0,0,0); glVertex3f(0,0,1.5)
        glEnd()
        glEnable(GL_LIGHTING)

    def _draw_aircraft(self):
        quad = gluNewQuadric()
        gluQuadricNormals(quad, GLU_SMOOTH)

        # ── Fuselage (cylinder along X) ───────────────────────────────────────
        glColor3f(0.76, 0.78, 0.85)
        glPushMatrix()
        glTranslatef(-0.7, 0, 0)
        glRotatef(90, 0, 1, 0)          # default gluCylinder goes +Z → rotate to +X
        gluCylinder(quad, 0.12, 0.07, 1.4, 14, 1)
        glPopMatrix()

        # ── Nose cone ─────────────────────────────────────────────────────────
        glColor3f(0.62, 0.65, 0.75)
        glPushMatrix()
        glTranslatef(0.7, 0, 0)
        glRotatef(90, 0, 1, 0)
        gluCylinder(quad, 0.07, 0, 0.36, 14, 1)
        glPopMatrix()

        # ── Wings (XY plane, Z-thin box) ──────────────────────────────────────
        glColor3f(0.30, 0.40, 0.52)
        _solid_box(-0.28, 0.12, -0.75, 0.75, -0.013, 0.013)

        # ── Vertical stabilizer (-Z, aircraft "up") ───────────────────────────
        glColor3f(0.30, 0.40, 0.52)
        _solid_box(-0.73, -0.43, -0.013, 0.013, -0.38, 0.01)

        # ── Horizontal stabilizer ─────────────────────────────────────────────
        glColor3f(0.30, 0.40, 0.52)
        _solid_box(-0.695, -0.445, -0.26, 0.26, -0.013, 0.013)

        # ── Body-axis arrows ──────────────────────────────────────────────────
        glDisable(GL_LIGHTING)
        # X (nose) — roll — orange
        _draw_arrow((0,0,0), (1,0,0), 1.3, GL_ROLL)
        # Y (right wing) — pitch — green
        _draw_arrow((0,0,0), (0,1,0), 1.3, GL_PITCH)
        # Z (down) — yaw — blue
        _draw_arrow((0,0,0), (0,0,1), 1.3, GL_YAW)
        glEnable(GL_LIGHTING)

        gluDeleteQuadric(quad)


# ── OpenGL geometry helpers ────────────────────────────────────────────────────
def _solid_box(x0, x1, y0, y1, z0, z1):
    verts = [
        (x0,y0,z0),(x1,y0,z0),(x1,y1,z0),(x0,y1,z0),  # -Z face
        (x0,y0,z1),(x1,y0,z1),(x1,y1,z1),(x0,y1,z1),  # +Z face
        (x0,y0,z0),(x0,y0,z1),(x0,y1,z1),(x0,y1,z0),  # -X face
        (x1,y0,z0),(x1,y0,z1),(x1,y1,z1),(x1,y1,z0),  # +X face
        (x0,y0,z0),(x1,y0,z0),(x1,y0,z1),(x0,y0,z1),  # -Y face
        (x0,y1,z0),(x1,y1,z0),(x1,y1,z1),(x0,y1,z1),  # +Y face
    ]
    normals = [(0,0,-1),(0,0,1),(-1,0,0),(1,0,0),(0,-1,0),(0,1,0)]
    glBegin(GL_QUADS)
    for fi, n in enumerate(normals):
        glNormal3f(*n)
        for vi in range(4):
            glVertex3f(*verts[fi*4 + vi])
    glEnd()


def _draw_arrow(origin, direction, length, color):
    ox, oy, oz = origin
    dx, dy, dz = direction
    ex = ox + dx * length
    ey = oy + dy * length
    ez = oz + dz * length
    glColor3f(*color)
    glLineWidth(2.0)
    glBegin(GL_LINES)
    glVertex3f(ox, oy, oz)
    glVertex3f(ex, ey, ez)
    glEnd()
    # Simple cone tip (GL_TRIANGLE_FAN approximation)
    tip_len = 0.18
    tip_r   = 0.07
    tx, ty, tz = ex - dx * tip_len, ey - dy * tip_len, ez - dz * tip_len
    # build perpendicular vectors
    perp1 = np.cross([dx, dy, dz], [0, 0, 1] if abs(dz) < 0.9 else [1, 0, 0])
    perp1 = perp1 / (np.linalg.norm(perp1) + 1e-12) * tip_r
    perp2 = np.cross([dx, dy, dz], perp1)
    perp2 = perp2 / (np.linalg.norm(perp2) + 1e-12) * tip_r
    glBegin(GL_TRIANGLE_FAN)
    glVertex3f(ex, ey, ez)
    for i in range(9):
        a = 2 * math.pi * i / 8
        c, s = math.cos(a), math.sin(a)
        bx = tx + c * perp1[0] + s * perp2[0]
        by = ty + c * perp1[1] + s * perp2[1]
        bz = tz + c * perp1[2] + s * perp2[2]
        glVertex3f(bx, by, bz)
    glEnd()


# ─────────────────────────────────────────────────────────────────────────────
# Shared UI helpers
# ─────────────────────────────────────────────────────────────────────────────
def _label(text, color=TEXT, bold=False, size=13, align=Qt.AlignmentFlag.AlignLeft):
    lbl = QLabel(text)
    lbl.setAlignment(align)
    style = f"color:{color}; font-size:{size}px;"
    if bold:
        style += " font-weight:bold;"
    lbl.setStyleSheet(style)
    return lbl


def _mono(text="0.0000", color=TEXT, width=80):
    lbl = QLabel(text)
    lbl.setStyleSheet(f"color:{color}; font-family:monospace; font-size:13px;")
    lbl.setFixedWidth(width)
    lbl.setAlignment(Qt.AlignmentFlag.AlignRight)
    return lbl


def _card(title, widget):
    gb = QGroupBox(title)
    gb.setStyleSheet(f"QGroupBox{{background:{CARD_BG};border:1px solid {BORDER};border-radius:6px;margin-top:10px;padding:8px;color:{TEXT};font-weight:bold;}}")
    lay = QVBoxLayout(gb)
    lay.addWidget(widget)
    return gb


def _angle_slider(lo=-180, hi=180, init=0, color=ACCENT):
    sl = QSlider(Qt.Orientation.Horizontal)
    sl.setRange(lo * 10, hi * 10)
    sl.setValue(int(init * 10))
    sl.setSingleStep(1)
    sl.setStyleSheet(f"""
        QSlider::groove:horizontal {{height:4px; background:{BORDER}; border-radius:2px;}}
        QSlider::handle:horizontal {{background:{color}; width:14px; height:14px; margin:-5px 0; border-radius:7px;}}
        QSlider::sub-page:horizontal {{background:{color}; border-radius:2px;}}
    """)
    return sl


def _matrix_grid(rows=3, cols=3, cell_w=72):
    frame = QFrame()
    grid = QGridLayout(frame)
    grid.setSpacing(3)
    cells = []
    for r in range(rows):
        row_cells = []
        for c in range(cols):
            lbl = QLabel("0.000")
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            lbl.setFixedWidth(cell_w)
            lbl.setStyleSheet(f"color:{TEXT}; font-family:monospace; font-size:12px; background:{PANEL_BG}; border-radius:3px; padding:3px;")
            grid.addWidget(lbl, r, c)
            row_cells.append(lbl)
        cells.append(row_cells)
    return frame, cells


def _separator():
    line = QFrame()
    line.setFrameShape(QFrame.Shape.HLine)
    line.setStyleSheet(f"background:{BORDER};")
    line.setFixedHeight(1)
    return line


# ─────────────────────────────────────────────────────────────────────────────
# Tab 1 — Rotation Matrix
# ─────────────────────────────────────────────────────────────────────────────
class RotationMatrixTab(QWidget):
    def __init__(self):
        super().__init__()
        self._anim_step = 0
        self._animating = False

        self.viewport = AircraftViewport()

        # Sliders
        self.sl_yaw   = _angle_slider(-180, 180, 0, YAW_C)
        self.sl_pitch = _angle_slider(-90,  90,  0, PITCH_C)
        self.sl_roll  = _angle_slider(-180, 180, 0, ROLL_C)

        self.lbl_yaw   = _mono("0°", YAW_C)
        self.lbl_pitch = _mono("0°", PITCH_C)
        self.lbl_roll  = _mono("0°", ROLL_C)

        # DCM cells (decomposed)
        self.dcm_cells = {}   # 'z','y','x','full' -> [[QLabel]]
        ctrl = self._build_controls()

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self.viewport)
        splitter.addWidget(ctrl)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(6, 6, 6, 6)
        lay.addWidget(splitter)

        # Connections
        self.sl_yaw.valueChanged.connect(self._update)
        self.sl_pitch.valueChanged.connect(self._update)
        self.sl_roll.valueChanged.connect(self._update)
        self._update()

    def _build_controls(self):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        container = QWidget()
        vlay = QVBoxLayout(container)
        vlay.setSpacing(8)

        # ── Angle sliders ─────────────────────────────────────────────────────
        gb_ang = QGroupBox("Euler Angles (ZYX)")
        glay = QGridLayout(gb_ang)
        pairs = [
            ("Yaw ψ (Z)", self.sl_yaw,   self.lbl_yaw,   YAW_C),
            ("Pitch θ (Y)", self.sl_pitch, self.lbl_pitch, PITCH_C),
            ("Roll φ (X)",  self.sl_roll,  self.lbl_roll,  ROLL_C),
        ]
        for i, (name, sl, lbl, col) in enumerate(pairs):
            glay.addWidget(_label(name, col), i, 0)
            glay.addWidget(sl,  i, 1)
            glay.addWidget(lbl, i, 2)
        vlay.addWidget(gb_ang)

        # ── Reset / Animate ───────────────────────────────────────────────────
        btn_row = QHBoxLayout()
        btn_rst = QPushButton("Reset")
        btn_rst.clicked.connect(self._reset)
        btn_anim = QPushButton("Animate")
        btn_anim.setCheckable(True)
        btn_anim.clicked.connect(self._toggle_anim)
        self._btn_anim = btn_anim
        btn_row.addWidget(btn_rst)
        btn_row.addWidget(btn_anim)
        vlay.addLayout(btn_row)

        # ── Decomposed DCMs ───────────────────────────────────────────────────
        self.dcm_cells['z'], gz = _matrix_grid(); self.dcm_cells['zf'] = gz
        self.dcm_cells['y'], gy = _matrix_grid(); self.dcm_cells['yf'] = gy
        self.dcm_cells['x'], gx = _matrix_grid(); self.dcm_cells['xf'] = gx
        self.dcm_cells['full'], gf = _matrix_grid(); self.dcm_cells['ff'] = gf

        for title, widget, cells_key in [
            ("Rz(ψ) — Yaw",   self.dcm_cells['z'],    'zf'),
            ("Ry(θ) — Pitch", self.dcm_cells['y'],    'yf'),
            ("Rx(φ) — Roll",  self.dcm_cells['x'],    'xf'),
            ("R = Rz·Ry·Rx",  self.dcm_cells['full'], 'ff'),
        ]:
            vlay.addWidget(_card(title, self.dcm_cells[cells_key[:-1]]))

        # ── Angular rates ─────────────────────────────────────────────────────
        gb_rates = QGroupBox("Angular Rates  (p=0.1, q=0.2, r=0.3 rad/s)")
        rlay = QGridLayout(gb_rates)
        self.rate_labels = {}
        for i, name in enumerate(["φ̇ (roll/s)", "θ̇ (pitch/s)", "ψ̇ (yaw/s)"]):
            rlay.addWidget(_label(name), i, 0)
            lbl = _mono("—", TEXT, 100)
            rlay.addWidget(lbl, i, 1)
            self.rate_labels[name] = lbl
        vlay.addWidget(gb_rates)

        vlay.addStretch()
        scroll.setWidget(container)

        # Fix cell references
        _, self.dcm_cells['zf']   = _matrix_grid()
        _, self.dcm_cells['yf']   = _matrix_grid()
        _, self.dcm_cells['xf']   = _matrix_grid()
        _, self.dcm_cells['ff']   = _matrix_grid()

        return scroll

    def _build_controls(self):   # noqa: F811  (intentional re-def to fix cell refs)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        container = QWidget()
        vlay = QVBoxLayout(container)
        vlay.setSpacing(8)

        # Angle sliders group
        gb_ang = QGroupBox("Euler Angles (ZYX)")
        glay = QGridLayout(gb_ang)
        pairs = [
            ("Yaw ψ (Z)",   self.sl_yaw,   self.lbl_yaw,   YAW_C),
            ("Pitch θ (Y)", self.sl_pitch, self.lbl_pitch, PITCH_C),
            ("Roll φ (X)",  self.sl_roll,  self.lbl_roll,  ROLL_C),
        ]
        for i, (name, sl, lbl, col) in enumerate(pairs):
            glay.addWidget(_label(name, col), i, 0)
            glay.addWidget(sl,  i, 1)
            glay.addWidget(lbl, i, 2)
        vlay.addWidget(gb_ang)

        btn_row = QHBoxLayout()
        btn_rst = QPushButton("Reset")
        btn_rst.clicked.connect(self._reset)
        self._btn_anim = QPushButton("▶ Animate")
        self._btn_anim.setCheckable(True)
        self._btn_anim.clicked.connect(self._toggle_anim)
        btn_row.addWidget(btn_rst)
        btn_row.addWidget(self._btn_anim)
        vlay.addLayout(btn_row)

        # Decomposed matrix grids
        self._rz_frame, self._rz_cells = _matrix_grid()
        self._ry_frame, self._ry_cells = _matrix_grid()
        self._rx_frame, self._rx_cells = _matrix_grid()
        self._rf_frame, self._rf_cells = _matrix_grid()

        vlay.addWidget(_card("Rz(ψ) — Yaw",   self._rz_frame))
        vlay.addWidget(_card("Ry(θ) — Pitch",  self._ry_frame))
        vlay.addWidget(_card("Rx(φ) — Roll",   self._rx_frame))
        vlay.addWidget(_card("R = Rz·Ry·Rx",   self._rf_frame))

        # Det / orthogonal row
        det_row = QHBoxLayout()
        det_row.addWidget(_label("det(R):"))
        self._lbl_det = _mono("1.0000", GOOD, 90)
        det_row.addWidget(self._lbl_det)
        det_row.addSpacing(20)
        det_row.addWidget(_label("orthogonal:"))
        self._lbl_orth = _mono("✓", GOOD, 30)
        det_row.addWidget(self._lbl_orth)
        det_row.addStretch()
        vlay.addLayout(det_row)

        # Angular rates
        gb_rates = QGroupBox("Angular Rates  (p=0.1, q=0.2, r=0.3 rad/s)")
        rlay = QGridLayout(gb_rates)
        self._rate_lbls = {}
        for i, (name, col) in enumerate([("φ̇ deg/s", ROLL_C), ("θ̇ deg/s", PITCH_C), ("ψ̇ deg/s", YAW_C)]):
            rlay.addWidget(_label(name, col), i, 0)
            lbl = _mono("—", col, 110)
            rlay.addWidget(lbl, i, 1)
            self._rate_lbls[i] = lbl
        vlay.addWidget(gb_rates)

        vlay.addStretch()
        scroll.setWidget(container)
        return scroll

    def _update(self):
        yaw   = self.sl_yaw.value()   / 10.0
        pitch = self.sl_pitch.value() / 10.0
        roll  = self.sl_roll.value()  / 10.0

        self.lbl_yaw.setText(f"{yaw:+.1f}°")
        self.lbl_pitch.setText(f"{pitch:+.1f}°")
        self.lbl_roll.setText(f"{roll:+.1f}°")

        self.viewport.set_angles(roll=roll, pitch=pitch, yaw=yaw)

        cy, sy = math.cos(yaw*D2R),   math.sin(yaw*D2R)
        cp, sp = math.cos(pitch*D2R), math.sin(pitch*D2R)
        cr, sr = math.cos(roll*D2R),  math.sin(roll*D2R)

        Rz = [[cy,-sy,0],[sy,cy,0],[0,0,1]]
        Ry = [[cp,0,sp],[0,1,0],[-sp,0,cp]]
        Rx = [[1,0,0],[0,cr,-sr],[0,sr,cr]]
        R  = DCM.euler_to_dcm(yaw, pitch, roll)

        def fill(cells, mat):
            for r in range(3):
                for c in range(3):
                    v = mat[r][c]
                    cells[r][c].setText(f"{v:+.3f}")
                    hi = abs(v) > 0.98
                    cells[r][c].setStyleSheet(
                        f"color:{'#fff' if hi else TEXT}; font-family:monospace; font-size:12px;"
                        f" background:{'#1e3a2a' if hi else PANEL_BG}; border-radius:3px; padding:3px;"
                    )

        fill(self._rz_cells, Rz)
        fill(self._ry_cells, Ry)
        fill(self._rx_cells, Rx)
        fill(self._rf_cells, R)

        det = DCM.determinant(R)
        orth = abs(det - 1) < 0.005
        self._lbl_det.setText(f"{det:.4f}")
        self._lbl_orth.setText("✓" if orth else "✗")
        self._lbl_orth.setStyleSheet(f"color:{'#4fd9a0' if orth else DANGER}; font-family:monospace;")

        # Angular rates
        try:
            fd, td, pd = K.euler_rates(roll, pitch, 0.1, 0.2, 0.3)
            self._rate_lbls[0].setText(f"{fd:+.2f}")
            self._rate_lbls[1].setText(f"{td:+.2f}")
            self._rate_lbls[2].setText(f"{pd:+.2f}")
        except K.GimbalLockError:
            for lbl in self._rate_lbls.values():
                lbl.setText("∞ (lock)")

    def _reset(self):
        self.sl_yaw.setValue(0)
        self.sl_pitch.setValue(0)
        self.sl_roll.setValue(0)

    def _toggle_anim(self, checked):
        self._animating = checked
        self._btn_anim.setText("■ Stop" if checked else "▶ Animate")
        if checked:
            self._anim_step = 0
            self._timer = QTimer()
            self._timer.timeout.connect(self._anim_tick)
            self._timer.start(33)
        else:
            self._timer.stop()

    def _anim_tick(self):
        t = self._anim_step * 0.02
        self.sl_yaw.setValue(int(math.sin(t * 0.7) * 450))
        self.sl_pitch.setValue(int(math.sin(t * 0.5) * 300))
        self.sl_roll.setValue(int(math.sin(t * 1.1) * 600))
        self._anim_step += 1


# ─────────────────────────────────────────────────────────────────────────────
# Tab 2 — Quaternions
# ─────────────────────────────────────────────────────────────────────────────
class QuaternionTab(QWidget):
    def __init__(self):
        super().__init__()
        self.viewport = AircraftViewport()

        self.sl_yaw   = _angle_slider(-180, 180, 0, YAW_C)
        self.sl_pitch = _angle_slider(-90,  90,  0, PITCH_C)
        self.sl_roll  = _angle_slider(-180, 180, 0, ROLL_C)

        self.lbl_yaw   = _mono("0°", YAW_C)
        self.lbl_pitch = _mono("0°", PITCH_C)
        self.lbl_roll  = _mono("0°", ROLL_C)

        ctrl = self._build_controls()

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self.viewport)
        splitter.addWidget(ctrl)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(6, 6, 6, 6)
        lay.addWidget(splitter)

        self.sl_yaw.valueChanged.connect(self._update)
        self.sl_pitch.valueChanged.connect(self._update)
        self.sl_roll.valueChanged.connect(self._update)
        self._update()

    def _build_controls(self):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        container = QWidget()
        vlay = QVBoxLayout(container)
        vlay.setSpacing(8)

        gb_ang = QGroupBox("Euler Angles (ZYX)")
        glay = QGridLayout(gb_ang)
        for i, (name, sl, lbl, col) in enumerate([
            ("Yaw ψ (Z)",   self.sl_yaw,   self.lbl_yaw,   YAW_C),
            ("Pitch θ (Y)", self.sl_pitch, self.lbl_pitch, PITCH_C),
            ("Roll φ (X)",  self.sl_roll,  self.lbl_roll,  ROLL_C),
        ]):
            glay.addWidget(_label(name, col), i, 0)
            glay.addWidget(sl,  i, 1)
            glay.addWidget(lbl, i, 2)
        vlay.addWidget(gb_ang)

        btn_row = QHBoxLayout()
        btn_rst = QPushButton("Reset")
        btn_rst.clicked.connect(self._reset)
        btn_row.addWidget(btn_rst)
        btn_row.addStretch()
        vlay.addLayout(btn_row)

        # Final quaternion display
        gb_q = QGroupBox("Quaternion  q = q_yaw ⊗ q_pitch ⊗ q_roll")
        qlay = QGridLayout(gb_q)
        self._q_lbls = {}
        for i, (comp, col) in enumerate([("w", TEXT), ("x", ROLL_C), ("y", PITCH_C), ("z", YAW_C)]):
            qlay.addWidget(_label(f"q.{comp}"), i, 0)
            lbl = _mono("0.0000", col, 90)
            qlay.addWidget(lbl, i, 1)
            self._q_lbls[comp] = lbl
        vlay.addWidget(gb_q)

        # Axis-angle
        gb_aa = QGroupBox("Axis-Angle")
        alay = QGridLayout(gb_aa)
        self._aa_lbls = {}
        for i, name in enumerate(["axis x", "axis y", "axis z", "angle"]):
            alay.addWidget(_label(name), i, 0)
            lbl = _mono("—", TEXT, 90)
            alay.addWidget(lbl, i, 1)
            self._aa_lbls[name] = lbl
        vlay.addWidget(gb_aa)

        # Decomposed quaternions
        gb_dec = QGroupBox("Decomposed:  q_yaw ⊗ q_pitch ⊗ q_roll")
        dlay = QGridLayout(gb_dec)
        dlay.addWidget(_label("q_yaw",   YAW_C,   bold=True), 0, 0)
        dlay.addWidget(_label("q_pitch", PITCH_C, bold=True), 1, 0)
        dlay.addWidget(_label("q_roll",  ROLL_C,  bold=True), 2, 0)
        self._dec_lbls = {
            'yaw_w':  _mono("1.0000", YAW_C),   'yaw_z':  _mono("0.0000", YAW_C),
            'pitch_w':_mono("1.0000", PITCH_C), 'pitch_y':_mono("0.0000", PITCH_C),
            'roll_w': _mono("1.0000", ROLL_C),  'roll_x': _mono("0.0000", ROLL_C),
        }
        dlay.addWidget(self._dec_lbls['yaw_w'],   0, 1)
        dlay.addWidget(_label("+ 0·i + 0·j +", DIM), 0, 2)
        dlay.addWidget(self._dec_lbls['yaw_z'],   0, 3)
        dlay.addWidget(_label("k", DIM), 0, 4)
        dlay.addWidget(self._dec_lbls['pitch_w'], 1, 1)
        dlay.addWidget(_label("+ 0·i +", DIM), 1, 2)
        dlay.addWidget(self._dec_lbls['pitch_y'], 1, 3)
        dlay.addWidget(_label("·j + 0·k", DIM), 1, 4)
        dlay.addWidget(self._dec_lbls['roll_w'],  2, 1)
        dlay.addWidget(_label("+", DIM), 2, 2)
        dlay.addWidget(self._dec_lbls['roll_x'],  2, 3)
        dlay.addWidget(_label("·i + 0·j + 0·k", DIM), 2, 4)
        vlay.addWidget(gb_dec)

        # Hamilton product steps
        gb_hp = QGroupBox("Hamilton Product Steps")
        hlay = QVBoxLayout(gb_hp)
        hlay.addWidget(_label("q_yaw ⊗ q_pitch:", DIM))
        self._lbl_step1 = QLabel("—")
        self._lbl_step1.setStyleSheet(f"color:{TEXT}; font-family:monospace; font-size:12px;")
        self._lbl_step1.setWordWrap(True)
        hlay.addWidget(self._lbl_step1)
        hlay.addWidget(_label("⊗ q_roll:", DIM))
        self._lbl_step2 = QLabel("—")
        self._lbl_step2.setStyleSheet(f"color:{ACCENT}; font-family:monospace; font-size:12px;")
        self._lbl_step2.setWordWrap(True)
        hlay.addWidget(self._lbl_step2)
        vlay.addWidget(gb_hp)

        # DCM from quaternion
        self._qdcm_frame, self._qdcm_cells = _matrix_grid()
        vlay.addWidget(_card("DCM from q", self._qdcm_frame))
        det_row2 = QHBoxLayout()
        det_row2.addWidget(_label("det:"))
        self._lbl_qdet  = _mono("1.0000", GOOD, 90)
        det_row2.addWidget(self._lbl_qdet)
        det_row2.addSpacing(10)
        det_row2.addWidget(_label("orth:"))
        self._lbl_qorth = _mono("✓", GOOD, 30)
        det_row2.addWidget(self._lbl_qorth)
        det_row2.addStretch()
        vlay.addLayout(det_row2)

        vlay.addStretch()
        scroll.setWidget(container)
        return scroll

    def _q_from_angle(self, angle_deg, axis):
        h = angle_deg * D2R / 2
        w, s = math.cos(h), math.sin(h)
        if axis == 0: return [w, s, 0, 0]
        if axis == 1: return [w, 0, s, 0]
        return [w, 0, 0, s]

    @staticmethod
    def _hamilton(q1, q2):
        w1,x1,y1,z1 = q1; w2,x2,y2,z2 = q2
        return [w1*w2-x1*x2-y1*y2-z1*z2,
                w1*x2+x1*w2+y1*z2-z1*y2,
                w1*y2-x1*z2+y1*w2+z1*x2,
                w1*z2+x1*y2-y1*x2+z1*w2]

    @staticmethod
    def _quat_to_dcm(w,x,y,z):
        return [[1-2*(y*y+z*z), 2*(x*y-w*z),   2*(x*z+w*y)  ],
                [2*(x*y+w*z),   1-2*(x*x+z*z), 2*(y*z-w*x)  ],
                [2*(x*z-w*y),   2*(y*z+w*x),   1-2*(x*x+y*y)]]

    def _update(self):
        yaw   = self.sl_yaw.value()   / 10.0
        pitch = self.sl_pitch.value() / 10.0
        roll  = self.sl_roll.value()  / 10.0

        self.lbl_yaw.setText(f"{yaw:+.1f}°")
        self.lbl_pitch.setText(f"{pitch:+.1f}°")
        self.lbl_roll.setText(f"{roll:+.1f}°")

        self.viewport.set_angles(roll=roll, pitch=pitch, yaw=yaw)

        qy = self._q_from_angle(yaw,   2)
        qp = self._q_from_angle(pitch, 1)
        qr = self._q_from_angle(roll,  0)
        q12 = self._hamilton(qy, qp)
        qt  = self._hamilton(q12, qr)

        # Final q
        w,x,y,z = qt
        self._q_lbls['w'].setText(f"{w:+.4f}")
        self._q_lbls['x'].setText(f"{x:+.4f}")
        self._q_lbls['y'].setText(f"{y:+.4f}")
        self._q_lbls['z'].setText(f"{z:+.4f}")

        # Axis-angle
        q_np = np.array([w,x,y,z])
        axis, angle = Q.axis_angle(q_np)
        if axis is not None:
            self._aa_lbls['axis x'].setText(f"{axis[0]:+.3f}")
            self._aa_lbls['axis y'].setText(f"{axis[1]:+.3f}")
            self._aa_lbls['axis z'].setText(f"{axis[2]:+.3f}")
            self._aa_lbls['angle'].setText(f"{angle:.2f}°")
        else:
            for k in self._aa_lbls: self._aa_lbls[k].setText("—")

        # Decomposed
        self._dec_lbls['yaw_w'].setText(f"{qy[0]:+.4f}")
        self._dec_lbls['yaw_z'].setText(f"{qy[3]:+.4f}")
        self._dec_lbls['pitch_w'].setText(f"{qp[0]:+.4f}")
        self._dec_lbls['pitch_y'].setText(f"{qp[2]:+.4f}")
        self._dec_lbls['roll_w'].setText(f"{qr[0]:+.4f}")
        self._dec_lbls['roll_x'].setText(f"{qr[1]:+.4f}")

        def fmt_q(q):
            return "[" + ", ".join(f"{v:+.4f}" for v in q) + "]"
        self._lbl_step1.setText(fmt_q(q12))
        self._lbl_step2.setText(fmt_q(qt))

        # DCM from q
        dcm_mat = self._quat_to_dcm(w,x,y,z)
        for r in range(3):
            for c in range(3):
                v = dcm_mat[r][c]
                self._qdcm_cells[r][c].setText(f"{v:+.3f}")
                hi = abs(v) > 0.98
                self._qdcm_cells[r][c].setStyleSheet(
                    f"color:{'#fff' if hi else TEXT}; font-family:monospace; font-size:12px;"
                    f" background:{'#1e3a2a' if hi else PANEL_BG}; border-radius:3px; padding:3px;"
                )
        det = (dcm_mat[0][0]*(dcm_mat[1][1]*dcm_mat[2][2]-dcm_mat[2][1]*dcm_mat[1][2])
              -dcm_mat[0][1]*(dcm_mat[1][0]*dcm_mat[2][2]-dcm_mat[2][0]*dcm_mat[1][2])
              +dcm_mat[0][2]*(dcm_mat[1][0]*dcm_mat[2][1]-dcm_mat[2][0]*dcm_mat[1][1]))
        orth = abs(det-1) < 0.005
        self._lbl_qdet.setText(f"{det:.4f}")
        self._lbl_qorth.setText("✓" if orth else "✗")
        self._lbl_qorth.setStyleSheet(f"color:{'#4fd9a0' if orth else DANGER}; font-family:monospace;")

    def _reset(self):
        self.sl_yaw.setValue(0)
        self.sl_pitch.setValue(0)
        self.sl_roll.setValue(0)


# ─────────────────────────────────────────────────────────────────────────────
# Tab 3 — Gimbal Lock
# ─────────────────────────────────────────────────────────────────────────────
class GimbalLockTab(QWidget):
    DEMO_SEQS = {
        "Standard":    [(30, 0, 30, "Normal flight")],
        "Near-lock":   [(0, 85, 0, "≈Gimbal lock"), (0, 89, 45, "Lock region")],
        "Safe (quat)": [(45, 85, 0, "Quat avoids ∞")],
    }

    def __init__(self):
        super().__init__()
        self.viewport = AircraftViewport()

        self.sl_yaw   = _angle_slider(-180, 180, 0,  YAW_C)
        self.sl_pitch = _angle_slider(-90,  90,  0,  PITCH_C)
        self.sl_roll  = _angle_slider(-180, 180, 0,  ROLL_C)
        self.lbl_yaw   = _mono("0°", YAW_C)
        self.lbl_pitch = _mono("0°", PITCH_C)
        self.lbl_roll  = _mono("0°", ROLL_C)

        ctrl = self._build_controls()
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self.viewport)
        splitter.addWidget(ctrl)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(6, 6, 6, 6)
        lay.addWidget(splitter)

        self.sl_yaw.valueChanged.connect(self._update)
        self.sl_pitch.valueChanged.connect(self._update)
        self.sl_roll.valueChanged.connect(self._update)
        self._update()

    def _build_controls(self):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        container = QWidget()
        vlay = QVBoxLayout(container)
        vlay.setSpacing(8)

        gb_ang = QGroupBox("Euler Angles (ZYX)")
        glay = QGridLayout(gb_ang)
        for i, (name, sl, lbl, col) in enumerate([
            ("Yaw ψ (Z)",   self.sl_yaw,   self.lbl_yaw,   YAW_C),
            ("Pitch θ (Y)", self.sl_pitch, self.lbl_pitch, PITCH_C),
            ("Roll φ (X)",  self.sl_roll,  self.lbl_roll,  ROLL_C),
        ]):
            glay.addWidget(_label(name, col), i, 0)
            glay.addWidget(sl,  i, 1)
            glay.addWidget(lbl, i, 2)
        vlay.addWidget(gb_ang)

        btn_rst = QPushButton("Reset")
        btn_rst.clicked.connect(self._reset)
        vlay.addWidget(btn_rst)

        # Warning box
        self._warn_lbl = QLabel("  ✓ No gimbal lock")
        self._warn_lbl.setStyleSheet(
            f"background:#1a2e1a; color:{GOOD}; border:1px solid {GOOD}; border-radius:5px; padding:6px; font-weight:bold;"
        )
        vlay.addWidget(self._warn_lbl)

        # Proximity bar
        gb_prox = QGroupBox("Singularity Proximity  |θ| / 90°")
        pbar_container = QWidget()
        pbar_container.setFixedHeight(20)
        pbar_container.setStyleSheet(f"background:{BORDER}; border-radius:4px;")
        self._prox_bar = QLabel(pbar_container)
        self._prox_bar.setFixedHeight(20)
        self._prox_bar.setStyleSheet(f"background:{PITCH_C}; border-radius:4px;")
        self._prox_bar.setFixedWidth(1)
        pbar_layout = QVBoxLayout(gb_prox)
        pbar_layout.addWidget(pbar_container)
        self._pbar_container = pbar_container
        vlay.addWidget(gb_prox)

        # cos(θ) and ψ̇ display
        gb_math = QGroupBox("Rate Equation  ψ̇ = (q·sinφ + r·cosφ) / cos θ")
        mlay = QGridLayout(gb_math)
        self._math_lbls = {}
        for i, (name, col) in enumerate([
            ("cos θ", TEXT), ("ψ̇ (deg/s)", YAW_C), ("φ̇ (deg/s)", ROLL_C), ("θ̇ (deg/s)", PITCH_C)
        ]):
            mlay.addWidget(_label(name, col), i, 0)
            lbl = _mono("—", col, 110)
            mlay.addWidget(lbl, i, 1)
            self._math_lbls[name] = lbl
        vlay.addWidget(gb_math)

        # Lock matrix at θ = ±90°
        self._gb_lock = QGroupBox("At θ ≈ ±90° — Matrix Collapse")
        llay = QVBoxLayout(self._gb_lock)
        self._lock_frame, self._lock_cells = _matrix_grid()
        llay.addWidget(self._lock_frame)
        lbl_insight = QLabel("Only (ψ − φ) appears — yaw and roll collapse into a single angle.\nNo matter how you set ψ and φ separately, only their difference affects the orientation. One degree of freedom is permanently lost.")
        lbl_insight.setStyleSheet(f"color:{DIM}; font-size:11px;")
        lbl_insight.setWordWrap(True)
        llay.addWidget(lbl_insight)
        vlay.addWidget(self._gb_lock)

        # Reference table
        gb_tbl = QGroupBox("Reference: ψ̇ at Various Pitch Angles  (p=0.1, q=0.2, r=0.3 rad/s)")
        tlay = QGridLayout(gb_tbl)
        headers = ["θ (deg)", "cos θ", "ψ̇ (deg/s)", "Status"]
        for c, h in enumerate(headers):
            lbl = QLabel(h)
            lbl.setStyleSheet(f"color:{DIM}; font-size:11px; font-weight:bold;")
            tlay.addWidget(lbl, 0, c)
        for row, theta in enumerate([0, 30, 60, 80, 85, 89], 1):
            ct = math.cos(theta * D2R)
            tlay.addWidget(QLabel(f"{theta}°"), row, 0)
            tlay.addWidget(QLabel(f"{ct:.3f}"), row, 1)
            try:
                _, _, pd = K.euler_rates(0.0, theta, 0.1, 0.2, 0.3)
                tlay.addWidget(QLabel(f"{pd:.1f}"), row, 2)
                tlay.addWidget(QLabel("OK"), row, 3)
            except K.GimbalLockError:
                tlay.addWidget(QLabel("∞"), row, 2)
                w_warn = QLabel("LOCK")
                w_warn.setStyleSheet(f"color:{DANGER}; font-weight:bold;")
                tlay.addWidget(w_warn, row, 3)
        vlay.addWidget(gb_tbl)

        vlay.addStretch()
        scroll.setWidget(container)
        return scroll

    def _update(self):
        yaw   = self.sl_yaw.value()   / 10.0
        pitch = self.sl_pitch.value() / 10.0
        roll  = self.sl_roll.value()  / 10.0

        self.lbl_yaw.setText(f"{yaw:+.1f}°")
        self.lbl_pitch.setText(f"{pitch:+.1f}°")
        self.lbl_roll.setText(f"{roll:+.1f}°")

        self.viewport.set_angles(roll=roll, pitch=pitch, yaw=yaw)

        ct = math.cos(pitch * D2R)
        prox = min(1.0, abs(pitch) / 90.0)
        self._math_lbls["cos θ"].setText(f"{ct:.4f}")

        # Proximity bar
        bar_w = max(2, int(self._pbar_container.width() * prox))
        self._prox_bar.setFixedWidth(bar_w)
        danger = prox > 0.94
        color = DANGER if danger else (YAW_C if prox > 0.7 else PITCH_C)
        self._prox_bar.setStyleSheet(f"background:{color}; border-radius:4px;")

        try:
            fd, td, pd = K.euler_rates(roll, pitch, 0.1, 0.2, 0.3)
            self._math_lbls["ψ̇ (deg/s)"].setText(f"{pd:+.2f}")
            self._math_lbls["φ̇ (deg/s)"].setText(f"{fd:+.2f}")
            self._math_lbls["θ̇ (deg/s)"].setText(f"{td:+.2f}")
            self._warn_lbl.setText(f"  ✓ No gimbal lock  (proximity {prox:.0%})")
            self._warn_lbl.setStyleSheet(
                f"background:#1a2e1a; color:{GOOD}; border:1px solid {GOOD}; border-radius:5px; padding:6px; font-weight:bold;"
            )
        except K.GimbalLockError:
            self._math_lbls["ψ̇ (deg/s)"].setText("∞")
            self._math_lbls["φ̇ (deg/s)"].setText("∞")
            self._math_lbls["θ̇ (deg/s)"].setText("—")
            self._warn_lbl.setText("  ⚠  GIMBAL LOCK — θ ≈ ±90°, ψ̇ → ∞")
            self._warn_lbl.setStyleSheet(
                f"background:#2e1a1a; color:{DANGER}; border:1px solid {DANGER}; border-radius:5px; padding:6px; font-weight:bold;"
            )

        # Lock Matrix update
        near_lock = abs(abs(pitch) - 90.0) < 12.0
        self._gb_lock.setVisible(near_lock)
        if near_lock:
            diff = (yaw - roll) * D2R
            sd, cd = math.sin(diff), math.cos(diff)
            sign = 1.0 if pitch > 0 else -1.0
            vals = [
                [0.0, -sign*sd, sign*cd],
                [0.0, cd,       sign*sd],
                [-sign, 0.0,    0.0]
            ]
            for r in range(3):
                for c in range(3):
                    v = vals[r][c]
                    self._lock_cells[r][c].setText(f"{v:+.3f}")
                    if r == 2 and c == 0:
                        color = DANGER
                    elif abs(v) < 1e-6:
                        color = DIM
                    else:
                        color = YAW_C
                    self._lock_cells[r][c].setStyleSheet(
                        f"color:{color}; font-family:monospace; font-size:12px;"
                        f" background:{PANEL_BG}; border-radius:3px; padding:3px;"
                    )

    def _reset(self):
        self.sl_yaw.setValue(0)
        self.sl_pitch.setValue(0)
        self.sl_roll.setValue(0)


# ─────────────────────────────────────────────────────────────────────────────
# Tab 4 — Reference (rich-text, scrollable)
# ─────────────────────────────────────────────────────────────────────────────
REF_HTML = f"""
<style>
  body {{ background:{BG}; color:{TEXT}; font-family:'Segoe UI',sans-serif; font-size:13px; line-height:1.7; }}
  h2 {{ color:{ACCENT}; margin-top:24px; border-bottom:1px solid {BORDER}; padding-bottom:4px; }}
  h3 {{ color:{YAW_C}; margin-top:16px; }}
  code {{ font-family:monospace; background:{CARD_BG}; padding:2px 5px; border-radius:3px; color:{ROLL_C}; }}
  table {{ border-collapse:collapse; width:100%; margin:8px 0; }}
  th {{ background:{CARD_BG}; color:{DIM}; padding:6px 10px; text-align:left; font-size:12px; }}
  td {{ padding:5px 10px; border-bottom:1px solid {BORDER}; }}
  table.matrix {{ border-left:2px solid {DIM}; border-right:2px solid {DIM}; width:auto; margin:4px 0; }}
  table.matrix td {{ border:none; padding:4px 10px; text-align:center; font-family:monospace; color:{TEXT}; }}
  table.layout {{ border:none; width:auto; margin:0; }}
  table.layout td {{ border:none; padding:0 6px; vertical-align:middle; }}
  .callout {{ background:{CARD_BG}; border-left:3px solid {ACCENT}; padding:8px 12px; margin:10px 0; border-radius:0 5px 5px 0; }}
  .yaw  {{ color:{YAW_C}; font-weight:bold; }}
  .pit  {{ color:{PITCH_C}; font-weight:bold; }}
  .roll {{ color:{ROLL_C}; font-weight:bold; }}
  .dim  {{ color:{DIM}; }}
</style>

<h2>Direction Cosine Matrix (DCM)</h2>
<h3>Body-Frame Convention</h3>
<p>Aerospace NED body frame: <span class="roll">X = nose (roll φ)</span>,
<span class="pit">Y = right wing (pitch θ)</span>,
<span class="yaw">Z = down (yaw ψ)</span>.</p>

<h3>Elementary Rotations</h3>
<table>
<tr><th>Axis</th><th>Angle</th><th>Matrix</th></tr>
<tr><td class="yaw">Z (yaw ψ)</td><td>ψ</td>
    <td><table class="layout"><tr>
        <td><code>Rz =</code></td>
        <td><table class="matrix">
          <tr><td>cψ</td><td>-sψ</td><td>0</td></tr>
          <tr><td>sψ</td><td>cψ</td><td>0</td></tr>
          <tr><td>0</td><td>0</td><td>1</td></tr>
        </table></td>
    </tr></table></td></tr>
<tr><td class="pit">Y (pitch θ)</td><td>θ</td>
    <td><table class="layout"><tr>
        <td><code>Ry =</code></td>
        <td><table class="matrix">
          <tr><td>cθ</td><td>0</td><td>sθ</td></tr>
          <tr><td>0</td><td>1</td><td>0</td></tr>
          <tr><td>-sθ</td><td>0</td><td>cθ</td></tr>
        </table></td>
    </tr></table></td></tr>
<tr><td class="roll">X (roll φ)</td><td>φ</td>
    <td><table class="layout"><tr>
        <td><code>Rx =</code></td>
        <td><table class="matrix">
          <tr><td>1</td><td>0</td><td>0</td></tr>
          <tr><td>0</td><td>cφ</td><td>-sφ</td></tr>
          <tr><td>0</td><td>sφ</td><td>cφ</td></tr>
        </table></td>
    </tr></table></td></tr>
</table>

<h3>ZYX Composition</h3>
<p><code>R = Rz(ψ) · Ry(θ) · Rx(φ)</code> — rightmost applied first to vectors.</p>

<h3>Properties</h3>
<table>
<tr><th>Property</th><th>Value</th></tr>
<tr><td>det(R)</td><td>= +1</td></tr>
<tr><td>Inverse</td><td>R⁻¹ = Rᵀ (transpose)</td></tr>
<tr><td>Orthogonal</td><td>R·Rᵀ = I</td></tr>
</table>

<h3>DCM → Euler ZYX</h3>
<p>
<code>ψ = atan2(R[1,0], R[0,0])</code><br>
<code>θ = arcsin(-R[2,0])</code><br>
<code>φ = atan2(R[2,1], R[2,2])</code>
</p>

<h2>Quaternion Algebra</h2>
<h3>Representation</h3>
<p>A unit quaternion: <code>q = w + xi + yj + zk</code>,  with  <code>|q| = 1</code>.</p>

<h3>Elementary Rotations</h3>
<table>
<tr><th>Rotation</th><th>Quaternion</th></tr>
<tr><td class="roll">Roll φ (X)</td><td><code>q_roll  = [cos(φ/2), sin(φ/2), 0, 0]</code></td></tr>
<tr><td class="pit">Pitch θ (Y)</td><td><code>q_pitch = [cos(θ/2), 0, sin(θ/2), 0]</code></td></tr>
<tr><td class="yaw">Yaw ψ (Z)</td><td><code>q_yaw   = [cos(ψ/2), 0, 0, sin(ψ/2)]</code></td></tr>
</table>

<h3>Euler Angles → Quaternion (ZYX order)</h3>
<p><code>q_total = q_yaw ⊗ q_pitch ⊗ q_roll</code></p>
<div class="callout">
<b>Why this order?</b>  The rightmost factor is applied first.  Reading right-to-left:<br>
1. Roll about X &nbsp;&nbsp; 2. Pitch about Y &nbsp;&nbsp; 3. Yaw about Z<br>
This matches R = Rz · Ry · Rx: last-written matrix acts first on the vector.
</div>

<h3>Quaternion Multiplication (Hamilton Product)</h3>
<p>For <code>q₁ = [w₁,x₁,y₁,z₁]</code> and <code>q₂ = [w₂,x₂,y₂,z₂]</code>:</p>
<p>
<code>w = w₁w₂ − x₁x₂ − y₁y₂ − z₁z₂</code><br>
<code>x = w₁x₂ + x₁w₂ + y₁z₂ − z₁y₂</code><br>
<code>y = w₁y₂ − x₁z₂ + y₁w₂ + z₁x₂</code><br>
<code>z = w₁z₂ + x₁y₂ − y₁x₂ + z₁w₂</code>
</p>
<div class="callout" style="border-color:{YAW_C}">
<b>Golden Rule of Quaternions</b><br>
&nbsp;&nbsp;Associative: <b style="color:{GOOD}">YES</b> &nbsp; (q₁⊗q₂)⊗q₃ = q₁⊗(q₂⊗q₃)<br>
&nbsp;&nbsp;Commutative: <b style="color:{DANGER}">NO</b> &nbsp;&nbsp; q₁⊗q₂ ≠ q₂⊗q₁ in general
</div>

<h3>Vector Rotation with Quaternion</h3>
<p>To rotate vector <code>v</code>:<br>
<code>v' = q ⊗ [0, v] ⊗ q*</code><br>
where <code>q* = [w, −x, −y, −z]</code> is the conjugate.</p>

<h3>Quaternion ↔ DCM</h3>
<p><b>q → R:</b></p>
<table class="layout"><tr>
  <td><code>R =</code></td>
  <td><table class="matrix">
    <tr><td>1−2(y²+z²)</td><td>2(xy−wz)</td><td>2(xz+wy)</td></tr>
    <tr><td>2(xy+wz)</td><td>1−2(x²+z²)</td><td>2(yz−wx)</td></tr>
    <tr><td>2(xz−wy)</td><td>2(yz+wx)</td><td>1−2(x²+y²)</td></tr>
  </table></td>
</tr></table>
<p><b>R → q (Shepperd method):</b> branch on whichever of w,x,y,z has the largest magnitude
(determined from the trace and diagonal of R) to avoid division by near-zero.
Each branch gives all four components from the chosen safe denominator.</p>

<h2>Attitude Kinematics</h2>
<h3>Euler Rate Equation (ZYX)</h3>
<p>
<code>φ̇ = p + (q sinφ + r cosφ) tanθ</code><br>
<code>θ̇ = q cosφ − r sinφ</code><br>
<code>ψ̇ = (q sinφ + r cosφ) / cosθ</code>
</p>
<div class="callout" style="border-color:{DANGER}">
<b>Gimbal Lock:</b> when θ → ±90°, cosθ → 0 and ψ̇ → ∞.
The yaw and roll axes align — one degree of freedom is lost.
Quaternions have no such singularity.
</div>

<h3>DCM Propagation</h3>
<p><code>Ṙ = R · [ω×]</code> &nbsp; where <code>[ω×]</code> is the skew-symmetric matrix of body rates.</p>

<h3>Quaternion Propagation</h3>
<p><code>q̇ = ½ Ξ(q) · ω</code></p>
<table class="layout"><tr>
  <td><code>Ξ(q) = ½</code></td>
  <td><table class="matrix">
    <tr><td>-x</td><td>-y</td><td>-z</td></tr>
    <tr><td>w</td><td>-z</td><td>y</td></tr>
    <tr><td>z</td><td>w</td><td>-x</td></tr>
    <tr><td>-y</td><td>x</td><td>w</td></tr>
  </table></td>
</tr></table>

<h2>Method Comparison</h2>
<table>
<tr><th>Feature</th><th>Euler ZYX</th><th>DCM</th><th>Quaternion</th></tr>
<tr><td>Parameters</td><td>3</td><td>9</td><td>4</td></tr>
<tr><td>Constraints</td><td>Range limits</td><td>6 orthogonality</td><td>1 unit norm</td></tr>
<tr><td>Singularity</td><td>θ = ±90°</td><td>None</td><td>None</td></tr>
<tr><td>Interpolation</td><td>Poor (LERP)</td><td>Poor</td><td>Excellent (SLERP)</td></tr>
<tr><td>Drift</td><td>Low</td><td>Orthogonality drift</td><td>Norm drift (small)</td></tr>
<tr><td>Composition</td><td>Trig-heavy</td><td>3×3 multiply</td><td>4-component product</td></tr>
<tr><td>Intuitive</td><td>High</td><td>Medium</td><td>Low</td></tr>
</table>
"""


class ReferenceTab(QWidget):
    def __init__(self):
        super().__init__()
        from PyQt6.QtWidgets import QTextBrowser
        browser = QTextBrowser()
        browser.setHtml(REF_HTML)
        browser.setOpenExternalLinks(True)
        browser.setStyleSheet(f"background:{BG}; color:{TEXT}; border:none;")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(4, 4, 4, 4)
        lay.addWidget(browser)


# ─────────────────────────────────────────────────────────────────────────────
# Main window
# ─────────────────────────────────────────────────────────────────────────────
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Rotation Explorer")
        self.resize(1100, 720)

        tabs = QTabWidget()
        tabs.addTab(RotationMatrixTab(), "Rotation Matrix")
        tabs.addTab(QuaternionTab(),     "Quaternions")
        tabs.addTab(GimbalLockTab(),     "Gimbal Lock")
        tabs.addTab(ReferenceTab(),      "Reference")

        self.setCentralWidget(tabs)


# ─────────────────────────────────────────────────────────────────────────────
def main():
    # Must be set before QApplication
    fmt = QSurfaceFormat()
    fmt.setDepthBufferSize(24)
    fmt.setProfile(QSurfaceFormat.OpenGLContextProfile.CompatibilityProfile)
    QSurfaceFormat.setDefaultFormat(fmt)

    app = QApplication(sys.argv)
    app.setStyleSheet(STYLE)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
