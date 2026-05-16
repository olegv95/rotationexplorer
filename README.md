# Rotation Explorer

An interactive graphical application and HTML guide for visualizing 3D rotation concepts like Direction Cosine Matrices (DCMs), Quaternions, and Gimbal Lock.

## Features

The application is split into four main tabs:
1. **Rotation Matrix Tab**: Adjust Yaw, Pitch, and Roll. See the 3D aircraft model rotate and watch the matrix multiplication `R = Rz · Ry · Rx` update live.
2. **Quaternions Tab**: Manipulate quaternion components, see axis-angle equivalents, and watch the Hamilton product combine rotations.
3. **Gimbal Lock Tab**: Demonstrates the mathematical singularity when pitch approaches &pm;90°. Shows live divergence of Euler rates.
4. **Reference Tab**: A quick reference for key mathematical formulas used in aerospace and 3D graphics.

## Requirements

- Python 3.9+
- Dependencies: `numpy`, `PyQt6`, `PyOpenGL`

## Installation & Setup

1. **Create a virtual environment:**
   ```bash
   python -m venv venv
   ```
2. **Activate the virtual environment:**
   - **Windows (CMD):** `venv\Scripts\activate.bat`
   - **Windows (PowerShell):** `venv\Scripts\Activate.ps1`
   - **Mac/Linux:** `source venv/bin/activate`
3. **Install the dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## How to Use

**1. Desktop Application (Recommended)**
Once your virtual environment is active, run the native PyQt6 desktop application:
```bash
python rotation_app.py
```

**2. Command Line Demo**
Run a pure terminal math demo to see the underlying library functions in action:
```bash
python demo.py
```

**3. Web Version**
There is also a standalone browser version of this tool. Run:
```bash
python generatehtml2.py
```
Then open the generated `rotation_guide.html` in any modern web browser.

## Building a Standalone Executable

You can compile the PyQt6 application into a standalone executable (requires `pyinstaller`):
```bash
pip install pyinstaller
python build_exe.py
```
Your fast-launching compiled app will be placed in the `dist/RotationExplorer/` folder.
