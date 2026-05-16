import PyInstaller.__main__
from pathlib import Path

def build_executable():
    # Define paths
    base_dir = Path(__file__).parent
    main_script = base_dir / "rotation_app.py"
    
    print(f"Starting PyInstaller build for {main_script.name}...")
    
    # PyInstaller configuration arguments
    args = [
        str(main_script),
        '--name=RotationExplorer',   # Name of the output executable
        '--onedir',                  # Create a one-folder bundle (fast startup, libraries next to exe)
        '--windowed',                # Do not show a command prompt console
        '--noconfirm',               # Overwrite existing build/dist folders without asking
        '--clean',                   # Clean PyInstaller cache before building
    ]
    
    # Run PyInstaller
    PyInstaller.__main__.run(args)
    
    print("\nBuild complete!")
    print(f"Your executable and libraries are located in: {base_dir / 'dist' / 'RotationExplorer'}")

if __name__ == "__main__":
    build_executable()