import subprocess
from pathlib import Path

def run_command(command):
    print(f"Running: {command}")
    subprocess.run(command, shell=True, check=True)

def setup_git():
    base_dir = Path(__file__).parent
    git_dir = base_dir / ".git"

    # 1. Initialize Git if it doesn't exist
    if not git_dir.exists():
        run_command("git init")
    
    # 2. Add all tracked files (respecting the .gitignore we created)
    run_command("git add .")
    
    # 3. Commit the code
    try:
        run_command('git commit -m "Initial commit: Rotation Explorer interactive application"')
    except subprocess.CalledProcessError:
        print("\nNote: Nothing new to commit. Files are already tracked!")

    print("\n" + "="*60)
    print("✅ Local Git Repository Setup Complete!")
    print("="*60)
    print("\nTo automatically create and push this to GitHub, install the GitHub CLI (https://cli.github.com/)")
    print("Once installed, just copy and paste this exact command into your terminal:\n")
    print('gh repo create RotationExplorer --public --source=. --remote=origin --push --description "Interactive 3D rotation concepts explorer (Euler ZYX, Quaternions, Gimbal Lock) built in PyQt6."\n')

if __name__ == "__main__":
    setup_git()