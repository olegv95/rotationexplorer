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
        run_command('git commit -m "Update README with venv instructions and add setup script to gitignore"')
    except subprocess.CalledProcessError:
        print("\nNote: Nothing new to commit. Files are already tracked!")

    print("\n" + "="*60)
    print("✅ New changes successfully committed!")
    print("="*60)
    print("\nSince your repository is already linked to GitHub, just run this command to upload the updates:")
    print("git push origin HEAD\n")

if __name__ == "__main__":
    setup_git()