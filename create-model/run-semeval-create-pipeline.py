import subprocess
import sys
from pathlib import Path


def main():
    print(f"Running SemEval pipeline...")
    src_dir = Path(__file__).parent

    scripts = [
        src_dir / "data-pull.py",
        src_dir / "semeval-clean.py",
        src_dir / "semeval-create-si-model.py",
        src_dir / "semeval-create-specialist-model.py"
    ]

    for script in scripts:
        print(f"\n{'='*60}")
        print(f"Running {script.name}...")
        print(f"{'='*60}\n")
        result = subprocess.run([sys.executable, str(script)], cwd=src_dir)
        if result.returncode != 0:
            print(f"\nError: {script.name} failed with exit code {result.returncode}")
            sys.exit(result.returncode)

    print(f"\n{'='*60}")
    print("Pipeline completed successfully.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
