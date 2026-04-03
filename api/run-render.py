import subprocess
import sys


def main():
    subprocess.run(
        ["gunicorn", "--bind", "0.0.0.0:5000", "api.app:app"],
        check=True,
    )
    sys.exit(0)
