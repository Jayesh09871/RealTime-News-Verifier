#!/usr/bin/env python3
"""Convenience launcher for the Real-Time News Verification System."""

import subprocess
import sys
from pathlib import Path


def main():
    root = Path(__file__).resolve().parent
    app_path = root / "app" / "streamlit_app.py"

    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(app_path),
        "--browser.gatherUsageStats",
        "false",
        "--server.headless",
        "false",
    ]

    print(f"Starting Real-Time News Verification System via Streamlit...")
    print(f"Target: {app_path}")
    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\nApplication stopped by user.")


if __name__ == "__main__":
    main()
