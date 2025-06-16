#!/usr/bin/env python3
"""
Claude Monitor Launcher
Starts the Claude state monitor as a background process
"""

import subprocess
import sys
import os
from pathlib import Path

def check_dependencies():
    """Check and install required dependencies"""
    required = ['pillow']
    optional = ['PyQt6', 'numpy']
    
    print("Checking dependencies...")
    
    # Check required dependencies
    for pkg in required:
        try:
            __import__(pkg.lower())
            print(f"✓ {pkg} found")
        except ImportError:
            print(f"✗ {pkg} not found. Installing...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])
    
    # Check optional dependencies
    for pkg in optional:
        try:
            __import__(pkg)
            print(f"✓ {pkg} found")
        except ImportError:
            print(f"⚠ {pkg} not found (optional)")
            response = input(f"Install {pkg} for better experience? (y/n): ")
            if response.lower() == 'y':
                subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])

def start_monitor():
    """Start the monitor applet"""
    script_path = Path(__file__).parent / "claude_monitor_applet.py"
    
    if not script_path.exists():
        print(f"Error: Monitor script not found at {script_path}")
        return
    
    print("\nStarting Claude State Monitor...")
    print("The monitor will run in the system tray (or as a window if PyQt6 not available)")
    print("\nPress Ctrl+C to stop")
    
    try:
        # Run the monitor
        subprocess.run([sys.executable, str(script_path)])
    except KeyboardInterrupt:
        print("\nMonitor stopped")
    except Exception as e:
        print(f"Error: {e}")

def main():
    print("Claude State Monitor Launcher")
    print("=" * 40)
    
    # Check dependencies
    check_dependencies()
    
    # Start monitor
    start_monitor()

if __name__ == "__main__":
    main()