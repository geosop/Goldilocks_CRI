# -*- coding: utf-8 -*-
"""
Created on Wed Jul  2 14:06:36 2025

@author: ADMIN

generate_figures.py: one-click script to run all figure-making modules in the figures/ directory.
"""
import subprocess
from pathlib import Path

def main():
    # Determine script and figures directory
    script_dir = Path(__file__).resolve().parent
    figures_dir = script_dir / 'figures'

    # Verify figures directory exists
    if not figures_dir.exists():
        print(f"Error: Figures directory not found at {figures_dir}")
        return

    # Debug: list directory contents
    print(f"Contents of {figures_dir}:")
    for item in sorted(figures_dir.iterdir()):
        print("  ", item.name)

    # Collect all figure scripts matching your naming convention
    figure_scripts = sorted(figures_dir.glob('make_*_figure.py'))
    if not figure_scripts:
        print("No figure scripts matching 'make_*_figure.py' found in figures/ directory.")
        return

    # Run each figure-generation script sequentially
    for script in figure_scripts:
        print(f"Running {script.name}...")
        result = subprocess.run(['python', str(script)], check=False)
        if result.returncode != 0:
            print(f"Error: {script.name} exited with code {result.returncode}.")
            exit(result.returncode)

    print("All figures have been generated successfully.")

if __name__ == '__main__':
    main()

