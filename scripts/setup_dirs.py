"""
Script to set up the directory structure for the FreshRetail Forecasting project.
"""
import os
import sys
from pathlib import Path

def create_directory_structure():
    """Create the necessary directory structure for the project."""
    # Get the project root directory (assuming this script is in scripts/)
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    
    # Define directories to create
    directories = [
        'app',
        'models',
        'notebooks',
        'scripts',
        'static',
        'utils',
        'model_cache',  # For storing trained models
        'data',         # For storing any downloaded data
    ]
    
    # Create directories if they don't exist
    for directory in directories:
        dir_path = project_root / directory
        if not dir_path.exists():
            print(f"Creating directory: {directory}")
            os.makedirs(dir_path, exist_ok=True)
        else:
            print(f"Directory already exists: {directory}")
    
    # Create __init__.py files in each directory if they don't exist
    for directory in directories:
        init_path = project_root / directory / "__init__.py"
        if not init_path.exists() and directory not in ['data', 'model_cache', 'static']:
            print(f"Creating __init__.py in {directory}")
            with open(init_path, 'w') as f:
                f.write(f'"""\nFreshRetail Forecasting {directory} package.\n"""\n')
    
    print("\nDirectory structure setup complete!")

def main():
    """Main function"""
    create_directory_structure()

if __name__ == "__main__":
    main() 