#!/usr/bin/env python3
"""
Build script to create executable for UFC Prediction GUI
"""

import subprocess
import sys
import os
import shutil

def install_pyinstaller():
    """Install PyInstaller if not already installed"""
    try:
        import PyInstaller
        print("✓ PyInstaller is already installed")
    except ImportError:
        print("Installing PyInstaller...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pyinstaller"])
        print("✓ PyInstaller installed successfully")

def build_executable():
    """Build the executable using PyInstaller"""
    print("Building UFC Prediction GUI executable...")
    
    # Find PyInstaller executable
    pyinstaller_path = None
    possible_paths = [
        "pyinstaller",
        "C:\\Users\\Bryan\\AppData\\Roaming\\Python\\Python313\\Scripts\\pyinstaller.exe",
        sys.executable.replace("python.exe", "Scripts\\pyinstaller.exe"),
        sys.executable.replace("python.exe", "Scripts\\pyinstaller")
    ]
    
    for path in possible_paths:
        try:
            subprocess.run([path, "--version"], capture_output=True, check=True)
            pyinstaller_path = path
            print(f"✓ Found PyInstaller at: {path}")
            break
        except (subprocess.CalledProcessError, FileNotFoundError):
            continue
    
    if pyinstaller_path is None:
        print("Could not find PyInstaller executable")
        return False
    
    # PyInstaller command
    cmd = [
        pyinstaller_path,
        "--onefile",  # Create a single executable file
        "--windowed",  # Don't show console window
        "--name=UFC_Prediction_GUI",  # Name of the executable
        "--add-data=dataUFC;dataUFC",  # Include data directory
        "--add-data=models;models",  # Include models directory
        "--add-data=utils;utils",  # Include utils directory
        "--hidden-import=pandas",
        "--hidden-import=numpy",
        "--hidden-import=joblib",
        "--hidden-import=sklearn",
        "--hidden-import=sklearn.ensemble",
        "--hidden-import=sklearn.tree",
        "--hidden-import=sklearn.model_selection",
        "--hidden-import=sklearn.metrics",
        "ufc_prediction_gui.py"
    ]
    
    try:
        subprocess.check_call(cmd)
        print("✓ Executable built successfully!")
        print("Executable location: dist/UFC_Prediction_GUI.exe")
        
        # Copy necessary files to dist directory
        print("Copying additional files...")
        if os.path.exists("dataUFC"):
            shutil.copytree("dataUFC", "dist/dataUFC", dirs_exist_ok=True)
        if os.path.exists("models"):
            shutil.copytree("models", "dist/models", dirs_exist_ok=True)
        if os.path.exists("utils"):
            shutil.copytree("utils", "dist/utils", dirs_exist_ok=True)
        
        print("✓ All files copied successfully!")
        
    except subprocess.CalledProcessError as e:
        print(f" Error building executable: {e}")
        return False
    
    return True

def create_readme():
    """Create a README file for the executable"""
    readme_content = """# UFC Prediction GUI - Executable

## Installation
1. Extract all files from the ZIP archive
2. Run `UFC_Prediction_GUI.exe`

## Requirements
- Windows 10/11 (64-bit)
- No additional software installation required

## Features
- Browse complete UFC fighter database (2,114 fighters)
- Search fighters by name
- Select two fighters for prediction
- Get detailed fight predictions with confidence levels
- Uses ELO-enhanced machine learning model (99.59% accuracy)

## How to Use
1. Launch the application
2. Go to "Fighter Database" tab to browse fighters
3. Search for specific fighters or scroll through the list
4. Double-click a fighter to select them for a fight
5. Switch to "Fight Prediction" tab
6. Select your two fighters
7. Click "Predict Fight Outcome" to get results

## Model Information
- Algorithm: Random Forest Classifier (Optimized)
- Accuracy: 99.59%
- Features: 121 total (including 9 ELO features)
- Training Data: Comprehensive UFC fight statistics

## Files Included
- `UFC_Prediction_GUI.exe` - Main application
- `dataUFC/` - Fighter data and statistics
- `models/` - Trained machine learning models
- `utils/` - Utility functions

## Note
Predictions are based on historical data and should be used for entertainment purposes only.
"""
    
    with open("dist/README.txt", "w") as f:
        f.write(readme_content)
    
    print("✓ README.txt created")

def main():
    """Main build process"""
    print("=" * 60)
    print("UFC PREDICTION GUI - EXECUTABLE BUILDER")
    print("=" * 60)
    
    # Check if required files exist
    required_files = [
        "ufc_prediction_gui.py",
        "dataUFC/fighter_id_mapping.csv",
        "dataUFC/ufc_engineered.csv",
        "models/random_forest_optimized_phase4_enhanced.joblib"
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print("❌ Missing required files:")
        for file in missing_files:
            print(f"   - {file}")
        print("\nPlease ensure all required files are present before building.")
        return
    
    print("✓ All required files found")
    
    # Install PyInstaller
    install_pyinstaller()
    
    # Build executable
    if build_executable():
        create_readme()
        
        print("\n" + "=" * 60)
        print("BUILD COMPLETE!")
        print("=" * 60)
        print("Executable location: dist/UFC_Prediction_GUI.exe")
        print("Documentation: dist/README.txt")
        print("\nYou can now distribute the 'dist' folder as a complete application.")
        print("Users can run UFC_Prediction_GUI.exe without installing Python.")
    else:
        print("❌ Build failed!")

if __name__ == "__main__":
    main()
