@echo off
echo ========================================
echo MLOps Setup for Skin Lesion Classifier
echo ========================================
echo.

:: Check if virtual environment exists
if not exist venv (
    echo Creating virtual environment...
    python -m venv venv
    echo ✓ Virtual environment created
) else (
    echo ✓ Virtual environment already exists
)

:: Activate virtual environment
echo.
echo Activating virtual environment...
call venv\Scripts\activate.bat

:: Install main dependencies
echo.
echo Installing main dependencies...
pip install --upgrade pip
pip install -r requirements.txt
echo ✓ Main dependencies installed

:: Install MLOps dependencies
echo.
echo Installing MLOps dependencies...
pip install -r requirements_mlops.txt
echo ✓ MLOps dependencies installed

:: Initialize DVC
echo.
echo Initializing DVC...
if not exist .dvc (
    dvc init
    echo ✓ DVC initialized
) else (
    echo ✓ DVC already initialized
)

:: Create necessary directories
echo.
echo Creating directories...
if not exist checkpoints mkdir checkpoints
if not exist metrics mkdir metrics
if not exist plots mkdir plots
if not exist mlruns mkdir mlruns
echo ✓ Directories created

:: Initialize Git LFS (optional)
echo.
echo Git LFS setup (optional)...
git lfs install
echo ✓ Git LFS installed

echo.
echo ========================================
echo MLOps Setup Complete!
echo ========================================
echo.
echo Next steps:
echo 1. Configure DVC remote: dvc remote add -d myremote gdrive://folder_id
echo 2. Track data with DVC: dvc add data/
echo 3. Run pipeline: dvc repro
echo 4. Push to GitHub: git push
echo 5. Push DVC data: dvc push
echo.
echo To activate the environment later, run:
echo   venv\Scripts\activate.bat
echo.
pause
