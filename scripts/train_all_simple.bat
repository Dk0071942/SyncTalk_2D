@echo off
REM Dynamic script to train all available datasets

SET GPU_ID=%1
IF "%GPU_ID%"=="" SET GPU_ID=0
SET DATASET_DIR=dataset

echo Training all datasets on GPU %GPU_ID%
echo ====================================

REM Check if dataset directory exists
IF NOT EXIST "%DATASET_DIR%" (
    echo Error: Dataset directory '%DATASET_DIR%' not found!
    exit /b 1
)

REM Find all datasets (directories containing at least one mp4 file)
echo Scanning for available datasets...
SETLOCAL ENABLEDELAYEDEXPANSION
SET count=0

FOR /D %%D IN ("%DATASET_DIR%\*") DO (
    REM Check if any mp4 files exist in the directory
    SET found=0
    FOR %%F IN ("%%D\*.mp4") DO (
        IF EXIST "%%F" SET found=1
    )
    
    IF !found!==1 (
        SET /A count+=1
        SET dataset!count!=%%~nxD
        echo   Found: %%~nxD
    )
)

REM Check if any datasets were found
IF %count%==0 (
    echo No datasets found in %DATASET_DIR%
    echo Datasets should contain at least one '.mp4' file
    exit /b 1
)

echo.
echo Found %count% dataset(s) to train
echo ====================================

REM Train each dataset
FOR /L %%i IN (1,1,%count%) DO (
    echo.
    echo Training dataset: !dataset%%i!
    echo ------------------------------------
    call scripts\train_328.bat !dataset%%i! %GPU_ID%
    
    REM Check if training was successful
    IF !ERRORLEVEL! NEQ 0 (
        echo Warning: Training failed for dataset !dataset%%i!
    ) ELSE (
        echo Successfully completed training for !dataset%%i!
    )
)

echo.
echo ====================================
echo All training completed!

REM Display all trained datasets
echo Trained datasets:
FOR /L %%i IN (1,1,%count%) DO (
    echo   - !dataset%%i!
)

ENDLOCAL
pause