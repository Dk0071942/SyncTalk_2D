@echo off
REM Simple script to train all datasets

SET GPU_ID=%1
IF "%GPU_ID%"=="" SET GPU_ID=0

echo Training all datasets on GPU %GPU_ID%
echo ====================================

REM Train each person
call scripts\train_328.bat LS1 %GPU_ID%
call scripts\train_328.bat 250627_CB %GPU_ID%
REM Add more datasets here as needed
REM call scripts\train_328.bat person3 %GPU_ID%
REM call scripts\train_328.bat person4 %GPU_ID%

echo All training completed!
pause