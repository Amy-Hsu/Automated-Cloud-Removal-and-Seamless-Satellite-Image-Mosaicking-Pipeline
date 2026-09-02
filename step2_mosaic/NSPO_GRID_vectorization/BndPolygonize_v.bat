@echo off
setlocal enabledelayedexpansion
set SourceTiff=%1

if not exist %SourceTiff% (
    echo The assigned source %SourceTiff% doesn't exist...
	goto :eof
)

set EXE=%~dp0
set GDAL_DATA=%EXE%share\data
set PATH=%PATH%;%EXE%bin
set PATH=%PATH%;C:\Program Files\MATLAB\MATLAB Runtime\v94\runtime\win64
set PATH=%PATH%;C:\Program Files\MATLAB\R2018a\runtime\win64

for %%A in ("%SourceTiff%") do (
    set SourcePath=%%~dpA
    set SourceFullName=%%~nxA
	set SourceNoExtName=%%~nA
	set SourceExt=%%~xA
)

echo Image mask processing for valid area: %SourceTiff%
call %EXE%GetBoundaryMask %SourceTiff%
set SourceBoudary=%SourcePath%%SourceNoExtName%.bnd.tif

set IDENTIFILER=%SourcePath%%SourceNoExtName%.info.txt:Pixel Size = (
call :strlen LENGTH IDENTIFILER

call "%EXE%bin\gdalinfo.exe" %SourceBoudary% > %SourcePath%%SourceNoExtName%.info.txt

for /F "tokens=* USEBACKQ" %%F in (`findstr /s "Pixel Size" %SourcePath%%SourceNoExtName%.info.txt`) do (
    set var=%%F
)
set str=!var:~%LENGTH%,-6!
for /f "tokens=1 delims=," %%i in ("%str%") do (set resolution=%%i)
echo Resolution: %resolution%

call %EXE%bin\python %EXE%bin\gdal_edit.py -a_nodata 0 %SourceBoudary%
call %EXE%bin\python %EXE%bin\gdal_polygonize.py -b 1 -f "ESRI Shapefile" %SourceBoudary% %SourcePath%%SourceNoExtName%.temp.shp

goto :eof

:strlen <resultVar> <stringVar>
(   
    setlocal EnableDelayedExpansion
    set "s=!%~2!#"
    set "len=0"
    for %%P in (4096 2048 1024 512 256 128 64 32 16 8 4 2 1) do (
        if "!s:~%%P,1!" NEQ "" ( 
            set /a "len+=%%P"
            set "s=!s:~%%P!"
        )
    )
)
( 
    endlocal
    set "%~1=%len%"
    exit /b
)
