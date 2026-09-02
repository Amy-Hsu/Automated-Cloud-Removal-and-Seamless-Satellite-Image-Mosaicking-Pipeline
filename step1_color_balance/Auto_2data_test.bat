@echo off
REM Full Step-1 color-balance chain.
REM Step1SPOTf.bat / Step2SPOTf.bat are generated on YOUR machine by
REM ERDAS IMAGINE Batch (load the .bcf + .bls files and submit) -- they contain
REM machine-specific environment settings and are therefore not shipped here.

call Step1SPOTf.bat

cd distrib
call runrunrun.exe

cd ..
call Step2SPOTf.bat
