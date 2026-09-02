# -*- coding: utf-8 -*-
"""
runrunrun.py
Re-maps the color-balance lookup tables (col1_*.tbl ... col4_*.tbl) produced by
Step 1 of the ERDAS color-balance workflow, using MappingReshapeXS.exe, so that
near-zero DN values are stretched and dark areas keep more detail.

For every group col1_N.tbl / col2_N.tbl / col3_N.tbl / col4_N.tbl found in this
folder it produces col1_N.fd.tbl ... col4_N.fd.tbl, which are then consumed by
Step2SPOTf (apply-table color balance).

Source recovered from runrunrun.cpython-37.pyc (originally compiled 2020-11-30);
runrunrun.exe in this folder is the PyInstaller build of this script.
"""
import os
import glob
import subprocess
import time

NameIn = glob.glob('col1_*.tbl')
# skip tables that are already remapped (*.fd.tbl)
NameCheck = [i for i in NameIn if i[-7:] != '.fd.tbl']

FileIn = []
for i in NameCheck:
    iList = i.split('_')
    FileIn.append([iList[0][:-1] + j + '_' + iList[1] for j in ['1', '2', '3', '4']])

for i in FileIn:
    checkV = True
    for j in i:
        if not os.path.exists(j):
            checkV = False
            break
    if checkV is False:
        continue

    ioutFileN = [j[:-4] + '.fd.tbl' for j in i]
    for j in ioutFileN:
        try:
            os.remove(j)
        except OSError:
            pass

    cmdIn = 'MappingReshapeXS ' + ' '.join(i)
    ret = subprocess.Popen(cmdIn)

    # wait until all four remapped tables exist
    while 1:
        fdCurrent = glob.glob('*.fd.tbl')
        checkValue = 0
        for k in ioutFileN:
            if k in fdCurrent:
                checkValue += 1
        if checkValue == 4:
            break

    time.sleep(1)
    while True:
        ret.kill()
        time.sleep(1)
        if not ret.poll() == 1:
            if ret.poll() == 0:
                pass
        break
