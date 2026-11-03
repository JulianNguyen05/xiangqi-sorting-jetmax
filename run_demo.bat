@echo off
REM Demo thu vien co tuong tren laptop Windows. Bam dup file nay.
chcp 65001 >nul
set PYTHONUTF8=1
cd /d "%~dp0"
where py >nul 2>nul && (set PY=py) || (set PY=python)
if not exist .deps_ok (
  echo Cai thu vien lan dau, mat vai phut...
  %PY% -m pip install -r requirements-demo.txt || goto :loi
  echo ok> .deps_ok
)
%PY% tools\demo_app.py
pause
exit /b
:loi
echo Cai thu vien that bai. Xem DEMO_WINDOWS.md muc Loi thuong gap.
pause
