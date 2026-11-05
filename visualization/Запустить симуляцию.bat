@echo off
chcp 65001 >nul
rem Двойной клик по этому файлу запускает интерфейс и открывает браузер.
rem Окно остаётся открытым: в нём видно ошибки и им же всё гасится.
setlocal enabledelayedexpansion
cd /d "%~dp0"

echo Симуляция оборота контейнеров
echo ==============================
echo.

rem --- ищем uv: в PATH, потом там, куда его кладёт установщик ---
set "UV="
where uv >nul 2>&1 && set "UV=uv"
if not defined UV if exist "%USERPROFILE%\.local\bin\uv.exe" set "UV=%USERPROFILE%\.local\bin\uv.exe"
if not defined UV (
    echo Не найден uv. Поставьте его одной командой в PowerShell,
    echo закройте это окно и запустите файл снова:
    echo.
    echo     powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 ^| iex"
    echo.
    pause
    exit /b 1
)

if not exist "model" if "%SIMVIZ_MODEL_PATH%"=="" (
    echo Отсутствует папка model/ (этого не должно было произойти)
    echo.
    pause
    exit /b 1
)

echo Готовлю окружение, в первый раз это займет минуту...
"%UV%" sync --quiet
if errorlevel 1 (
    echo.
    echo Не удалось собрать окружение, смотрите ошибку выше.
    pause
    exit /b 1
)

rem --- свободный порт из диапазона 8050-8070 ---
set PORT=8050
:findport
netstat -ano -p tcp | findstr /r /c:":%PORT% .*LISTENING" >nul 2>&1
if not errorlevel 1 (
    set /a PORT=!PORT!+1
    if !PORT! gtr 8070 (
        echo Все порты 8050-8070 заняты.
        pause
        exit /b 1
    )
    goto findport
)

echo Запускаю на http://127.0.0.1:!PORT!
echo Первый расчёт идет около четырех секунд.
echo.
echo Чтобы остановить, закройте это окно или нажмите Ctrl+C.
echo.

start "" /b cmd /c "timeout /t 8 >nul & start """" http://127.0.0.1:!PORT!"
"%UV%" run simviz --port !PORT!

echo.
echo Сервер остановлен.
pause
