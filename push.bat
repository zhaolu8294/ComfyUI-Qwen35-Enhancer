@echo off
REM ============================================================
REM  一键推送：把本目录的改动提交并推到 GitHub
REM  - 放在仓库根目录，双击即可；用 %~dp0 定位，换机器也能用
REM  - 首次 push 会弹窗要凭据，密码栏填 Personal Access Token
REM  - 只有 pyproject.toml 有改动时 GitHub 才会自动发布新版本
REM ============================================================
setlocal
cd /d "%~dp0"

echo [1/4] 检查仓库状态...
git status --short
if errorlevel 1 (
    echo [错误] 这不是一个 git 仓库，或 git 未安装。
    pause
    exit /b 1
)

git add -A
git diff --cached --quiet
if not errorlevel 1 (
    echo.
    echo [跳过] 没有需要提交的改动。
    pause
    exit /b 0
)

echo.
echo [2/4] 提交...
git commit -m "chore: sync local changes"
if errorlevel 1 (
    echo [错误] 提交失败。
    pause
    exit /b 1
)

echo.
echo [3/4] 推送到 GitHub...
git push -u origin main
if errorlevel 1 (
    echo.
    echo [错误] 推送失败。常见原因：
    echo   - 还没配置 remote：git remote add origin https://github.com/zhaolu8294/ComfyUI-Qwen35-Enhancer.git
    echo   - 凭据过期：在 Windows 凭据管理器里删掉 github.com 后重试
    pause
    exit /b 1
)

echo.
echo [4/4] 完成。
git log --oneline -3
echo.
echo 若本次改动了 pyproject.toml，GitHub Actions 会自动发布到 ComfyUI Registry。
pause
