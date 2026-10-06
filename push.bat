@echo off
chcp 65001 >nul 2>&1
REM ============================================================
REM  One-click push: commit local changes and push to GitHub.
REM  - Put this file in the repo root; just double-click it.
REM    It uses %~dp0 so it works from any location.
REM  - Auth is SSH (passwordless). The public key must already be
REM    added at GitHub -> Settings -> SSH and GPG keys.
REM  - GitHub only publishes a new version when pyproject.toml changes.
REM  - KEEP THIS FILE IN CRLF LINE ENDINGS or cmd will exit instantly.
REM ============================================================
setlocal
cd /d "%~dp0"

echo [1/4] Checking repository status...
git status --short
if errorlevel 1 goto no_git

git add -A
git diff --cached --quiet
if not errorlevel 1 goto nothing

echo.
echo [2/4] Committing...
git commit -m "chore: sync local changes"
if errorlevel 1 goto commit_fail

echo.
echo [3/4] Pushing to GitHub...
git push -u origin main
if errorlevel 1 goto push_fail

echo.
echo [4/4] Done.
git log --oneline -3
echo.
echo If pyproject.toml changed, GitHub Actions will publish to the ComfyUI Registry.
pause
exit /b 0

:no_git
echo.
echo [ERROR] Not a git repository, or git is not installed.
goto end

:nothing
echo.
echo [SKIP] Nothing to commit - working tree is clean.
pause
exit /b 0

:commit_fail
echo.
echo [ERROR] Commit failed.
goto end

:push_fail
echo.
echo [ERROR] Push failed. Common causes:
echo   - no remote configured:
echo       git remote add origin git@github.com:zhaolu8294/ComfyUI-Qwen35-Enhancer.git
echo   - public key not added yet: https://github.com/settings/keys
echo   - the GitHub repo does not exist yet (create it EMPTY, no README)
echo   - port 22 blocked by your network: switch to 443 (see README)
goto end

:end
echo.
pause
exit /b 1
