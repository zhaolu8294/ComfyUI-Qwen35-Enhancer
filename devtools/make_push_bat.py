# -*- coding: utf-8 -*-
"""生成 CRLF 换行的 push.bat。

为什么需要这个脚本：Write 工具写出来的 .bat 是 LF 换行，而 cmd.exe 解析
LF 换行的 `if (...)` 多行括号块会出错，`pause` 被吞 → 双击一闪而过。
所以这里强制 newline="\r\n"，并且脚本正文改用单行 `if ... goto` 结构，
对行尾不敏感，双保险。

以后要改 push.bat，改本脚本再跑一次，别直接编辑那个 .bat。
"""
import os

REPO = r"E:\AI\ComfyUI-aki-v3\ComfyUI\custom_nodes\ComfyUI-Qwen35-Enhancer"
TARGET = os.path.join(REPO, "push.bat")

CONTENT = r"""@echo off
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
if not errorlevel 1 goto no_changes

echo.
echo [2/4] Committing...
git commit -m "chore: sync local changes"
if errorlevel 1 goto commit_fail
goto do_push

:no_changes
echo.
echo [2/4] Nothing to commit - working tree is clean. Continuing to push.

:do_push
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
"""


def main():
    # newline="\r\n" 是关键：把每个 \n 落盘成 \r\n
    with open(TARGET, "w", encoding="utf-8", newline="\r\n") as f:
        f.write(CONTENT)

    data = open(TARGET, "rb").read()
    crlf = data.count(b"\r\n")
    lone_lf = data.count(b"\n") - crlf
    print(f"已写入: {TARGET}")
    print(f"  大小     : {len(data)} 字节")
    print(f"  CRLF 行数: {crlf}")
    print(f"  孤立 LF  : {lone_lf}  {'<-- 必须是 0' if lone_lf else '(OK)'}")
    assert lone_lf == 0, "还有孤立的 LF，cmd 会闪退"
    print("校验通过。")


if __name__ == "__main__":
    main()
