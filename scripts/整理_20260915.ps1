﻿# ForenSearcher 项目整理脚本（2026-09-15）
# 作用：把文档按项目发展阶段归档到 docs/，把运行残留移到 _trash_20260915/，然后建立 git 仓库并做第一次提交。
# 原则：只"移动"不"删除"（唯一例外是 __pycache__ 缓存目录，它会被 Python 自动重建）。
# 移错了可以从 _trash_20260915/ 里原样移回来。

$ErrorActionPreference = 'Continue'
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$root = 'E:\Codex\workspaces\ChatGPT\ForenSearcher'
Set-Location -LiteralPath $root
$log = Join-Path $root '_整理日志_20260915.txt'
if (Test-Path -LiteralPath $log) { Remove-Item -LiteralPath $log }

function Log($m) {
    $line = "$(Get-Date -Format 'HH:mm:ss') $m"
    Write-Host $line
    Add-Content -LiteralPath $log -Value $line -Encoding UTF8
}

function MoveTo($src, $dstDir, $newName) {
    $p = Join-Path $root $src
    if (-not (Test-Path -LiteralPath $p)) { Log "不存在，跳过: $src"; return }
    $d = Join-Path $root $dstDir
    New-Item -ItemType Directory -Force -Path $d | Out-Null
    if (-not $newName) { $newName = Split-Path $p -Leaf }
    $target = Join-Path $d $newName
    if (Test-Path -LiteralPath $target) { Log "目标已存在，跳过: $src -> $dstDir\$newName"; return }
    try {
        Move-Item -LiteralPath $p -Destination $target -ErrorAction Stop
        Log "移动: $src  ->  $dstDir\$newName"
    } catch { Log "失败: $src  ($($_.Exception.Message))" }
}

Log "===== 开始整理 ====="

# ---------- 1. 文档按发展阶段归档 ----------
Log "--- 1. docs/01 早期探索（已放弃）---"
foreach ($f in 'step1.md','step2.md','step3.md','step4.md','step5.md','step6.md','step7.md','allinone.md',
               'ECSO_HIERARCHICAL_ACTION_TREE_20260817.md','ECSO_DATASET_TOOLBOX_LANDSCAPE_20260816.md',
               'claimunit_video_sham_1004','ideaspark_run','novelty-checks','literature-review') {
    MoveTo $f 'docs\01_早期探索_已放弃'
}

Log "--- 2. docs/02 方案讨论 ---"
MoveTo '研究方案讨论交接文档.md' 'docs\02_方案讨论'

Log "--- 3. docs/03 框架实现 ---"
MoveTo '代码与框架审查报告.md' 'docs\03_框架实现'
MoveTo 'README.md' 'docs\03_框架实现' 'README_旧版_详细技术说明.md'
MoveTo 'academic_group_meeting_site' 'docs\03_框架实现' '组会网站'
MoveTo 'figures' 'docs\03_框架实现'

Log "--- 4. docs/04 训练与数据 ---"
MoveTo '双模型联合自进化训练方案.md' 'docs\04_训练与数据'

Log "--- 5. docs/05 项目审查 ---"
MoveTo '项目审查与梳理_20260915.md' 'docs\05_项目审查'

# 新 README 是以 README.new.md 的名字写进来的，旧 README 移走后再改名
if (Test-Path -LiteralPath (Join-Path $root 'README.new.md')) {
    Rename-Item -LiteralPath (Join-Path $root 'README.new.md') -NewName 'README.md'
    Log "启用新 README.md"
}

# ---------- 2. 运行残留移到 _trash ----------
Log "--- 6. 运行残留 -> _trash_20260915 ---"
$trash = '_trash_20260915'
Get-ChildItem -LiteralPath $root -Force | Where-Object { $_.Name -like '.pytest-*' -or $_.Name -eq '.pytest_cache' } | ForEach-Object {
    MoveTo $_.Name $trash
}
foreach ($f in '.mypy_cache','.pnpm-store','build','dist','dist-final','tmp','refine-logs','debug.log',
               'src\foren_searcher.egg-info','Demo 网站.cmd','OpenRouter 设置.cmd',
               'scripts\reorganize_project_20260915.py','项目结构说明.md') {   # 后两个是另一份重复的整理方案，弃用
    MoveTo $f $trash
}

Log "--- 7. outputs/ 里的一次性验证目录 -> _trash_20260915\outputs ---"
$keepOutputs = @('demo','demo-examples','figures','web-demo')
Get-ChildItem -LiteralPath (Join-Path $root 'outputs') -Force -ErrorAction SilentlyContinue | ForEach-Object {
    if ($keepOutputs -notcontains $_.Name) { MoveTo ("outputs\" + $_.Name) "$trash\outputs" }
}

Log "--- 8. experiments/sft_data 里的一次性探测产物 ---"
$sft = 'experiments\sft_data'
Get-ChildItem -LiteralPath (Join-Path $root $sft) -Force -Directory | ForEach-Object {
    $n = $_.Name
    if ($n -like 'test_tmp_*') { MoveTo "$sft\$n" "$trash\sft_data" }
    elseif ($n -like 'student_inputs_*' -or $n -like 'student_forward_*' -or $n -like 'search_binding_capture*' -or
            $n -like 'search_engine_probe*' -or $n -like '*_probe_*' -or $n -like 'reference_access_*' -or
            $n -eq 'source_article_alias_check') { MoveTo "$sft\$n" "$sft\_diagnostics_archive" }
}

Log "--- 9. 删除 __pycache__ 缓存目录（Python 会自动重建）---"
Get-ChildItem -LiteralPath $root -Recurse -Force -Directory -Filter '__pycache__' -ErrorAction SilentlyContinue |
    Where-Object { $_.FullName -notlike "*\$trash\*" -and $_.FullName -notlike '*\external\*' -and $_.FullName -notlike '*\.models\*' } |
    ForEach-Object {
        try { Remove-Item -LiteralPath $_.FullName -Recurse -Force -ErrorAction Stop; Log "删除缓存: $($_.FullName.Substring($root.Length+1))" }
        catch { Log "删除失败: $($_.FullName)" }
    }

# ---------- 3. git ----------
Log "--- 10. 建立 git 仓库 ---"
$gitCmd = Get-Command git -ErrorAction SilentlyContinue
if (-not $gitCmd) {
    Log "未找到 git。请安装 Git for Windows (https://git-scm.com/download/win) 后重新运行本脚本，前面的整理不会重复执行。"
} else {
    Log ("git 版本: " + (git --version))
    git init 2>&1 | ForEach-Object { Log "git: $_" }
    git config core.quotepath false
    git config core.autocrlf false
    if (-not (git config user.name))  { git config user.name  'Zhiya' }
    if (-not (git config user.email)) { git config user.email 'taniya48121337@gmail.com' }
    git add -A 2>&1 | ForEach-Object { Log "git add: $_" }
    $n = (git diff --cached --name-only | Measure-Object).Count
    Log "准备提交 $n 个文件"
    $big = git diff --cached --name-only | ForEach-Object { $fp = Join-Path $root $_; if (Test-Path -LiteralPath $fp) { Get-Item -LiteralPath $fp } } | Where-Object { $_.Length -gt 20MB }
    foreach ($b in $big) { Log "注意：大文件 >20MB 将被提交: $($b.FullName.Substring($root.Length+1)) ($([math]::Round($b.Length/1MB)) MB)" }
    git commit -q -m "整理项目结构：文档按发展阶段归档到 docs/，清理运行残留，建立 git 仓库" 2>&1 | ForEach-Object { Log "git commit: $_" }
    git log --oneline -1 2>&1 | ForEach-Object { Log "最新提交: $_" }
    git status --short | Measure-Object | ForEach-Object { Log "提交后未跟踪/未提交项: $($_.Count)" }
}

Log "===== 整理完成 ====="
Copy-Item -LiteralPath $log -Destination (Join-Path $root 'docs\05_项目审查\整理日志_20260915.txt') -Force
