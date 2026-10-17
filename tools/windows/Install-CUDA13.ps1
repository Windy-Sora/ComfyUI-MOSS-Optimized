param([string]$ComfyRoot, [string]$PythonPath, [string]$RuntimeDir, [switch]$Resume)
$ErrorActionPreference = 'Stop'
if (-not $ComfyRoot) { $ComfyRoot = Read-Host 'ComfyUI root (contains main.py)' }
if (-not $PythonPath) {
    $candidates = @((Join-Path $ComfyRoot 'python/python.exe'), (Join-Path $ComfyRoot '.venv/Scripts/python.exe'), (Join-Path (Split-Path $ComfyRoot -Parent) 'python_embeded/python.exe'))
    $PythonPath = $candidates | Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1
    if (-not $PythonPath) { $PythonPath = 'python' }
}
$ComfyRoot = (Resolve-Path -LiteralPath $ComfyRoot).Path
if (-not (Test-Path -LiteralPath (Join-Path $ComfyRoot 'requirements.txt'))) { throw 'Missing ComfyUI requirements.txt' }
if (-not $RuntimeDir) { $RuntimeDir = Join-Path $PSScriptRoot '..\..\.runtime' }
$RuntimeDir = [IO.Path]::GetFullPath($RuntimeDir)
if ((Test-Path -LiteralPath $RuntimeDir) -and -not $Resume) { throw 'Runtime directory already exists. Choose a new -RuntimeDir or use -Resume for this dedicated venv.' }
Write-Host 'Downloads: PyTorch CUDA 13 wheels, pinned runtime packages and this ComfyUI version requirements.'
Write-Host "Creating a separate environment: $RuntimeDir"
if ($Resume) {
    if (-not (Test-Path -LiteralPath (Join-Path $RuntimeDir 'pyvenv.cfg')) -or -not (Test-Path -LiteralPath (Join-Path $RuntimeDir '.moss-optimized-runtime'))) { throw '-Resume requires a dedicated venv created by this tool.' }
} else {
    & $PythonPath -m venv $RuntimeDir
    if ($LASTEXITCODE -ne 0) { throw 'venv creation failed' }
    Set-Content -LiteralPath (Join-Path $RuntimeDir '.moss-optimized-runtime') -Value 'ComfyUI-MOSS-Optimized 0.1' -Encoding ASCII
}
$runtimePython = Join-Path $RuntimeDir 'Scripts\python.exe'
& $runtimePython -m pip install --upgrade pip --index-url https://pypi.org/simple
if ($LASTEXITCODE -ne 0) { throw 'pip installation failed' }
$constraints = Join-Path $RuntimeDir 'constraints.txt'
@('torch==2.10.0+cu130','torchvision==0.25.0+cu130','torchaudio==2.10.0+cu130','comfy-kitchen==0.2.37','comfy-aimdo==0.5.6','triton-windows==3.6.0.post26','numpy==1.26.4','transformers==4.57.1','tokenizers==0.22.1','safetensors==0.6.2') | Set-Content -LiteralPath $constraints -Encoding ASCII
& $runtimePython -m pip install torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 --index-url https://download.pytorch.org/whl/cu130
if ($LASTEXITCODE -ne 0) { throw 'PyTorch installation failed' }
& $runtimePython -m pip install --index-url https://pypi.org/simple -r (Join-Path $ComfyRoot 'requirements.txt') -c $constraints triton-windows==3.6.0.post26
if ($LASTEXITCODE -ne 0) { throw 'ComfyUI dependencies differ from the tested runtime. Original environment is unchanged; see pip error above.' }
$checker = Join-Path $PSScriptRoot '..\check_runtime.py'
& $runtimePython $checker $ComfyRoot --compile
if ($LASTEXITCODE -ne 0) { throw 'Runtime verification failed. Original environment is unchanged.' }
$pluginRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot "..\.."))
$config = @{comfy_root=$ComfyRoot; python=$runtimePython; state_dir=$RuntimeDir; plugin_root=$pluginRoot}
$config | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $RuntimeDir 'runtime.json') -Encoding UTF8
Write-Host 'Runtime verified. Use Start-CUDA13.bat -RuntimeDir with this directory; plugin source can remain external.'
