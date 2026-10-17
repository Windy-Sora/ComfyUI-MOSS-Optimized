param([string]$ComfyRoot, [string]$PythonPath)
$ErrorActionPreference = 'Stop'
if (-not $ComfyRoot) { $ComfyRoot = Read-Host 'ComfyUI root (contains main.py)' }
if (-not $PythonPath) {
    $candidates = @((Join-Path $ComfyRoot 'python/python.exe'), (Join-Path $ComfyRoot '.venv/Scripts/python.exe'), (Join-Path (Split-Path $ComfyRoot -Parent) 'python_embeded/python.exe'))
    $PythonPath = $candidates | Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1
    if (-not $PythonPath) { $PythonPath = 'python' }
}
$installer = Join-Path $PSScriptRoot '..\install_plugin.py'
& $PythonPath $installer $ComfyRoot
if ($LASTEXITCODE -ne 0) { throw 'Plugin installation failed.' }
