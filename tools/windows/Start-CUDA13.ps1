param([string]$RuntimeDir, [int]$Port = 8189, [switch]$AllCustomNodes, [switch]$NoBrowser)
$ErrorActionPreference = 'Stop'
if (-not $RuntimeDir) { $RuntimeDir = Join-Path $PSScriptRoot '..\..\.runtime' }
$config = Get-Content -LiteralPath (Join-Path $RuntimeDir 'runtime.json') -Raw | ConvertFrom-Json
if (Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue) {
    Write-Host "Port $Port already has a listener; no instance started."
    return
}
$installedPlugin = Join-Path $config.comfy_root 'custom_nodes/ComfyUI-MOSS-Optimized'
$pluginRoot = if (Test-Path -LiteralPath (Join-Path $installedPlugin 'nodes.py')) { $installedPlugin } else { $config.plugin_root }
if (-not $pluginRoot -or -not (Test-Path -LiteralPath (Join-Path $pluginRoot 'nodes.py'))) { throw 'Plugin source is missing. Install the plugin or rerun the runtime setup from its source folder.' }
$pluginName = Split-Path $pluginRoot -Leaf
$userDir = Join-Path $config.state_dir 'user'
$outputDir = Join-Path $config.state_dir 'output'
$logDir = Join-Path $config.state_dir 'logs'
$tempDir = Join-Path $config.state_dir 'temp'
$inputDir = Join-Path $config.state_dir 'input'
foreach ($path in $userDir,$outputDir,$logDir,$tempDir,$inputDir) { New-Item -ItemType Directory -Path $path -Force | Out-Null }
$extraPaths = Join-Path $config.state_dir 'plugin_paths.json'
@{moss_optimized=@{custom_nodes=(Split-Path $pluginRoot -Parent)}} | ConvertTo-Json -Depth 3 | Set-Content -LiteralPath $extraPaths -Encoding UTF8
$entry = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\run_comfy.py'))
$arguments = @(('"'+$entry+'"'), ('"'+$config.comfy_root+'"'), '--listen','127.0.0.1','--port',"$Port",'--use-ck-attention','--disable-xformers','--cuda-malloc','--preview-method','none','--extra-model-paths-config',('"'+$extraPaths+'"'),'--user-directory',('"'+$userDir+'"'),'--output-directory',('"'+$outputDir+'"'),'--temp-directory',('"'+$tempDir+'"'),'--input-directory',('"'+$inputDir+'"'))
if (-not $AllCustomNodes) { $arguments += @('--disable-all-custom-nodes','--whitelist-custom-nodes',('"'+$pluginName+'"')) }
if (-not $NoBrowser) { $arguments += '--auto-launch' }
Start-Process -FilePath $config.python -ArgumentList $arguments -WorkingDirectory $config.comfy_root -WindowStyle Hidden -RedirectStandardOutput (Join-Path $logDir "stdout-$Port.log") -RedirectStandardError (Join-Path $logDir "stderr-$Port.log")
Write-Host "Starting http://127.0.0.1:$Port ; logs: $logDir"
