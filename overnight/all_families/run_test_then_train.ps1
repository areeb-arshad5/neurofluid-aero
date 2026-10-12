# Wait for enough free RAM, run the resume test (model\test_resume.py), and start the
# all-families retrain (run_all_families.ps1) only if the test passes.
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File overnight\all_families\run_test_then_train.ps1

$ErrorActionPreference = "Continue"
$env:OMP_NUM_THREADS = "2"; $env:MKL_NUM_THREADS = "2"; $env:OPENBLAS_NUM_THREADS = "2"
$env:Path = [Environment]::GetEnvironmentVariable("Path", "Machine") + ";" + [Environment]::GetEnvironmentVariable("Path", "User")
$py = (Get-Command python).Source
$root = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$out = Join-Path $root "overnight\all_families"
$log = Join-Path $out "launcher.log"
$tmp = Join-Path $env:TEMP "neurofluid_allfam"      # bracket-free: `>` cannot write under [ ] paths
New-Item -ItemType Directory -Force -Path $tmp | Out-Null
$StartMinGB = 5.0

function Log($m) { "$(Get-Date -Format 'MM-dd HH:mm:ss')  $m" | Add-Content -LiteralPath $log }
function FreeGB { [math]::Round((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory / 1MB, 2) }

$waited = 0
while ((FreeGB) -lt $StartMinGB) {
    if ($waited % 600 -eq 0) { Log "resume test waiting: free RAM $(FreeGB) GB < $StartMinGB GB" }
    Start-Sleep -Seconds 30
    $waited += 30
}
Log "resume test starting (free RAM $(FreeGB) GB)"
& $py (Join-Path $root "model\test_resume.py") > (Join-Path $tmp "test_resume.out") 2>&1
$code = $LASTEXITCODE
Copy-Item -LiteralPath (Join-Path $tmp "test_resume.out") -Destination (Join-Path $out "test_resume.out") -Force
if ($code -ne 0) {
    Log "resume test FAILED (exit $code): training NOT started, see test_resume.out"
    "failed" | Out-File -LiteralPath (Join-Path $out "launcher_done.flag")
    exit 1
}
Log "resume test PASSED; starting the retrain launcher"
& powershell.exe -NoProfile -ExecutionPolicy Bypass -File (Join-Path $out "run_all_families.ps1")
