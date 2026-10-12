# Retrain Stage 1 on ALL 15 families (seeds 42, 43, 44), one after the other.
# Written for a shared laptop where Fluent is running: it never starts a seed unless
# there is enough free RAM, and train_stage1.py itself pauses if free RAM drops
# below 3 GB. Safe to re-run: seeds whose results file exists are skipped.
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File overnight\all_families\run_all_families.ps1

$ErrorActionPreference = "Continue"
$env:OMP_NUM_THREADS = "2"; $env:MKL_NUM_THREADS = "2"; $env:OPENBLAS_NUM_THREADS = "2"
$env:Path = [Environment]::GetEnvironmentVariable("Path", "Machine") + ";" + [Environment]::GetEnvironmentVariable("Path", "User")
$py = (Get-Command python).Source
$root = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$out = Join-Path $root "overnight\all_families"
$log = Join-Path $out "launcher.log"

# PowerShell's `>` and Start-Process -Redirect* treat [ ] in a path as wildcards, and this
# repo lives under "...\[2] Porfolio\[4] Personal Projects\...". Training logs therefore go to a
# bracket-free temp folder and are copied next to the results afterwards.
$logs = Join-Path $env:TEMP "neurofluid_allfam"
New-Item -ItemType Directory -Force -Path $logs | Out-Null

# 3 GB floor for the other jobs + about 2.2 GB for one CUDA training process.
$StartMinGB = 5.2

function Log($m) { "$(Get-Date -Format 'MM-dd HH:mm:ss')  $m" | Add-Content -LiteralPath $log }
function FreeGB { [math]::Round((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory / 1MB, 2) }

Log "launcher started (python $py)"
foreach ($seed in 42, 43, 44) {
    $json = Join-Path $out "stage1_results_allfam_s$seed.json"
    if (Test-Path -LiteralPath $json) { Log "seed $seed already done, skipping"; continue }

    $waited = 0
    while ((FreeGB) -lt $StartMinGB) {
        if ($waited % 300 -eq 0) { Log "seed $seed waiting: free RAM $(FreeGB) GB < $StartMinGB GB start threshold" }
        Start-Sleep -Seconds 30
        $waited += 30
    }
    Log "seed $seed starting (free RAM $(FreeGB) GB, waited $([int]($waited/60)) min)"

    for ($attempt = 1; $attempt -le 2; $attempt++) {
        & $py -u (Join-Path $root "model\train_stage1.py") --seed $seed --tag "allfam_s$seed" `
            --splits (Join-Path $root "data_generation\splits_all_families.csv") `
            --out_dir $out --threads 2 --min_free_gb 3 --epochs 400 --time_cap_min 40 `
            > (Join-Path $logs "train_allfam_s$seed.log") 2> (Join-Path $logs "train_allfam_s$seed.err")
        foreach ($ext in "log", "err") {
            $src = Join-Path $logs "train_allfam_s$seed.$ext"
            if (Test-Path -LiteralPath $src) { Copy-Item -LiteralPath $src -Destination (Join-Path $out "train_allfam_s$seed.$ext") -Force }
        }
        if (Test-Path -LiteralPath $json) { Log "seed $seed finished (attempt $attempt)"; break }
        Log "seed $seed attempt $attempt FAILED (exit $LASTEXITCODE), see $logs\train_allfam_s$seed.err"
    }
}
Log "launcher finished"
"done" | Out-File -LiteralPath (Join-Path $out "launcher_done.flag")
