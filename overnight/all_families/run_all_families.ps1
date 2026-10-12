# Retrain Stage 1 on ALL 15 families (seeds 42, 43, 44), one after the other.
# Written for a shared laptop where Fluent is running:
#   - a seed only starts when enough RAM is free ($StartMinGB);
#   - while training, free RAM below 3 GB pauses it; if the pause lasts 2 minutes the
#     trainer saves its state and EXITS (code 75), so it stops holding its own ~1.4 GB;
#   - this launcher then waits for RAM again and resumes from the saved epoch.
# Safe to re-run: seeds whose results file exists are skipped, and an interrupted seed resumes.
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

# 3 GB floor for the other jobs + about 1.4 GB measured for one CUDA training process,
# plus margin so a small fluctuation does not immediately trigger a pause.
$StartMinGB = 5.0
$MaxFailures = 2

function Log($m) { "$(Get-Date -Format 'MM-dd HH:mm:ss')  $m" | Add-Content -LiteralPath $log }
function FreeGB { [math]::Round((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory / 1MB, 2) }
function CopyLogs($seed) {
    foreach ($ext in "log", "err") {
        $src = Join-Path $logs "train_allfam_s$seed.$ext"
        if (Test-Path -LiteralPath $src) { Copy-Item -LiteralPath $src -Destination (Join-Path $out "train_allfam_s$seed.$ext") -Force }
    }
}

Log "launcher started (python $py)"
foreach ($seed in 42, 43, 44) {
    $json = Join-Path $out "stage1_results_allfam_s$seed.json"
    $failures = 0
    while (-not (Test-Path -LiteralPath $json) -and $failures -lt $MaxFailures) {
        $waited = 0
        while ((FreeGB) -lt $StartMinGB) {
            if ($waited % 600 -eq 0) { Log "seed $seed waiting: free RAM $(FreeGB) GB < $StartMinGB GB start threshold" }
            Start-Sleep -Seconds 30
            $waited += 30
        }
        Log "seed $seed starting or resuming (free RAM $(FreeGB) GB, waited $([int]($waited/60)) min)"

        & $py -u (Join-Path $root "model\train_stage1.py") --seed $seed --tag "allfam_s$seed" `
            --splits (Join-Path $root "data_generation\splits_all_families.csv") `
            --out_dir $out --threads 2 --min_free_gb 3 --max_pause_s 120 --resume `
            --epochs 400 --time_cap_min 40 `
            >> (Join-Path $logs "train_allfam_s$seed.log") 2>> (Join-Path $logs "train_allfam_s$seed.err")
        $code = $LASTEXITCODE
        CopyLogs $seed

        if (Test-Path -LiteralPath $json) { Log "seed $seed finished"; break }
        if ($code -eq 75) { Log "seed $seed released its RAM (low-RAM pause), will resume when RAM allows"; continue }
        $failures++
        Log "seed $seed FAILED (exit $code), failure $failures of $MaxFailures, see $logs\train_allfam_s$seed.err"
    }
}
Log "launcher finished"
"done" | Out-File -LiteralPath (Join-Path $out "launcher_done.flag")
