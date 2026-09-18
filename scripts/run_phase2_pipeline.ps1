# Waits for phase 1 folds 0-4, then builds leak-free crops and runs the
# band-masked-loss ablation end to end.
#
#   powershell -File scripts/run_phase2_pipeline.ps1
#
# Each case's prior comes from the fold whose VAL split contains it, i.e. from a
# phase 1 model that never trained on it. Without that, phase 2 trains against
# priors far cleaner than it sees at inference and its measured gain is inflated.

$ErrorActionPreference = "Stop"
& "C:\Users\user\miniconda3\shell\condabin\conda-hook.ps1"
conda activate fetal-head
Set-Location "C:\Users\user\Desktop\shell_refineV2"

# --- wait for all five phase 1 folds to reach epoch 250 -----------------------
Write-Output "=== waiting for phase 1 folds ==="
foreach ($f in 0, 1, 2, 3, 4) {
    $log = "phase1/runs/fold$f/train_log.csv"
    while ($true) {
        if (Test-Path $log) {
            $last = (Get-Content $log | Select-Object -Last 1).Split(",")[0]
            if ($last -eq "250") { Write-Output "fold$f ready"; break }
        }
        Start-Sleep -Seconds 30
    }
}

# --- leak-free priors: each fold predicts only its own val split -------------
Write-Output "`n=== phase 1 predictions (val split per fold) ==="
foreach ($f in 0, 1, 2, 3, 4) {
    $ckpt = "phase1/runs/fold$f/checkpoints/phase1_best.pth"
    python phase1/predict.py --checkpoint $ckpt --fold $f --split val `
        --out-dir "phase1/runs/fold$f/preds_val" 2>&1 | Select-Object -Last 3
}

# --- crops: all five prediction sets merged into one directory ---------------
Write-Output "`n=== phase 2 crops (90 cases, leak-free) ==="
foreach ($f in 0, 1, 2, 3, 4) {
    python phase2/make_crops.py --pred-dir "phase1/runs/fold$f/preds_val" `
        --out-dir "phase2/crops/all" 2>&1 | Select-Object -Last 2
}
$nCrops = (Get-ChildItem "phase2/crops/all" -Filter *.npz).Count
Write-Output "crops built: $nCrops (expect 90)"

# --- ablation: band-masked loss vs whole-patch loss, same fold --------------
Write-Output "`n=== phase 2 training: band-masked ==="
python phase2/train.py --fold 0 --crops-dir phase2/crops/all --seed 42 `
    --run-dir phase2/runs/fold0_masked 2>&1 | Select-Object -Last 6

Write-Output "`n=== phase 2 training: whole-patch (ablation) ==="
python phase2/train.py --fold 0 --crops-dir phase2/crops/all --seed 42 `
    --no-band-masked-loss --run-dir phase2/runs/fold0_unmasked 2>&1 | Select-Object -Last 6

# --- evaluate both on the honest metric (stitched, full resolution) ---------
foreach ($arm in "masked", "unmasked") {
    Write-Output "`n=== phase 2 predict+evaluate: $arm ==="
    python phase2/predict.py --checkpoint "phase2/runs/fold0_$arm/checkpoints/phase2_best.pth" `
        --crops-dir phase2/crops/all --fold 0 --split val `
        --out-dir "phase2/runs/fold0_$arm/preds_val" 2>&1 | Select-Object -Last 3
    python phase2/evaluate.py --pred-dir "phase2/runs/fold0_$arm/preds_val" `
        --out-dir "phase2/runs/fold0_$arm/eval_val" 2>&1 | Select-Object -Last 22
}

Write-Output "`n=== done ==="
