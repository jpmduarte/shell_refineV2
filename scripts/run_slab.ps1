# Waits for the isotropic patch sweep to finish, then runs the slab arm.
#
# The slab config (128x128x32 x4) is matched to the 64^3 x8 cube on both compute
# (~2.10M voxels/step) and supervision (~0.91M band voxels/step), so patch shape is
# the only thing that differs. The cube sweep could not control both at once.

& "C:\Users\user\miniconda3\shell\condabin\conda-hook.ps1"
conda activate fetal-head
Set-Location "C:\Users\user\Desktop\shell_refineV2"

Write-Output "=== waiting for the isotropic sweep ==="
while (-not (Test-Path "phase2/runs/patch128/eval_val/summary.json")) {
    Start-Sleep -Seconds 60
}
Write-Output "sweep finished"

$rd = "phase2/runs/slab128x128x32"
Write-Output "`n=== slab 128x128x32 x4 ==="
python phase2/train_slab.py --fold 0 --crops-dir phase2/crops/all --seed 42 `
    --long 128 --thin 32 --patches-per-case 4 --run-dir $rd 2>&1 | Select-Object -Last 4

python phase2/predict_slab.py --checkpoint "$rd/checkpoints/phase2_best.pth" `
    --crops-dir phase2/crops/all --fold 0 --split val --long 128 --thin 32 `
    --out-dir "$rd/preds_val" 2>$null | Select-Object -Last 4

python phase2/evaluate.py --pred-dir "$rd/preds_val" --out-dir "$rd/eval_val" 2>$null |
    Select-Object -Last 14

python phase2/diagnose.py --pred-dir "$rd/preds_val" 2>$null | Select-Object -Last 7

Write-Output "`n=== slab done ==="
