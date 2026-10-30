#!/usr/bin/env bash
# Ubuntu + NVIDIA driver + CUDA-enabled PyTorch
# Example: GPU=0 DURATION=60 BATCH=32 IMAGE_SIZE=224 bash gpu_bench.sh
set -Eeuo pipefail

GPU="${GPU:-0}"
DURATION="${DURATION:-60}"
BATCH="${BATCH:-32}"
IMAGE_SIZE="${IMAGE_SIZE:-224}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
OUT_DIR="${OUT_DIR:-gpu-bench-$(date +%Y%m%d-%H%M%S)-$$}"

for cmd in nvidia-smi "$PYTHON_BIN"; do
    command -v "$cmd" >/dev/null || { echo "Missing command: $cmd" >&2; exit 1; }
done

# Resolve the physical GPU once, so CUDA and nvidia-smi use the same device.
GPU_UUID=$(nvidia-smi -i "$GPU" --query-gpu=uuid --format=csv,noheader | tr -d '[:space:]')
[[ "$GPU_UUID" =~ ^GPU-[0-9a-fA-F-]+$ ]] || { echo "Select one physical GPU with GPU=0 or GPU=GPU-..." >&2; exit 1; }
export CUDA_VISIBLE_DEVICES="$GPU_UUID"
mkdir -p -- "$OUT_DIR"

QUERY="timestamp,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw"
nvidia-smi -i "$GPU_UUID" --query-gpu="$QUERY" --format=csv,noheader,nounits >/dev/null
MON_PID=""
TRAIN_PID=""
cleanup() {
    local pid
    for pid in "$TRAIN_PID" "$MON_PID"; do
        if [[ -n "$pid" ]]; then
            kill "$pid" 2>/dev/null || true
            wait "$pid" 2>/dev/null || true
        fi
    done
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

printf 'timestamp,gpu_util_pct,memory_used_mib,memory_total_mib,temperature_c,power_w\n' > "$OUT_DIR/gpu.csv"
nvidia-smi -i "$GPU_UUID" --query-gpu="$QUERY" --format=csv,noheader,nounits -l 1 \
    >> "$OUT_DIR/gpu.csv" 2> "$OUT_DIR/monitor.log" &
MON_PID=$!
echo "GPU=$GPU_UUID | duration=${DURATION}s | batch=$BATCH | image=$IMAGE_SIZE | logs=$OUT_DIR"

"$PYTHON_BIN" -u - "$DURATION" "$BATCH" "$IMAGE_SIZE" "$OUT_DIR" <<'PY' &
import logging
import math
from pathlib import Path
import signal
import sys
import time

# Bash background jobs inherit ignored SIGINT; restore Ctrl+C handling.
signal.signal(signal.SIGINT, signal.default_int_handler)
out = Path(sys.argv[4])
logging.basicConfig(level=logging.INFO, format="%(message)s", handlers=[
    logging.StreamHandler(sys.stdout), logging.FileHandler(out / "train.log")
])
log = logging.info

try:
    duration, batch, size = float(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
    if not math.isfinite(duration) or duration <= 0 or batch <= 0 or size < 32:
        raise ValueError("DURATION > 0, BATCH > 0, IMAGE_SIZE >= 32 required")
    import torch
    from torch import nn
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable. Install a CUDA-enabled PyTorch build compatible with your driver/GPU.")

    torch.manual_seed(42)
    torch.set_num_threads(1)
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda:0")
    model = nn.Sequential(
        nn.Conv2d(3, 64, 3, stride=2, padding=1), nn.ReLU(),
        nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.ReLU(),
        nn.Conv2d(128, 256, 3, stride=2, padding=1), nn.ReLU(),
        nn.Conv2d(256, 512, 3, stride=2, padding=1), nn.ReLU(),
        nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(),
        nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(512, 1000)
    ).to(device).train()
    # Reuse GPU-resident synthetic data to avoid disk/CPU data-loading bottlenecks.
    x = torch.randn(batch, 3, size, size, device=device)
    y = torch.randint(0, 1000, (batch,), device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    def train_step():
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
        return loss

    log("GPU: %s | PyTorch: %s | CUDA runtime: %s | FP32", torch.cuda.get_device_name(0), torch.__version__, torch.version.cuda)
    log("Warmup: 5 training steps (excluded from throughput)")
    for _ in range(5):
        train_step()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    steps = 0
    start = last_report = time.perf_counter()
    elapsed = 0.0
    while elapsed < duration:
        loss = train_step()
        # CUDA is asynchronous: wait for each completed training step.
        torch.cuda.synchronize()
        steps += 1
        now = time.perf_counter()
        elapsed = now - start
        if now - last_report >= 5:
            log("elapsed=%.1fs | steps=%d | images/s=%.1f | allocated=%.0f MiB | loss=%.4f",
                elapsed, steps, steps * batch / elapsed,
                torch.cuda.memory_allocated() / 2**20, loss.item())
            last_report = now
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    log("\nTraining result: steps=%d | time=%.2fs | images/s=%.1f | ms/step=%.2f",
        steps, elapsed, steps * batch / elapsed, elapsed * 1000 / steps)
    log("PyTorch peak allocated=%.1f MiB | peak reserved=%.1f MiB",
        torch.cuda.max_memory_allocated() / 2**20,
        torch.cuda.max_memory_reserved() / 2**20)
except KeyboardInterrupt:
    log("Interrupted")
    sys.exit(130)
except Exception as exc:
    logging.error("Benchmark failed: %s", exc)
    if "out of memory" in str(exc).lower():
        logging.error("Reduce BATCH (e.g. BATCH=8) or IMAGE_SIZE (e.g. IMAGE_SIZE=128).")
    sys.exit(1)
PY
TRAIN_PID=$!
STATUS=0
wait "$TRAIN_PID" || STATUS=$?
TRAIN_PID=""
if ! kill -0 "$MON_PID" 2>/dev/null; then
    echo "Monitor exited early; check $OUT_DIR/monitor.log" >&2
    STATUS=1
fi
cleanup
MON_PID=""

# GPU-wide sampled statistics, including initialization and warmup.
awk -F, '
NR > 1 {
    for (i=1; i<=NF; i++) gsub(/^[[:space:]]+|[[:space:]]+$/, "", $i)
    if ($2 ~ /^[0-9]+([.][0-9]+)?$/) {u=$2+0; n++; sum+=u; if (u>umax) umax=u}
    if ($3 ~ /^[0-9]+([.][0-9]+)?$/) {mem=$3+0; total=$4+0; m++; if (mem>mmax) mmax=mem; if (total>0 && mem/total*100>mpct) mpct=mem/total*100}
}
END {
    if (n) printf "GPU utilization (whole run): avg=%.1f%% | peak=%.1f%%\n", sum/n, umax
    if (m) printf "GPU-wide sampled VRAM peak: %.0f MiB (%.1f%% of total)\n", mmax, mpct
}' "$OUT_DIR/gpu.csv"
echo "Saved: $OUT_DIR/gpu.csv, $OUT_DIR/train.log, $OUT_DIR/monitor.log"
exit "$STATUS"
