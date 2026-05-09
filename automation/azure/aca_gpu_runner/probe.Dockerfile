FROM nvidia/cuda:12.4.1-base-ubuntu22.04

CMD ["/bin/bash", "-lc", "set -euo pipefail; nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader | tee /tmp/gpu_inventory.txt; awk -F, 'BEGIN { ok = 0 } NF >= 2 { gsub(/ /, \"\", $2); if (($2 + 0) >= 7.5) ok = 1 } END { exit ok ? 0 : 1 }' /tmp/gpu_inventory.txt"]
