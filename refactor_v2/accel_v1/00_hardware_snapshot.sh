#!/bin/bash
# Read-only snapshot of Mac hardware, power, memory, disk and Python runtimes.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"; OUT="$HERE/hardware_snapshot.txt"
{
echo "=== date"; date
echo "=== macOS"; sw_vers
echo "=== hardware"; system_profiler SPHardwareDataType SPDisplaysDataType SPPowerDataType 2>/dev/null
echo "=== cpu"; sysctl -n machdep.cpu.brand_string; sysctl hw.ncpu hw.perflevel0.physicalcpu hw.perflevel1.physicalcpu hw.memsize
echo "=== power"; pmset -g; pmset -g therm; pmset -g batt
echo "=== memory"; vm_stat; sysctl vm.swapusage; memory_pressure 2>/dev/null | tail -3
echo "=== disk"; df -h / /Users/uyen/Downloads/BONEVQA; ls /Volumes
echo "=== top processes"; top -l 1 -o cpu -n 15 -stats pid,command,cpu,mem
echo "=== python runtimes"
for P in /Users/uyen/Downloads/BoneMedVQA/.venv/bin/python /Users/uyen/Downloads/BONEVQA/.evaluation_runtime/lens_venv_v2/bin/python /Users/uyen/Downloads/BONEVQA/.evaluation_runtime/accel_venv_v1/bin/python; do
  [ -x "$P" ] && "$P" -c "import sys,platform,torch;print(sys.executable,platform.python_version(),'torch',torch.__version__,'mps',torch.backends.mps.is_available())"
done
} > "$OUT" 2>&1
echo "Wrote $OUT"
