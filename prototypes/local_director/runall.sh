#!/bin/bash
cd "$(dirname "$0")"
for m in qwen2.5-0.5b-instruct-q4_0 qwen2.5-1.5b-instruct-q4_k_m; do
  while true; do t=$(vcgencmd measure_temp | sed 's/[^0-9.]//g' | cut -d. -f1); [ "$t" -lt 65 ] && break; sleep 10; done
  echo "$(date +%T) start $m temp $(vcgencmd measure_temp)" >> results/progress.log
  python3 bench.py $m >> results/progress.log 2>&1
done
echo done >> results/progress.log
