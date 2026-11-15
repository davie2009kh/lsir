#!/usr/bin/env bash
# Push queued Kaggle runs as GPU slots free up. 07b waits until 07a has COMPLETED (it needs 07a's output).
K=~/.local/bin/kaggle
cd "$(dirname "$0")/.."
pushed_07b=0; pushed_08f=0; pushed_08c=0
try() { out=$($K kernels push -p "$1" --accelerator NvidiaTeslaT4 2>&1); echo "$out" | grep -q "successfully pushed" && { echo "pushed $1"; return 0; }; return 1; }
for i in $(seq 1 120); do
  s7a=$($K kernels status davit2009/lsir-07a-synthetic-armenian-speech 2>&1 | grep -oE 'COMPLETE|ERROR|CANCEL')
  if [ "$pushed_07b" = 0 ] && [ "$s7a" = "COMPLETE" ]; then try kaggle/07b && pushed_07b=1; fi
  if [ "$s7a" = "ERROR" ] && [ "$pushed_07b" = 0 ]; then echo "07a failed; 07b not pushed"; pushed_07b=2; fi
  if [ "$pushed_08f" = 0 ]; then try kaggle/08fleurs && pushed_08f=1; fi
  if [ "$pushed_08f" = 1 ] && [ "$pushed_08c" = 0 ]; then try kaggle/08cv && pushed_08c=1; fi
  [ "$pushed_07b" != 0 ] && [ "$pushed_08f" = 1 ] && [ "$pushed_08c" = 1 ] && { echo "all queued runs pushed"; exit 0; }
  sleep 180
done
echo "queue timed out: 07b=$pushed_07b 08fleurs=$pushed_08f 08cv=$pushed_08c"
