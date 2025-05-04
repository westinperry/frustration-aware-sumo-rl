#!/usr/bin/env bash
set -uo pipefail  # Do not use -e to avoid stopping on errors

OUT_CSV="eval/results.csv"
echo "model,mean_reward" > "$OUT_CSV"

for model in models/*.zip; do
  echo "Evaluating $model…"

  # Run evaluation and capture output
  if output=$(python eval/evaluate_policy.py --model "$model" 2>&1); then
    # Extract mean reward from line: "Mean Reward over 1 episodes: -2797.5 ± 0.0"
    mean=$(echo "$output" | grep -i "Mean Reward over" | sed -E 's/.*: ([^ ±]+).*/\1/')
    if [[ -z "$mean" ]]; then
      mean="NA"
    fi
  else
    echo "⚠️  Error evaluating $model"
    mean="ERROR"
  fi

  echo "$(basename "$model"),$mean" >> "$OUT_CSV"
  echo
done

echo "✅ All evaluations complete. Results saved to $OUT_CSV"
