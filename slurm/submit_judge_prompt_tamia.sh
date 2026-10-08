#!/bin/bash
# judge_prompt_v1 on TamIA, small whole-node jobs (login node: bash slurm/submit_judge_prompt_tamia.sh)
#   4 encode groups (4 shards each, ~1 h)
#     -> geometry (all four readouts)
#     -> span seed 42 (4 vector cells) -> span seed 43, span seed 44
#     -> verdict seed 42 (2 cells) + zero-shot -> verdict seeds 43+44
# The step_tokens cell runs on Rorqual (slurm/submit_judge_prompt_rorqual.sh).
set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-$HOME/CoT-checker}"
RUN="${RUN:-$SCRATCH/cot_mech/judge_prompt_v1}"
cd "$PROJECT_ROOT"
[[ -e "$RUN/vec" && -n "$(ls -A "$RUN/vec" 2>/dev/null)" ]] && {
  echo "[FATAL] $RUN/vec is not empty; never submit a second DAG into it" >&2; exit 2; }
mkdir -p "$RUN/logs"
JOBS="$RUN/submitted_jobs_tamia.txt"; echo "source $(git rev-parse HEAD)" > "$JOBS"
sb() { sbatch --parsable --output="$RUN/logs/%x-%j.out" --export=ALL,RUN=$RUN,PROJECT_ROOT=$PROJECT_ROOT"${EXTRA:+,$EXTRA}" "$@"; }

enc=""
for g in 0 1 2 3; do
  j=$(EXTRA="GROUP=$g" sb --job-name=jp_enc$g slurm/judge_prompt_encode_tamia.sh)
  enc="$enc:$j"; echo "encode $g $j" | tee -a "$JOBS"
done
dep="--dependency=afterok$enc"
j=$(sb $dep --job-name=jp_geo slurm/judge_prompt_geometry_tamia.sh); echo "geometry $j" | tee -a "$JOBS"

span() { local s=$1; echo "span boundary_stats mlp:h1024x2 $s;span boundary_stats linear $s;span last_token linear $s;span step_mean linear $s"; }
p1=$(EXTRA="CELLS=$(span 42)" sb $dep --job-name=jp_span42 slurm/judge_prompt_cells_tamia.sh)
echo "span seed42 $p1" | tee -a "$JOBS"
for s in 43 44; do
  j=$(EXTRA="CELLS=$(span $s)" sb --dependency=afterok:$p1 --job-name=jp_span$s slurm/judge_prompt_cells_tamia.sh)
  echo "span seed$s $j" | tee -a "$JOBS"
done
v1=$(EXTRA="ZEROSHOT=1,CELLS=verdict last_token linear 42;verdict last_token mlp:h1024x2 42" \
     sb $dep --job-name=jp_verd42 slurm/judge_prompt_cells_tamia.sh)
echo "verdict seed42 + zeroshot $v1" | tee -a "$JOBS"
j=$(EXTRA="CELLS=verdict last_token linear 43;verdict last_token mlp:h1024x2 43;verdict last_token linear 44;verdict last_token mlp:h1024x2 44" \
    sb --dependency=afterok:$v1 --job-name=jp_verd4344 slurm/judge_prompt_cells_tamia.sh)
echo "verdict seeds43,44 $j" | tee -a "$JOBS"
