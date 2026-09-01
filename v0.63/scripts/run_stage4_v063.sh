#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT=/Users/wwiras/Documents/src/AHBNProj/ahbn/v0.63
PYTHON=/Users/wwiras/Documents/src/AHBNProj/venv0.6/bin/python
EXPERIMENT="${1:?usage: run_stage4_v063.sh exp07|exp08|exp09 smoke|formal}"
RUN_KIND="${2:?usage: run_stage4_v063.sh exp07|exp08|exp09 smoke|formal}"
case "${EXPERIMENT}:${RUN_KIND}" in
  exp07:smoke|exp07:formal|exp08:smoke|exp08:formal|exp09:smoke|exp09:formal) ;;
  *) echo "invalid experiment/run kind: ${EXPERIMENT}:${RUN_KIND}" >&2; exit 2 ;;
esac

[ -x "${PYTHON}" ] || { echo "required Python is not executable: ${PYTHON}" >&2; exit 1; }
case "${EXPERIMENT}" in
  exp07) CONFIG_NAME=exp07_fanout.yaml; RERUN_DOC=stage4_exp07v0.63_rerun.md ;;
  exp08) CONFIG_NAME=exp08_ch_bottleneck.yaml; RERUN_DOC=stage4_exp08v0.63_rerun.md ;;
  exp09) CONFIG_NAME=exp09_dense_topology.yaml; RERUN_DOC=stage4_exp09v0.63_rerun.md ;;
esac
STAMP="$(date +%Y%m%d_%H%M%S)"
OUTPUT_ROOT="${PROJECT_ROOT}/outputs/stage4_${EXPERIMENT}_v063-${STAMP}"
mkdir -p "${OUTPUT_ROOT}"
TERMINAL_LOG="${OUTPUT_ROOT}/terminal.log"
TRANSCRIPT_DOC="${PROJECT_ROOT}/docs/${RERUN_DOC}"
exec > >(tee -a "${TERMINAL_LOG}" "${TRANSCRIPT_DOC}") 2>&1
trap 'status=$?; echo "EXIT CODE: ${status}"; echo "OUTPUT DIRECTORY: ${OUTPUT_ROOT}"; exit "${status}"' EXIT

CONFIG="${PROJECT_ROOT}/configs/${CONFIG_NAME}"
if [ "${RUN_KIND}" = smoke ]; then
  CONFIG="${OUTPUT_ROOT}/${CONFIG_NAME%.yaml}_smoke.yaml"
  "${PYTHON}" - "${PROJECT_ROOT}/configs/${CONFIG_NAME}" "${CONFIG}" <<'PY'
import sys
from pathlib import Path
import yaml

source, destination = map(Path, sys.argv[1:])
cfg = yaml.safe_load(source.read_text(encoding="utf-8"))
cfg["runs_per_setting"] = 1
cfg["seed"] = 42
destination.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
PY
fi

echo "Stage 4 ${EXPERIMENT} ControlSim v0.63 ${RUN_KIND}"
echo "Command: cd ${PROJECT_ROOT}"
echo "Command: bash scripts/run_stage4_${EXPERIMENT}_v063_${RUN_KIND}.sh"
echo "Python: ${PYTHON}"
echo "Config: ${CONFIG}"
echo "Output directory: ${OUTPUT_ROOT}"
if [ "${EXPERIMENT}:${RUN_KIND}" = exp08:formal ]; then
  "${PYTHON}" - "${CONFIG}" <<'PY'
import sys
from pathlib import Path
import yaml

cfg = yaml.safe_load(Path(sys.argv[1]).read_text(encoding="utf-8"))
strategies = list(cfg["strategies"])
overloads = [float(value) for value in cfg["ch_overload_factor"]]
runs = int(cfg["runs_per_setting"])
print("EXP08 FORMAL RUN")
print(f"project root: /Users/wwiras/Documents/src/AHBNProj/ahbn/v0.63")
print(f"treatments: {strategies}")
print(f"overload factors: {overloads}")
print(f"runs per cell: {runs}")
print(f"expected run count: {len(strategies) * len(overloads) * runs}")
PY
fi
cd "${OUTPUT_ROOT}"
"${PYTHON}" "${PROJECT_ROOT}/run_batch.py" --config "${CONFIG}"
"${PYTHON}" "${PROJECT_ROOT}/scripts/analyze_stage4_v063.py" \
  --root "${OUTPUT_ROOT}" --experiment "${EXPERIMENT}"
echo "TECHNICAL VALIDATION: PASS"
