#!/usr/bin/env bash
set -Eeuo pipefail
PROJECT_ROOT=/Users/wwiras/Documents/src/AHBNProj/ahbn/v0.63
PYTHON=/Users/wwiras/Documents/src/AHBNProj/venv0.6/bin/python
[ "$#" -eq 1 ] || { echo "usage: $0 outputs/stage4_exp09_v063-YYYYMMDD_HHMMSS" >&2; exit 2; }
case "$1" in /*) FORMAL_OUTPUT="$1" ;; *) FORMAL_OUTPUT="${PROJECT_ROOT}/$1" ;; esac
[ -d "${FORMAL_OUTPUT}" ] || { echo "formal output does not exist: ${FORMAL_OUTPUT}" >&2; exit 1; }
[ -x "${PYTHON}" ] || { echo "required Python is not executable: ${PYTHON}" >&2; exit 1; }
[ -f "${FORMAL_OUTPUT}/exp09_v063_summary.csv" ] || { echo "missing Exp09 summary" >&2; exit 1; }
[ -f "${FORMAL_OUTPUT}/exp09_v063_ahbn_adaptive_summary.csv" ] || { echo "missing Exp09 adaptive summary" >&2; exit 1; }
LOG="${FORMAL_OUTPUT}/exp09_v063_figure_generation_terminal.log"
DOC="${PROJECT_ROOT}/docs/stage4_exp09v0.63_rerun.md"
exec > >(tee -a "${LOG}" "${DOC}") 2>&1
trap 'status=$?; echo "FIGURE GENERATION EXIT CODE: ${status}"; exit "${status}"' EXIT
echo "EXP09 FORMAL FIGURE GENERATION"
echo "dataset: ${FORMAL_OUTPUT}"
echo "python: ${PYTHON}"
cd "${PROJECT_ROOT}"
"${PYTHON}" scripts/plot_stage4_exp09_v063.py "${FORMAL_OUTPUT}"
