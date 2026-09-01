#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT=/Users/wwiras/Documents/src/AHBNProj/ahbn/v0.63
PYTHON=/Users/wwiras/Documents/src/AHBNProj/venv0.6/bin/python
[ "$#" -eq 1 ] || { echo "usage: $0 outputs/stage4_exp07_v063-YYYYMMDD_HHMMSS" >&2; exit 2; }
[ -d "${PROJECT_ROOT}" ] || { echo "project root does not exist: ${PROJECT_ROOT}" >&2; exit 1; }
[ -x "${PYTHON}" ] || { echo "required Python is not executable: ${PYTHON}" >&2; exit 1; }
case "$1" in
  /*) FORMAL_OUTPUT="$1" ;;
  *) FORMAL_OUTPUT="${PROJECT_ROOT}/$1" ;;
esac
[ -d "${FORMAL_OUTPUT}" ] || { echo "formal output directory does not exist: ${FORMAL_OUTPUT}" >&2; exit 1; }
[ -f "${FORMAL_OUTPUT}/terminal.log" ] || { echo "missing formal terminal log" >&2; exit 1; }
grep -q 'Stage 4 exp07 ControlSim v0.63 formal' "${FORMAL_OUTPUT}/terminal.log" || { echo "dataset is not a formal Exp07 output" >&2; exit 1; }
[ -f "${FORMAL_OUTPUT}/exp07_v063_summary.csv" ] || { echo "missing Exp07 summary CSV" >&2; exit 1; }
[ -f "${FORMAL_OUTPUT}/exp07_v063_ahbn_adaptive_summary.csv" ] || { echo "missing AHBN adaptive summary CSV" >&2; exit 1; }

PLOT_LOG="${FORMAL_OUTPUT}/exp07_v063_figure_generation_terminal.log"
TRANSCRIPT_DOC="${PROJECT_ROOT}/docs/stage4_exp07v0.63_rerun.md"
exec > >(tee -a "${PLOT_LOG}" "${TRANSCRIPT_DOC}") 2>&1
trap 'status=$?; echo "FIGURE GENERATION EXIT CODE: ${status}"; exit "${status}"' EXIT

echo "EXP07 FORMAL FIGURE GENERATION"
echo "dataset: ${FORMAL_OUTPUT}"
echo "python: ${PYTHON}"
cd "${PROJECT_ROOT}"
"${PYTHON}" scripts/plot_stage4_exp07_v063.py "${FORMAL_OUTPUT}"
