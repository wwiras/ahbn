#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT=/Users/wwiras/Documents/src/AHBNProj/ahbn/v0.63
PYTHON=/Users/wwiras/Documents/src/AHBNProj/venv0.6/bin/python
[ "$#" -eq 1 ] || { echo "usage: $0 outputs/stage4_exp07_v063-YYYYMMDD_HHMMSS" >&2; exit 2; }
case "$1" in
  /*) FORMAL_OUTPUT="$1" ;;
  *) FORMAL_OUTPUT="${PROJECT_ROOT}/$1" ;;
esac
[ -d "${FORMAL_OUTPUT}" ] || { echo "formal output directory does not exist: ${FORMAL_OUTPUT}" >&2; exit 1; }
[ -x "${PYTHON}" ] || { echo "required Python is not executable: ${PYTHON}" >&2; exit 1; }
TERMINAL_LOG="${FORMAL_OUTPUT}/exp07_v063_analysis_terminal.log"
TRANSCRIPT_DOC="${PROJECT_ROOT}/docs/stage4_exp07v0.63_rerun.md"
exec > >(tee -a "${TERMINAL_LOG}" "${TRANSCRIPT_DOC}") 2>&1
trap 'status=$?; echo "ANALYSIS EXIT CODE: ${status}"; exit "${status}"' EXIT
cd "${PROJECT_ROOT}"
echo "Analysis command: bash scripts/run_stage4_exp07_v063_analysis.sh ${1}"
echo "Selected dataset: ${FORMAL_OUTPUT}"
"${PYTHON}" scripts/analyze_stage4_exp07_v063.py "${FORMAL_OUTPUT}"
