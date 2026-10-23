#!/usr/bin/env bash
set -euo pipefail

if [[ $# -gt 1 ]]; then
  echo "Usage: run_mfa.sh [config.json]" >&2
  exit 1
fi

CONFIG_PATH="${1:-}"

if ! command -v jq >/dev/null 2>&1; then
  echo "jq is required but was not found on PATH" >&2
  exit 1
fi

if [[ -n "$CONFIG_PATH" && ! -f "$CONFIG_PATH" ]]; then
  echo "Config file not found: $CONFIG_PATH" >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
PROJECT_CONFIG="$ROOT_DIR/src/package_data/config.json"

if [[ ! -f "$PROJECT_CONFIG" ]]; then
  echo "Project config not found: $PROJECT_CONFIG" >&2
  exit 1
fi

if [[ -n "$CONFIG_PATH" ]]; then
  MFA_ENV_NAME="$(jq -r '.mfa_env_name // "mfa"' "$CONFIG_PATH")"
else
  MFA_ENV_NAME="mfa"
fi

# split_utterances.py writes split utterances and MFA reads/writes its own
# outputs under this same project-configured directory, so both are derived
# from it instead of being duplicated in this script's own config.
MFA_OUTPUT_DIR="$(jq -r '.mfa_output_dir // empty' "$PROJECT_CONFIG")"

if [[ -z "$MFA_OUTPUT_DIR" ]]; then
  echo "Project config is missing required field 'mfa_output_dir': $PROJECT_CONFIG" >&2
  exit 1
fi

if [[ "$MFA_OUTPUT_DIR" != /* ]]; then
  MFA_OUTPUT_DIR="$ROOT_DIR/$MFA_OUTPUT_DIR"
fi

SPLIT_ROOT="$MFA_OUTPUT_DIR/split_utterances"
OUTPUT_DIR="$MFA_OUTPUT_DIR/raw_mfa_outputs"

MFA_DICTIONARY="$(jq -r '.mfa_dictionary // empty' "$PROJECT_CONFIG")"
MFA_ACOUSTIC_MODEL="$(jq -r '.mfa_acoustic_model // empty' "$PROJECT_CONFIG")"

if [[ -z "$MFA_DICTIONARY" ]]; then
  echo "Project config is missing required field 'mfa_dictionary': $PROJECT_CONFIG" >&2
  exit 1
fi

if [[ -z "$MFA_ACOUSTIC_MODEL" ]]; then
  echo "Project config is missing required field 'mfa_acoustic_model': $PROJECT_CONFIG" >&2
  exit 1
fi

if ! command -v conda >/dev/null 2>&1; then
  echo "conda is required but was not found on PATH" >&2
  exit 1
fi

CONDA_BASE="$(conda info --base)"
# shellcheck disable=SC1091
source "$CONDA_BASE/etc/profile.d/conda.sh"

if ! conda env list | awk '{print $1}' | grep -Fxq "$MFA_ENV_NAME"; then
  echo "MFA environment '$MFA_ENV_NAME' does not exist" >&2
  exit 1
fi

if [[ ! -d "$SPLIT_ROOT" ]]; then
  echo "Split utterance root not found: $SPLIT_ROOT" >&2
  exit 1
fi

mkdir -p "$OUTPUT_DIR"

set +u
conda activate "$MFA_ENV_NAME"
set -u
trap 'set +u; conda deactivate' EXIT

for utterance_dir in "$SPLIT_ROOT"/*; do
  if [[ ! -d "$utterance_dir" ]]; then
    continue
  fi

  echo "Running MFA for $(basename "$utterance_dir")"
  # --no_cleanup_textgrids: MFA's default post-processing strips aligned
  # silence phones from the exported tiers and backfills a gap up to the
  # clip's end. For clips cropped tight to their transcript boundary, that
  # gap can be zero, which trips an assertion (`assert begin < end`) deep in
  # kalpy's TextGrid export. Skipping the cleanup avoids the crash entirely.
  mfa align --clean --no_cleanup_textgrids "$utterance_dir" "$MFA_DICTIONARY" "$MFA_ACOUSTIC_MODEL" "$OUTPUT_DIR/$(basename "$utterance_dir")"
 done
