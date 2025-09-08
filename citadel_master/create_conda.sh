#!/usr/bin/env bash
set -euo pipefail

# ─── Settings ──────────────────────────────────────────────────
ENV_NAME='master'               # conda env name
PY_VERSION='3.13.2'             # create …python=$PY_VERSION
MINICONDA_URL='https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh'
MINICONDA_DIR="${HOME}/miniconda3"
# ───────────────────────────────────────────────────────────────

# Determine working directory (where this script resides)
WORK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONDA_EXE="${MINICONDA_DIR}/bin/conda"

# --- 1. Ensure Miniconda is present ------------------------------------------
if [ ! -x "${CONDA_EXE}" ]; then
    echo "→ Installing Miniconda to ${MINICONDA_DIR} …"
    TMP_INSTALLER="$(mktemp)"
    curl -fsSL "${MINICONDA_URL}" -o "${TMP_INSTALLER}"
    bash "${TMP_INSTALLER}" -b -p "${MINICONDA_DIR}"
    rm -f "${TMP_INSTALLER}"
fi

# --- 2. Enable conda in this shell -------------------------------------------
# Load conda functions
# shellcheck disable=SC1090
source "${MINICONDA_DIR}/etc/profile.d/conda.sh"
# Disable auto activation of base
conda config --set auto_activate_base false > /dev/null

# --- 3. Create / update the 'master' env --------------------------------------
if ! conda env list | awk '{print $1}' | grep -qw "${ENV_NAME}"; then
    echo "→ Creating conda env '${ENV_NAME}' (Python ${PY_VERSION})…"
    conda create -y -n "${ENV_NAME}" python="${PY_VERSION}"
fi

echo "→ Installing requirements into '${ENV_NAME}'…"
conda run -n "${ENV_NAME}" python -m pip install --upgrade pip
conda run -n "${ENV_NAME}" python -m pip install --no-cache-dir -r "${WORK_DIR}/requirements.txt"
