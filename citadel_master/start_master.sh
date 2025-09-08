#!/usr/bin/env bash
set -euo pipefail

# ─── Parameters ──────────────────────────────────────────────────────────────
ENV_NAME="${1:-master}"   # default conda env name (override by passing as first arg)
# ─────────────────────────────────────────────────────────────────────────────

# Determine script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Conda executable
CONDA_EXE="${HOME}/miniconda3/bin/conda"

if [ ! -x "${CONDA_EXE}" ]; then
  echo "Error: Conda not found. Run create_conda.sh first." >&2
  exit 1
fi

echo "→ Running in current shell in '${SCRIPT_DIR}' (env '${ENV_NAME}') …"

# Load conda and activate env
source "${HOME}/miniconda3/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"

# Switch to script dir and set PYTHONPATH
cd "${SCRIPT_DIR}"
export PYTHONPATH="../:../proto/python"

# Start the worker server
python "./start_master_server.py"

# When the server exits, drop into an interactive shell
exec bash
#!/usr/bin/env bash
set -euo pipefail

# ─── Parameters ──────────────────────────────────────────────────────────────
ENV_NAME="${1:-master}"   # default conda env name (override by passing as first arg)
# ─────────────────────────────────────────────────────────────────────────────

# Determine script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Conda executable
CONDA_EXE="${HOME}/miniconda3/bin/conda"

if [ ! -x "${CONDA_EXE}" ]; then
  echo "Error: Conda not found. Run create_conda.sh first." >&2
  exit 1
fi

echo "→ Running in current shell in '${SCRIPT_DIR}' (env '${ENV_NAME}') …"

# Load conda and activate env
source "${HOME}/miniconda3/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"

# Switch to script dir and set PYTHONPATH
cd "${SCRIPT_DIR}"
export PYTHONPATH="../:../proto/python"

# Start the worker server
python "./start_master_server.py"

# When the server exits, drop into an interactive shell
exec bash
