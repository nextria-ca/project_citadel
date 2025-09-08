#!/usr/bin/env bash
# ---------------------------------------------------------------------
# Initialise Citadel's PostgreSQL instance on Linux
# Roles, database, pgvector extension, and schema.
# Assumes PostgreSQL 16 and pgvector extension are installed.
# Safe to re-run.
# ---------------------------------------------------------------------

set -euo pipefail

# Default parameters
PGSUPERPASS="NewCitadelPw!"
CITADELPASS="NewCitadelPw!"
INSTALLROOT="/usr/local/pgsql"
SCHEMAFILE="$(dirname "$0")/../traindata_schema.sql"

usage() {
  cat <<EOF
Usage: $(basename "$0") [-s superuser_password] [-c citadel_password] \
                           [-r install_root] [-f schema_file]

Options:
  -s    PostgreSQL superuser (postgres) password (default: \$PGSUPERPASS)
  -c    Citadel role password (default: \$CITADELPASS)
  -r    PostgreSQL install root (default: \$INSTALLROOT)
  -f    Path to traindata_schema.sql (default: \$SCHEMAFILE)
  -h    Show this help message
EOF
  exit 1
}

# Parse CLI options
do_getopts() {
  while getopts ":s:c:r:f:h" opt; do
    case "$opt" in
      s) PGSUPERPASS="$OPTARG" ;; 
      c) CITADELPASS="$OPTARG" ;; 
      r) INSTALLROOT="$OPTARG" ;; 
      f) SCHEMAFILE="$OPTARG" ;; 
      h) usage ;; 
      *) usage ;; 
    esac
  done
  shift $((OPTIND -1))
}
do_getopts "$@"

# Add psql to PATH if needed
export PATH="$INSTALLROOT/bin:$PATH"

# 1. Ensure PostgreSQL service is running
echo ">> Checking PostgreSQL service..."
if ! pg_isready -q; then
  echo ">> Starting PostgreSQL service..."
  if command -v systemctl &>/dev/null; then
    sudo systemctl start postgresql
  else
    sudo service postgresql start
  fi
else
  echo ">> PostgreSQL is already running."
fi

# 2. Check psql availability
if ! command -v psql &>/dev/null; then
  echo "!! psql not found in PATH (searched in $INSTALLROOT/bin)." >&2
  exit 1
fi

# Helper to run psql as postgres superuser
run_as_postgres() {
  PGPASSWORD="$PGSUPERPASS" psql -v ON_ERROR_STOP=1 -h localhost -U postgres "$@"
}

# Helper to run psql as citadel role
run_as_citadel() {
  PGPASSWORD="$CITADELPASS" psql -v ON_ERROR_STOP=1 -h localhost -U citadel "$@"
}

# 3. Create login role + database (idempotent)
echo ">> Ensuring citadel role exists..."
SQL_ROLE="DO \$\$
BEGIN
  IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'citadel') THEN
    EXECUTE format('CREATE ROLE citadel LOGIN PASSWORD %L', '$CITADELPASS');
  END IF;
END
\$\$;"
run_as_postgres -d postgres -c "$SQL_ROLE"

echo ">> Checking for database 'citadel_db'..."
DB_EXISTS=$(run_as_postgres -d postgres -tAc "SELECT 1 FROM pg_database WHERE datname = 'citadel_db';")
if [[ -z "$DB_EXISTS" ]]; then
  echo ">> Creating database 'citadel_db'..."
  run_as_postgres -d postgres -c "CREATE DATABASE citadel_db OWNER citadel ENCODING 'UTF8';"
else
  echo ">> Database 'citadel_db' already exists."
fi

# 4. Enable pgvector extension in citadel_db
echo ">> Ensuring pgvector extension exists in citadel_db..."
run_as_postgres -d citadel_db -c "CREATE EXTENSION IF NOT EXISTS vector;"

# 5. Apply traindata schema (safe to re-run)
if [[ ! -f "$SCHEMAFILE" ]]; then
  echo "!! Schema file not found at '$SCHEMAFILE'; skipping schema application." >&2
  exit 0
fi

# Detect first table name in schema
FIRST_TABLE=$(grep -Eoi '^[[:space:]]*create[[:space:]]+table[[:space:]]+([[:alnum:]_]+)' "$SCHEMAFILE" \
              | head -n1 | awk '{print \$3}')

if [[ -z "$FIRST_TABLE" ]]; then
  echo ">> No tables detected; applying full schema..."
  run_as_citadel -d citadel_db -f "$SCHEMAFILE"
else
  echo ">> Checking for existing table '$FIRST_TABLE'..."
  TABLE_EXISTS=$(run_as_citadel -d citadel_db -tAc \
    "SELECT 1 FROM information_schema.tables WHERE table_name = '$FIRST_TABLE';")

  if [[ -n "$TABLE_EXISTS" ]]; then
    echo ">> Schema already present; skipping traindata_schema.sql."
  else
    echo ">> Applying schema..."
    run_as_citadel -d citadel_db -f "$SCHEMAFILE"
  fi
fi

echo ">> Citadel DB bootstrap finished successfully."
