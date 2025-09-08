<#---------------------------------------------------------------------
.SYNOPSIS
  Initialise Citadel's PostgreSQL instance - roles, database, pgvector
  extension and schema.
  Assumes PostgreSQL 16 + pgvector binaries are already installed
  by the third-party bootstrap script.

.NOTES
  Run from an *elevated* PowerShell window.  Safe to re-run.
---------------------------------------------------------------------#>

param(
    [string]$PgSuperPass = 'NewCitadelPw!',   # super-user password (matches install)
    [string]$CitadelPass = 'NewCitadelPw!',   # citadel role password
    [string]$InstallRoot = 'C:\Postgres',
    [string]$SchemaFile  = (Join-Path $PSScriptRoot '..\traindata_schema.sql')
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
function Write-Info { param($m) Write-Host ">> $m" -ForegroundColor Cyan }
function Write-Warn { param($m) Write-Host "!! $m" -ForegroundColor Yellow }

# -------------------------------------------------------------------
# 1. Ensure PostgreSQL service is running
# -------------------------------------------------------------------
$svcName = 'postgresql-x64-16'
if (-not (Get-Service -Name $svcName -ErrorAction SilentlyContinue)) {
    throw "PostgreSQL service '$svcName' not found - run bootstrap-third-party.ps1 first."
}

Start-Service $svcName
Write-Info 'PostgreSQL service is running.'

# -------------------------------------------------------------------
# 2. Locate psql.exe for this session
# -------------------------------------------------------------------
$pgBin = "$InstallRoot\pgsql\bin"
if (-not ($env:PATH.Split(';') -contains $pgBin)) {
    $env:PATH = "$pgBin;$env:PATH"
}
$psqlExe = Join-Path $pgBin 'psql.exe'
if (-not (Test-Path $psqlExe)) {
    throw "psql.exe not found at $pgBin"
}

# -------------------------------------------------------------------
# 3. Create login role + database (idempotent)
# -------------------------------------------------------------------
$env:PGPASSWORD = $PgSuperPass

$sqlRole = @'
DO $$
BEGIN
  IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'citadel') THEN
    EXECUTE format('CREATE ROLE citadel LOGIN PASSWORD %L', '{0}');
  END IF;
END $$;
'@ -f $CitadelPass

& $psqlExe -U postgres -d postgres -v ON_ERROR_STOP=1 -c $sqlRole
if ($LASTEXITCODE -ne 0) { throw 'Failed to create/login role citadel.' }

$dbExists = & $psqlExe -U postgres -d postgres -tAc `
            "SELECT 1 FROM pg_database WHERE datname = 'citadel_db';"
if (-not $dbExists) {
    & $psqlExe -U postgres -d postgres -v ON_ERROR_STOP=1 `
        -c "CREATE DATABASE citadel_db OWNER citadel ENCODING 'UTF8';"
    Write-Info "Database 'citadel_db' created."
} else {
    Write-Info "Database 'citadel_db' already exists."
}

# -------------------------------------------------------------------
# 4. Enable pgvector extension in citadel_db
# -------------------------------------------------------------------
Write-Info 'Ensuring pgvector extension exists...'
& $psqlExe -U postgres -d citadel_db -v ON_ERROR_STOP=1 `
    -c "CREATE EXTENSION IF NOT EXISTS vector;"
if ($LASTEXITCODE -ne 0) { throw 'Failed to create pgvector extension.' }

# Switch to normal role afterwards
$env:PGPASSWORD = $CitadelPass

# -------------------------------------------------------------------
# 5. Apply traindata schema (safe to re-run)
# -------------------------------------------------------------------
if (-not (Test-Path $SchemaFile)) {
    Write-Warn "Schema file not found at $SchemaFile; skipping."
    return
}

$firstMatch = Select-String -Path $SchemaFile -Pattern '(?i)^\s*create\s+table\s+(\w+)' |
              Select-Object -First 1
if (-not $firstMatch) {
    Write-Info 'Applying schema (no tables detected for idempotence test)...'
    $schemaResult = & $psqlExe -U citadel -d citadel_db -v ON_ERROR_STOP=1 -f $SchemaFile 2>&1
    if ($LASTEXITCODE -ne 0) {
        # Check if the error is actually just NOTICEs (which are not real errors)
        $schemaOutput = $schemaResult | Out-String
        if ($schemaOutput -match 'NOTICE:' -and $schemaOutput -notmatch 'ERROR:' -and $schemaOutput -notmatch 'FATAL:') {
            Write-Info 'Schema applied successfully (PostgreSQL NOTICEs are normal).'
        } else {
            throw "Failed to apply schema: $schemaResult"
        }
    } else {
        Write-Info 'Schema applied successfully.'
    }
} else {
    $firstTable  = $firstMatch.Matches[0].Groups[1].Value
    $tableExists = & $psqlExe -U citadel -d citadel_db -tAc `
                    "SELECT 1 FROM information_schema.tables WHERE table_name = '$firstTable';"
    if ($tableExists) {
        Write-Info 'Schema already present; skipping traindata_schema.sql.'
    } else {
        Write-Info 'Applying schema...'
        $schemaResult = & $psqlExe -U citadel -d citadel_db -v ON_ERROR_STOP=1 -f $SchemaFile 2>&1
        if ($LASTEXITCODE -ne 0) {
            # Check if the error is actually just NOTICEs (which are not real errors)
            $schemaOutput = $schemaResult | Out-String
            if ($schemaOutput -match 'NOTICE:' -and $schemaOutput -notmatch 'ERROR:' -and $schemaOutput -notmatch 'FATAL:') {
                Write-Info 'Schema applied successfully (PostgreSQL NOTICEs are normal).'
            } else {
                throw "Failed to apply schema: $schemaResult"
            }
        } else {
            Write-Info 'Schema applied successfully.'
        }
    }
}

Write-Info 'Citadel DB bootstrap finished.'
