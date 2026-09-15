param([string]$Output = 'results/reproduction_best_checkpoint')
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath (Split-Path -Parent $PSScriptRoot)
conda activate aspire
python reproduce.py --report-only --output $Output
exit $LASTEXITCODE
