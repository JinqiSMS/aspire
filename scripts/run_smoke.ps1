$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath (Split-Path -Parent $PSScriptRoot)
conda activate aspire
python -c "import sys; assert sys.version_info[:2] == (3, 10), 'Activate the Python 3.10 aspire environment'"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
python reproduce.py --preset smoke
exit $LASTEXITCODE
