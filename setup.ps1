$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
if (-not (Test-Path -LiteralPath '.venv\Scripts\python.exe')) {
    python -m venv .venv
    if ($LASTEXITCODE -ne 0) { throw 'Falha ao criar o ambiente Python.' }
}
& .\.venv\Scripts\python.exe -m pip install -r requirements.txt --cache-dir .cache/pip
if ($LASTEXITCODE -ne 0) { throw 'Falha na instalação das dependências.' }
& .\.venv\Scripts\python.exe check_gpu.py
if ($LASTEXITCODE -ne 0) { throw 'CUDA não passou na validação.' }
Write-Host 'Ambiente pronto. Consulte README.md para os próximos comandos.'
