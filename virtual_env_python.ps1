$ErrorActionPreference = "Stop"

Set-Location $PSScriptRoot

$VenvName = ".venv"
$VenvPath = Join-Path $PSScriptRoot $VenvName
$RequirementsPath = Join-Path $PSScriptRoot "requirements.txt"

if (Get-Command py -ErrorAction SilentlyContinue) {
    $PythonCommand = "py"
}
elseif (Get-Command python -ErrorAction SilentlyContinue) {
    $PythonCommand = "python"
}
else {
    throw "Python was not found in PATH. Please install Python and try again."
}

Write-Host "Using Python command: $PythonCommand"

if (-not (Test-Path $VenvPath)) {
    Write-Host "Creating virtual environment in $VenvPath..."
    if ($PythonCommand -eq "py") {
        & py -3 -m venv $VenvPath
    }
    else {
        & python -m venv $VenvPath
    }
}

$ActivateScript = Join-Path $VenvPath "Scripts\Activate.ps1"
if (-not (Test-Path $ActivateScript)) {
    throw "Activation script not found at $ActivateScript. The virtual environment may be incomplete."
}

Write-Host "Activating virtual environment..."
& $ActivateScript

Write-Host "Upgrading pip..."
python -m pip install --upgrade pip

if (Test-Path $RequirementsPath) {
    Write-Host "Installing dependencies from $RequirementsPath..."
    python -m pip install -r $RequirementsPath
}
else {
    Write-Host "requirements.txt not found at $RequirementsPath. Skipping dependency installation."
}

Write-Host "Virtual environment is ready."
