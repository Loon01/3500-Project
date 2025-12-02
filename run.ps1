# PowerShell script to run the Java OOP ML Library

Write-Host "Running Java OOP ML Library..." -ForegroundColor Green
Write-Host ""

# Check if compiled
if (-not (Test-Path "bin\Main.class")) {
    Write-Host "Program not compiled yet. Running compile script..." -ForegroundColor Yellow
    .\compile.ps1
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Compilation failed. Please fix errors first." -ForegroundColor Red
        exit 1
    }
}

# Check if data file exists
if (-not (Test-Path "adult_income_cleaned.csv")) {
    Write-Host "WARNING: adult_income_cleaned.csv not found!" -ForegroundColor Red
    Write-Host "Please copy the dataset file to this directory." -ForegroundColor Yellow
    Write-Host ""
    
    # Try to copy from parent directory
    if (Test-Path "..\adult_income_cleaned.csv") {
        Write-Host "Found dataset in parent directory. Copying..." -ForegroundColor Yellow
        Copy-Item "..\adult_income_cleaned.csv" "adult_income_cleaned.csv"
        Write-Host "Dataset copied successfully!" -ForegroundColor Green
        Write-Host ""
    }
}

# Run the program
java -cp bin Main
