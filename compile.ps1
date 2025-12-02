# PowerShell script to compile the Java OOP ML Library

Write-Host "Compiling Java OOP Implementation..." -ForegroundColor Green

# Create bin directory if it doesn't exist
if (-not (Test-Path "bin")) {
    New-Item -ItemType Directory -Path "bin" | Out-Null
    Write-Host "Created bin directory" -ForegroundColor Yellow
}

# Compile all Java files
javac -d bin `
    src\Main.java `
    src\models\Model.java `
    src\models\LinearRegression.java `
    src\models\LogisticRegression.java `
    src\models\KNearestNeighbors.java `
    src\models\DecisionTree.java `
    src\models\GaussianNaiveBayes.java `
    src\data\DataLoader.java `
    src\data\Dataset.java `
    src\data\DataPreprocessor.java `
    src\metrics\ClassificationMetrics.java `
    src\metrics\RegressionMetrics.java `
    src\utils\ArrayUtils.java `
    src\utils\MatrixUtils.java

if ($LASTEXITCODE -eq 0) {
    Write-Host "`nCompilation successful!" -ForegroundColor Green
    Write-Host "Run the program with: .\run.ps1" -ForegroundColor Cyan
} else {
    Write-Host "`nCompilation failed!" -ForegroundColor Red
    exit 1
}
