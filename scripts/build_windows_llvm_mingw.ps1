param(
    [string]$ToolchainBin = "",
    [string]$Configuration = "Release"
)

$ErrorActionPreference = "Stop"

$Repo = Split-Path -Parent $PSScriptRoot
$BuildDir = Join-Path $Repo "build-windows"
New-Item -ItemType Directory -Force -Path $BuildDir | Out-Null

if ([string]::IsNullOrWhiteSpace($ToolchainBin)) {
    $clang = Get-Command clang++.exe -ErrorAction SilentlyContinue
    if ($clang) {
        $ToolchainBin = Split-Path -Parent $clang.Source
    } else {
        $default = "C:\Users\User\cppdev\llvm-mingw-20260421-ucrt-x86_64\bin"
        if (Test-Path -LiteralPath (Join-Path $default "clang++.exe")) {
            $ToolchainBin = $default
        }
    }
}

if ([string]::IsNullOrWhiteSpace($ToolchainBin) -or -not (Test-Path -LiteralPath (Join-Path $ToolchainBin "clang++.exe"))) {
    throw "clang++.exe not found. Pass -ToolchainBin C:\path\to\llvm-mingw\bin or put clang++.exe on PATH."
}

$env:PATH = "$ToolchainBin;$env:PATH"
$clangxx = Join-Path $ToolchainBin "clang++.exe"
$flags = @("-std=c++17", "-O2", "-Iinclude")

Push-Location $Repo
try {
    & $clangxx @flags "tailslayer_example.cpp" "-o" (Join-Path $BuildDir "tailslayer_example.exe")
    if ($LASTEXITCODE -ne 0) { throw "tailslayer_example compile failed with exit $LASTEXITCODE" }

    & $clangxx @flags "experiments\transformer_tailslayer\transformer_tailslayer_bench.cpp" "-o" (Join-Path $BuildDir "transformer_tailslayer_bench.exe")
    if ($LASTEXITCODE -ne 0) { throw "transformer_tailslayer_bench compile failed with exit $LASTEXITCODE" }

    Write-Host "Built:"
    Write-Host "  $BuildDir\tailslayer_example.exe"
    Write-Host "  $BuildDir\transformer_tailslayer_bench.exe"
} finally {
    Pop-Location
}
