param(
    [ValidatePattern('^[A-Za-z]$')][string]$Drive = 'R',
    [ValidateRange(1, 2)][int]$Jobs = 2
)

$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent $PSScriptRoot
$Mapped = $false
$Pushed = $false
if (Test-Path "${Drive}:/") { throw "Drive ${Drive}: is already in use" }
$CMake = (Get-Command cmake.exe -ErrorAction Stop).Source

function Invoke-Checked([string]$Executable, [string[]]$Arguments) {
    & $Executable @Arguments
    if ($LASTEXITCODE -ne 0) { throw "$Executable exited with $LASTEXITCODE" }
}

try {
    # Build tools use an ASCII drive alias while the checkout retains its name.
    Invoke-Checked 'subst.exe' @("${Drive}:", $Root)
    $Mapped = $true
    Push-Location "${Drive}:/"
    $Pushed = $true
    $Base = "${Drive}:/outputs/toolchain"
    $Source = "$Base/systemc-src"
    if (-not (Test-Path -LiteralPath "$Source/CMakeLists.txt")) {
        Invoke-Checked 'git.exe' @('clone', '--depth', '1', '--branch', '3.0.2', 'https://github.com/accellera-official/systemc.git', $Source)
    }
    $Revision = & git.exe -C $Source rev-parse HEAD
    if ($LASTEXITCODE -ne 0 -or $Revision -ne '70b0fc8e4a74acc677b0fc73cea08f940c2115d5') {
        throw 'SystemC source must match the verified 3.0.2 release commit'
    }
    $Common = @('-G', 'Visual Studio 17 2022', '-A', 'x64', '-DCMAKE_CXX_STANDARD=17')
    Invoke-Checked $CMake (@('-S', $Source, '-B', "$Base/systemc-msvc-build") + $Common + @('-DBUILD_SHARED_LIBS=OFF', '-DENABLE_EXAMPLES=OFF', "-DCMAKE_INSTALL_PREFIX=$Base/systemc-msvc-install", '-DBUILD_SOURCE_DOCUMENTATION=OFF'))
    Invoke-Checked $CMake @('--build', "$Base/systemc-msvc-build", '--config', 'Release', '--parallel', "$Jobs")
    Invoke-Checked $CMake @('--install', "$Base/systemc-msvc-build", '--config', 'Release')
    Invoke-Checked $CMake (@('-S', "${Drive}:/hardware_sim/systemc", '-B', "$Base/traffic-systemc-msvc-build") + $Common + @("-DCMAKE_PREFIX_PATH=$Base/systemc-msvc-install"))
    Invoke-Checked $CMake @('--build', "$Base/traffic-systemc-msvc-build", '--config', 'Release', '--parallel', "$Jobs")
    Write-Output "Built: $Root/outputs/toolchain/traffic-systemc-msvc-build/Release/npu_queue_sim.exe"
}
finally {
    if ($Pushed) { Pop-Location }
    if ($Mapped) { & subst.exe "${Drive}:" /d }
}
