param(
    [ValidateSet(15, 20, 25, 30, 40, 60)]
    [int]$Lead = 20,
    [int]$Width = 48,
    [ValidateSet('fan', 'tcn')]
    [string]$MftFeatures = 'tcn',
    [ValidateSet('power_cwt', 'wind_cwt')]
    [string]$InputLayout = 'power_cwt',
    [ValidateSet('strip', 'original')]
    [string]$CwtEncoder = 'strip',
    [ValidateSet('cafr', 'gated')]
    [string]$Fusion = 'cafr',
    [ValidateSet('none', 'lstm', 'bilstm', 'bigru')]
    [string]$TemporalRnn = 'none'
)

$conda = 'D:\anaconda3\Scripts\conda.exe'
$environment = 'D:\anaconda3\envs\windpower'
$entryPoint = Join-Path $PSScriptRoot 'train_multiheight_short.py'
$trainingArgs = @(
    $entryPoint, '--leads', $Lead, '--width', $Width,
    '--mft-features', $MftFeatures, '--input-layout', $InputLayout,
    '--cwt-encoder', $CwtEncoder, '--fusion', $Fusion,
    '--temporal-rnn', $TemporalRnn, '--ours-only'
)

if ((Test-Path -LiteralPath $conda -PathType Leaf) -and
    (Test-Path -LiteralPath (Join-Path $environment 'python.exe') -PathType Leaf)) {
    & $conda run --no-capture-output -p $environment python @trainingArgs
} else {
    & python @trainingArgs
}
exit $LASTEXITCODE
