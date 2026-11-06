# Follow a training log live, showing only the key metrics per iteration.
# Usage: .\scripts\watch_train.ps1 [-Log train_gait_first.log]
param([string]$Log = "train_gait_first.log")

$pattern = 'Learning iteration|Steps per second|Mean reward|Mean episode length|' +
  'Episode_Reward/(air_time|single_foot_contact|track_linear_velocity|gait_schedule)|' +
  'Metrics/twist/error_vel_xy|Metrics/air_time_mean|Episode_Termination/fell_over|ETA|' +
  'Metrics/(double_support_frac|flight_frac|zmp_margin_mm|stance_roll_deg)'

$ansi = [char]27 + '\[[0-9;]*m'

Get-Content $Log -Tail 60 -Wait | ForEach-Object {
  $line = ($_ -replace $ansi, '').Trim()
  if ($line -match 'Learning iteration') {
    Write-Host ""
    Write-Host "==== $line ====" -ForegroundColor Cyan
  } elseif ($line -match $pattern) {
    Write-Host "  $line"
  }
}
