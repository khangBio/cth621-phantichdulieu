Add-Type -AssemblyName System.Windows.Forms.DataVisualization
Add-Type -AssemblyName System.Drawing

$root = $PSScriptRoot
$results = Join-Path $root 'outputs\ml_experiment\results'
$predictions = Import-Csv -LiteralPath (Join-Path $results 'test_predictions.csv')
$metrics = Import-Csv -LiteralPath (Join-Path $results 'test_metrics.csv') |
    Sort-Object { [double]$_.mae }, { [double]$_.rmse }
$best = $metrics[0]

$predictionColumn = switch ($best.model) {
    'Naive_Close_t' { 'pred_naive_close_t' }
    'LinearRegression' { 'pred_linear_regression' }
    'RandomForestRegressor' { 'pred_random_forest' }
    default { throw "Unknown model: $($best.model)" }
}

$chart = New-Object System.Windows.Forms.DataVisualization.Charting.Chart
$chart.Width = 1900
$chart.Height = 1050
$chart.BackColor = [System.Drawing.Color]::White
$chart.AntiAliasing = 'All'
$chart.TextAntiAliasingQuality = 'High'

$title = New-Object System.Windows.Forms.DataVisualization.Charting.Title
$title.Text = 'Actual vs Predicted Close - Best Model: Naive Persistence'
$title.Font = New-Object System.Drawing.Font('Segoe UI', 18, [System.Drawing.FontStyle]::Bold)
$chart.Titles.Add($title)
$subtitle = New-Object System.Windows.Forms.DataVisualization.Charting.Title
$subtitle.Text = ('Final test: MAE {0:N4} | RMSE {1:N4} | R2 {2:N6} | MAPE {3:N4}%' -f [double]$best.mae, [double]$best.rmse, [double]$best.r2, [double]$best.mape_percent)
$subtitle.Font = New-Object System.Drawing.Font('Segoe UI', 11)
$chart.Titles.Add($subtitle)

$scatterArea = New-Object System.Windows.Forms.DataVisualization.Charting.ChartArea('AllTest')
$scatterArea.Position = New-Object System.Windows.Forms.DataVisualization.Charting.ElementPosition(4, 12, 44, 79)
$scatterArea.AxisX.Title = 'Actual Close (log scale)'
$scatterArea.AxisY.Title = 'Predicted Close (log scale)'
$scatterArea.AxisX.IsLogarithmic = $true
$scatterArea.AxisY.IsLogarithmic = $true
$scatterArea.AxisX.Minimum = 1
$scatterArea.AxisY.Minimum = 1
$scatterArea.AxisX.Maximum = 2500
$scatterArea.AxisY.Maximum = 2500
$scatterArea.AxisX.MajorGrid.LineColor = [System.Drawing.Color]::FromArgb(228, 232, 238)
$scatterArea.AxisY.MajorGrid.LineColor = [System.Drawing.Color]::FromArgb(228, 232, 238)
$scatterArea.AxisX.LabelStyle.Font = New-Object System.Drawing.Font('Segoe UI', 9)
$scatterArea.AxisY.LabelStyle.Font = New-Object System.Drawing.Font('Segoe UI', 9)
$chart.ChartAreas.Add($scatterArea)

$scatterSeries = New-Object System.Windows.Forms.DataVisualization.Charting.Series('Test observations')
$scatterSeries.ChartArea = 'AllTest'
$scatterSeries.ChartType = 'Point'
$scatterSeries.MarkerStyle = 'Circle'
$scatterSeries.MarkerSize = 3
$scatterSeries.Color = [System.Drawing.Color]::FromArgb(80, 37, 99, 235)
for ($i = 0; $i -lt $predictions.Count; $i += 18) {
    $actual = [double]$predictions[$i].actual_close_next
    $predicted = [double]$predictions[$i].$predictionColumn
    if ($actual -gt 0 -and $predicted -gt 0) {
        [void]$scatterSeries.Points.AddXY($actual, $predicted)
    }
}
$chart.Series.Add($scatterSeries)

$identity = New-Object System.Windows.Forms.DataVisualization.Charting.Series('Perfect prediction')
$identity.ChartArea = 'AllTest'
$identity.ChartType = 'Line'
$identity.BorderWidth = 2
$identity.Color = [System.Drawing.Color]::FromArgb(210, 220, 38, 38)
$identity.Points.AddXY(1, 1) | Out-Null
$identity.Points.AddXY(2500, 2500) | Out-Null
$chart.Series.Add($identity)

$aaplArea = New-Object System.Windows.Forms.DataVisualization.Charting.ChartArea('AAPL')
$aaplArea.Position = New-Object System.Windows.Forms.DataVisualization.Charting.ElementPosition(53, 12, 43, 79)
$aaplArea.AxisX.Title = 'Target date'
$aaplArea.AxisY.Title = 'Close price'
$aaplArea.AxisX.LabelStyle.Format = 'MMM yyyy'
$aaplArea.AxisX.IntervalType = 'Months'
$aaplArea.AxisX.Interval = 2
$aaplArea.AxisX.MajorGrid.LineColor = [System.Drawing.Color]::FromArgb(228, 232, 238)
$aaplArea.AxisY.MajorGrid.LineColor = [System.Drawing.Color]::FromArgb(228, 232, 238)
$aaplArea.AxisX.LabelStyle.Font = New-Object System.Drawing.Font('Segoe UI', 9)
$aaplArea.AxisY.LabelStyle.Font = New-Object System.Drawing.Font('Segoe UI', 9)
$chart.ChartAreas.Add($aaplArea)

$actualSeries = New-Object System.Windows.Forms.DataVisualization.Charting.Series('AAPL actual')
$actualSeries.ChartArea = 'AAPL'
$actualSeries.ChartType = 'Line'
$actualSeries.BorderWidth = 3
$actualSeries.Color = [System.Drawing.Color]::FromArgb(37, 99, 235)
$actualSeries.XValueType = 'DateTime'

$predictedSeries = New-Object System.Windows.Forms.DataVisualization.Charting.Series('AAPL predicted')
$predictedSeries.ChartArea = 'AAPL'
$predictedSeries.ChartType = 'Line'
$predictedSeries.BorderWidth = 2
$predictedSeries.BorderDashStyle = 'Dash'
$predictedSeries.Color = [System.Drawing.Color]::FromArgb(220, 38, 38)
$predictedSeries.XValueType = 'DateTime'

foreach ($row in $predictions | Where-Object { $_.Name -eq 'AAPL' }) {
    $date = [datetime]::ParseExact($row.target_date, 'yyyy-MM-dd', [System.Globalization.CultureInfo]::InvariantCulture)
    $x = $date.ToOADate()
    $actualPoint = New-Object System.Windows.Forms.DataVisualization.Charting.DataPoint
    $actualPoint.XValue = $x
    $actualPoint.YValues = @([double]$row.actual_close_next)
    [void]$actualSeries.Points.Add($actualPoint)
    $predictedPoint = New-Object System.Windows.Forms.DataVisualization.Charting.DataPoint
    $predictedPoint.XValue = $x
    $predictedPoint.YValues = @([double]$row.$predictionColumn)
    [void]$predictedSeries.Points.Add($predictedPoint)
}
$chart.Series.Add($actualSeries)
$chart.Series.Add($predictedSeries)

$legend = New-Object System.Windows.Forms.DataVisualization.Charting.Legend
$legend.Docking = 'Bottom'
$legend.Font = New-Object System.Drawing.Font('Segoe UI', 10)
$chart.Legends.Add($legend)

$output = Join-Path $results 'actual_vs_predicted_best_model.png'
$chart.SaveImage($output, 'Png')
Write-Output ("AAPL points: {0}/{1}" -f $actualSeries.Points.Count, $predictedSeries.Points.Count)
$chart.Dispose()
Write-Output $output
