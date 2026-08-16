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

$modelLabel = switch ($best.model) {
    'Naive_Close_t' { 'Naive Persistence (Naive_Close_t)' }
    'LinearRegression' { 'Linear Regression' }
    'RandomForestRegressor' { 'Random Forest Regressor' }
}

function Add-CommonTitles {
    param(
        [System.Windows.Forms.DataVisualization.Charting.Chart]$Chart,
        [string]$MainTitle,
        [string]$SubTitle
    )

    $mainTitleObject = New-Object System.Windows.Forms.DataVisualization.Charting.Title
    $mainTitleObject.Text = $MainTitle
    $mainTitleObject.Font = New-Object System.Drawing.Font('Segoe UI', 25, [System.Drawing.FontStyle]::Bold)
    $mainTitleObject.ForeColor = [System.Drawing.Color]::FromArgb(15, 23, 42)
    [void]$Chart.Titles.Add($mainTitleObject)

    $subTitleObject = New-Object System.Windows.Forms.DataVisualization.Charting.Title
    $subTitleObject.Text = $SubTitle
    $subTitleObject.Font = New-Object System.Drawing.Font('Segoe UI', 15)
    $subTitleObject.ForeColor = [System.Drawing.Color]::FromArgb(71, 85, 105)
    [void]$Chart.Titles.Add($subTitleObject)
}

function New-BaseChart {
    $chart = New-Object System.Windows.Forms.DataVisualization.Charting.Chart
    $chart.Width = 2000
    $chart.Height = 1200
    $chart.BackColor = [System.Drawing.Color]::White
    $chart.AntiAliasing = 'All'
    $chart.TextAntiAliasingQuality = 'High'
    return $chart
}

# Chart 1: actual close versus predicted close for the complete final test set.
$scatterChart = New-BaseChart
Add-CommonTitles `
    -Chart $scatterChart `
    -MainTitle "Actual Close vs Predicted Close - Best Model: $modelLabel" `
    -SubTitle ('S&P 500 Stock final test set | MAE = {0:N4} | RMSE = {1:N4} | R2 = {2:N6} | MAPE = {3:N4}%' -f [double]$best.mae, [double]$best.rmse, [double]$best.r2, [double]$best.mape_percent)

$scatterArea = New-Object System.Windows.Forms.DataVisualization.Charting.ChartArea('ScatterArea')
$scatterArea.Position = New-Object System.Windows.Forms.DataVisualization.Charting.ElementPosition(7, 14, 86, 72)
$scatterArea.AxisX.Title = 'Actual Close Price (USD, log scale)'
$scatterArea.AxisY.Title = 'Predicted Close Price (USD, log scale)'
$scatterArea.AxisX.TitleFont = New-Object System.Drawing.Font('Segoe UI', 17, [System.Drawing.FontStyle]::Bold)
$scatterArea.AxisY.TitleFont = New-Object System.Drawing.Font('Segoe UI', 17, [System.Drawing.FontStyle]::Bold)
$scatterArea.AxisX.IsLogarithmic = $true
$scatterArea.AxisY.IsLogarithmic = $true
$scatterArea.AxisX.Minimum = 1
$scatterArea.AxisY.Minimum = 1
$scatterArea.AxisX.Maximum = 2500
$scatterArea.AxisY.Maximum = 2500
$scatterArea.AxisX.MajorGrid.LineColor = [System.Drawing.Color]::FromArgb(226, 232, 240)
$scatterArea.AxisY.MajorGrid.LineColor = [System.Drawing.Color]::FromArgb(226, 232, 240)
$scatterArea.AxisX.LabelStyle.Font = New-Object System.Drawing.Font('Segoe UI', 14, [System.Drawing.FontStyle]::Bold)
$scatterArea.AxisY.LabelStyle.Font = New-Object System.Drawing.Font('Segoe UI', 14, [System.Drawing.FontStyle]::Bold)
$scatterChart.ChartAreas.Add($scatterArea)

$scatterSeries = New-Object System.Windows.Forms.DataVisualization.Charting.Series('Test observations: actual vs predicted')
$scatterSeries.ChartArea = 'ScatterArea'
$scatterSeries.ChartType = 'Point'
$scatterSeries.MarkerStyle = 'Circle'
$scatterSeries.MarkerSize = 6
$scatterSeries.Color = [System.Drawing.Color]::FromArgb(95, 37, 99, 235)

# Plot a regular sample for readability while keeping the whole test price range.
for ($i = 0; $i -lt $predictions.Count; $i += 12) {
    $actual = [double]$predictions[$i].actual_close_next
    $predicted = [double]$predictions[$i].$predictionColumn
    if ($actual -gt 0 -and $predicted -gt 0) {
        [void]$scatterSeries.Points.AddXY($actual, $predicted)
    }
}
$scatterChart.Series.Add($scatterSeries)

$identity = New-Object System.Windows.Forms.DataVisualization.Charting.Series('Perfect prediction: Predicted = Actual')
$identity.ChartArea = 'ScatterArea'
$identity.ChartType = 'Line'
$identity.BorderWidth = 4
$identity.BorderDashStyle = 'Dash'
$identity.Color = [System.Drawing.Color]::FromArgb(220, 38, 38)
$identity.Points.AddXY(1, 1) | Out-Null
$identity.Points.AddXY(2500, 2500) | Out-Null
$scatterChart.Series.Add($identity)

$scatterLegend = New-Object System.Windows.Forms.DataVisualization.Charting.Legend
$scatterLegend.Docking = 'Bottom'
$scatterLegend.Font = New-Object System.Drawing.Font('Segoe UI', 15)
$scatterChart.Legends.Add($scatterLegend)

$scatterOutput = Join-Path $results 'chart_1_actual_close_vs_predicted_close.png'
$scatterPointCount = $scatterSeries.Points.Count
$scatterChart.SaveImage($scatterOutput, 'Png')
$scatterChart.Dispose()

# Chart 2: target date versus close price for one representative stock.
$symbol = 'AAPL'
$symbolRows = $predictions |
    Where-Object { $_.Name -eq $symbol } |
    Sort-Object { [datetime]$_.target_date }

$timeChart = New-BaseChart
Add-CommonTitles `
    -Chart $timeChart `
    -MainTitle "Actual and Predicted Close Price over Target Date - $symbol" `
    -SubTitle "Best model: $modelLabel | S&P 500 Stock final test period"

$timeArea = New-Object System.Windows.Forms.DataVisualization.Charting.ChartArea('TimeArea')
$timeArea.Position = New-Object System.Windows.Forms.DataVisualization.Charting.ElementPosition(7, 14, 87, 72)
$timeArea.AxisX.Title = 'Target Date'
$timeArea.AxisY.Title = 'Close Price (USD)'
$timeArea.AxisX.TitleFont = New-Object System.Drawing.Font('Segoe UI', 17, [System.Drawing.FontStyle]::Bold)
$timeArea.AxisY.TitleFont = New-Object System.Drawing.Font('Segoe UI', 17, [System.Drawing.FontStyle]::Bold)
$timeArea.AxisX.LabelStyle.Format = 'MMM yyyy'
$timeArea.AxisX.IntervalType = 'Months'
$timeArea.AxisX.Interval = 1
$timeArea.AxisX.LabelStyle.Angle = -35
$timeArea.AxisX.MajorGrid.LineColor = [System.Drawing.Color]::FromArgb(226, 232, 240)
$timeArea.AxisY.MajorGrid.LineColor = [System.Drawing.Color]::FromArgb(226, 232, 240)
$timeArea.AxisX.LabelStyle.Font = New-Object System.Drawing.Font('Segoe UI', 13, [System.Drawing.FontStyle]::Bold)
$timeArea.AxisY.LabelStyle.Font = New-Object System.Drawing.Font('Segoe UI', 14, [System.Drawing.FontStyle]::Bold)
$timeChart.ChartAreas.Add($timeArea)

$actualSeries = New-Object System.Windows.Forms.DataVisualization.Charting.Series('Actual Close')
$actualSeries.ChartArea = 'TimeArea'
$actualSeries.ChartType = 'Line'
$actualSeries.BorderWidth = 6
$actualSeries.Color = [System.Drawing.Color]::FromArgb(37, 99, 235)
$actualSeries.XValueType = 'DateTime'

$predictedSeries = New-Object System.Windows.Forms.DataVisualization.Charting.Series('Predicted Close')
$predictedSeries.ChartArea = 'TimeArea'
$predictedSeries.ChartType = 'Line'
$predictedSeries.BorderWidth = 5
$predictedSeries.BorderDashStyle = 'Dash'
$predictedSeries.Color = [System.Drawing.Color]::FromArgb(220, 38, 38)
$predictedSeries.XValueType = 'DateTime'

foreach ($row in $symbolRows) {
    $date = [datetime]::ParseExact($row.target_date, 'yyyy-MM-dd', [System.Globalization.CultureInfo]::InvariantCulture)
    [void]$actualSeries.Points.AddXY($date.ToOADate(), [double]$row.actual_close_next)
    [void]$predictedSeries.Points.AddXY($date.ToOADate(), [double]$row.$predictionColumn)
}
$timeChart.Series.Add($actualSeries)
$timeChart.Series.Add($predictedSeries)

$timeLegend = New-Object System.Windows.Forms.DataVisualization.Charting.Legend
$timeLegend.Docking = 'Bottom'
$timeLegend.Font = New-Object System.Drawing.Font('Segoe UI', 15)
$timeChart.Legends.Add($timeLegend)

$timeOutput = Join-Path $results 'chart_2_target_date_close_price_aapl.png'
$timeChart.SaveImage($timeOutput, 'Png')
$timeChart.Dispose()

Write-Output ("Best model: {0}" -f $best.model)
Write-Output ("Scatter points: {0}" -f $scatterPointCount)
Write-Output ("{0} time-series points: {1}" -f $symbol, $symbolRows.Count)
Write-Output $scatterOutput
Write-Output $timeOutput
