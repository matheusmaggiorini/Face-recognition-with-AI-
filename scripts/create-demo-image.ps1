# Creates assets/demo-preview.png for the README.
Add-Type -AssemblyName System.Drawing

$width = 960
$height = 540
$bitmap = New-Object System.Drawing.Bitmap $width, $height
$graphics = [System.Drawing.Graphics]::FromImage($bitmap)
$graphics.SmoothingMode = [System.Drawing.Drawing2D.SmoothingMode]::AntiAlias
$graphics.Clear([System.Drawing.Color]::FromArgb(28, 32, 38))

$room = New-Object System.Drawing.Drawing2D.LinearGradientBrush(
    [System.Drawing.Point]::new(0, 0),
    [System.Drawing.Point]::new(0, $height),
    [System.Drawing.Color]::FromArgb(55, 65, 78),
    [System.Drawing.Color]::FromArgb(22, 26, 32)
)
$graphics.FillRectangle($room, 0, 0, $width, $height)

$titleBrush = New-Object System.Drawing.SolidBrush ([System.Drawing.Color]::FromArgb(45, 49, 57))
$graphics.FillRectangle($titleBrush, 40, 40, 880, 460)
$graphics.DrawString(
    "Face Detection with Loading Bar",
    (New-Object System.Drawing.Font("Segoe UI", 11)),
    [System.Drawing.Brushes]::White,
    52, 48
)

$faceX = 355
$faceY = 130
$faceW = 250
$faceH = 300
$skinBrush = New-Object System.Drawing.SolidBrush ([System.Drawing.Color]::FromArgb(210, 175, 145))
$graphics.FillEllipse($skinBrush, $faceX + 40, $faceY + 40, 170, 200)

$eyeBrush = New-Object System.Drawing.SolidBrush ([System.Drawing.Color]::FromArgb(35, 40, 48))
$graphics.FillEllipse($eyeBrush, $faceX + 70, $faceY + 95, 35, 20)
$graphics.FillEllipse($eyeBrush, $faceX + 145, $faceY + 95, 35, 20)

$penBlue = New-Object System.Drawing.Pen ([System.Drawing.Color]::FromArgb(255, 0, 0)), 3
$graphics.DrawRectangle($penBlue, $faceX, $faceY, $faceW, $faceH)

$barY = $faceY + $faceH + 8
$graphics.FillRectangle([System.Drawing.Brushes]::Black, $faceX, $barY, $faceW, 20)
$graphics.FillRectangle([System.Drawing.Brushes]::Lime, $faceX, $barY, 180, 20)

$outputPath = Join-Path $PSScriptRoot "..\assets\demo-preview.png"
$bitmap.Save($outputPath, [System.Drawing.Imaging.ImageFormat]::Png)

$graphics.Dispose()
$bitmap.Dispose()
$room.Dispose()
$titleBrush.Dispose()
$skinBrush.Dispose()
$eyeBrush.Dispose()
$penBlue.Dispose()

Write-Host "Saved $outputPath"
