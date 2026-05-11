<?php

declare(strict_types=1);

error_reporting(E_ALL & ~E_DEPRECATED & ~E_WARNING & ~E_NOTICE);
ini_set('display_errors', '0');

require_once __DIR__ . '/../vendor/autoload.php';
require_once __DIR__ . '/../vendor/dapphp/securimage/securimage.php';

function render_png(string $code): string
{
    $img = new Securimage([
        'no_session' => true,
        'send_headers' => false,
        'no_exit' => true,
        'display_value' => $code,
    ]);

    $img->charset = '0123456789';
    $img->image_width = 150;
    $img->image_height = 80;
    $img->perturbation = 0.80;
    $img->use_transparent_text = false;
    $img->num_lines = 5;
    $img->code_length = 4;

    ob_start();
    $img->show();
    return (string)ob_get_clean();
}

while (($line = fgets(STDIN)) !== false) {
    $request = json_decode($line, true);
    if (!is_array($request)) {
        echo json_encode(['error' => 'Invalid JSON request.'], JSON_UNESCAPED_SLASHES), PHP_EOL;
        continue;
    }

    if (!empty($request['shutdown'])) {
        exit(0);
    }

    $code = trim((string)($request['code'] ?? ''));
    $count = max(1, (int)($request['count'] ?? 1));
    $outDir = trim((string)($request['out_dir'] ?? ''));
    $prefix = trim((string)($request['prefix'] ?? $code));

    if (!preg_match('/^\d{4}$/', $code)) {
        echo json_encode(['error' => 'Code must be exactly 4 digits.'], JSON_UNESCAPED_SLASHES), PHP_EOL;
        continue;
    }

    if ($outDir === '') {
        echo json_encode(['error' => 'Output directory is required.'], JSON_UNESCAPED_SLASHES), PHP_EOL;
        continue;
    }

    if (!is_dir($outDir) && !mkdir($outDir, 0777, true) && !is_dir($outDir)) {
        echo json_encode(['error' => "Unable to create output directory: {$outDir}"], JSON_UNESCAPED_SLASHES), PHP_EOL;
        continue;
    }

    $files = [];
    for ($index = 0; $index < $count; $index++) {
        $file = rtrim($outDir, DIRECTORY_SEPARATOR) . DIRECTORY_SEPARATOR . $prefix . '_' . $index . '.png';
        $png = render_png($code);
        file_put_contents($file, $png);
        $files[] = $file;
    }

    echo json_encode(['files' => $files], JSON_UNESCAPED_SLASHES), PHP_EOL;
}
