<?php

declare(strict_types=1);

error_reporting(E_ALL & ~E_DEPRECATED & ~E_WARNING & ~E_NOTICE);
ini_set('display_errors', '0');

require_once __DIR__ . '/../vendor/autoload.php';
require_once __DIR__ . '/../vendor/dapphp/securimage/securimage.php';

if ($argc < 2) {
    fwrite(STDERR, "Usage: php oauth/render_securimage.php <code> [count [outdir [prefix]]]\n");
    exit(1);
}

$code = trim((string)$argv[1]);
if (!preg_match('/^\d{4}$/', $code)) {
    fwrite(STDERR, "Code must be exactly 4 digits.\n");
    exit(1);
}

$count = max(1, (int)($argv[2] ?? 1));
$outDir = isset($argv[3]) ? trim((string)$argv[3]) : null;
$prefix = isset($argv[4]) ? trim((string)$argv[4]) : $code;

if ($outDir !== null && $outDir !== '') {
    if (!is_dir($outDir) && !mkdir($outDir, 0777, true) && !is_dir($outDir)) {
        fwrite(STDERR, "Unable to create output directory: {$outDir}\n");
        exit(1);
    }
}

for ($index = 0; $index < $count; $index++) {
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

    if ($outDir !== null && $outDir !== '') {
        $file = rtrim($outDir, DIRECTORY_SEPARATOR) . DIRECTORY_SEPARATOR . $prefix . '_' . $index . '.png';
        ob_start();
        $img->show();
        $png = (string)ob_get_clean();
        file_put_contents($file, $png);
        echo $file, PHP_EOL;
        continue;
    }

    $img->show();
}
