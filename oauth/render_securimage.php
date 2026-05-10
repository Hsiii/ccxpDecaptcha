<?php

declare(strict_types=1);

error_reporting(E_ALL & ~E_DEPRECATED & ~E_WARNING & ~E_NOTICE);
ini_set('display_errors', '0');

require_once __DIR__ . '/../vendor/autoload.php';
require_once __DIR__ . '/../vendor/dapphp/securimage/securimage.php';

if ($argc < 2) {
    fwrite(STDERR, "Usage: php oauth/render_securimage.php <code>\n");
    exit(1);
}

$code = trim((string)$argv[1]);
if (!preg_match('/^\d{4}$/', $code)) {
    fwrite(STDERR, "Code must be exactly 4 digits.\n");
    exit(1);
}

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

$img->show();
