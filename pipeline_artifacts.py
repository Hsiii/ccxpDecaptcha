import shutil
from pathlib import Path
from typing import Dict, List, Optional


def build_training_output_paths(output_dir: Path) -> Dict[str, Path]:
    checkpoints_dir = output_dir / 'checkpoints'
    eval_dir = output_dir / 'eval'
    return {
        'checkpoints_dir': checkpoints_dir,
        'eval_dir': eval_dir,
        'last_checkpoint': checkpoints_dir / 'last.pt',
        'best_checkpoint': checkpoints_dir / 'best.pt',
        'quantized_checkpoint': checkpoints_dir / 'int8.pt',
        'val_failures': eval_dir / 'val.csv',
        'test_failures': eval_dir / 'test.csv',
        'val_confusion': eval_dir / 'val_cm.npy',
        'test_confusion': eval_dir / 'test_cm.npy',
        'metrics_summary': eval_dir / 'metrics.json',
    }


def prepare_training_output_dir(
    output_dir: Path,
    overwrite_output: bool,
    preserve_paths: Optional[List[Path]] = None,
) -> Dict[str, Path]:
    paths = build_training_output_paths(output_dir)
    preserved = {path.resolve() for path in (preserve_paths or []) if path.exists()}
    existing = [path for key, path in paths.items() if key not in {'checkpoints_dir', 'eval_dir'} and path.exists()]
    if existing and not overwrite_output:
        joined = ', '.join(str(path) for path in existing)
        raise FileExistsError(
            f'Refusing to overwrite existing training artifacts in {output_dir}: {joined}. '
            'Pass --overwrite to replace them or choose a new --out.'
        )

    paths['checkpoints_dir'].mkdir(parents=True, exist_ok=True)
    paths['eval_dir'].mkdir(parents=True, exist_ok=True)
    if overwrite_output:
        for path in existing:
            if path.resolve() in preserved:
                continue
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
    return paths
