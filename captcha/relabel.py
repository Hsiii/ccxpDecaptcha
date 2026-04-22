import argparse
import io
from pathlib import Path
from typing import Iterable, List, Tuple

from PIL import Image


def render_image_in_terminal(content: bytes):
    with Image.open(io.BytesIO(content)) as image:
        rgb_image = image.convert('RGB')
        width, height = rgb_image.size
        scale = max(1, 180 // width)
        resized = rgb_image.resize((width * scale, height * scale), Image.Resampling.NEAREST)
        pixels = resized.load()

        print('\x1b[2J\x1b[H', end='')
        print()
        for y in range(0, resized.height - 1, 2):
            row = []
            for x in range(resized.width):
                top = pixels[x, y]
                bottom = pixels[x, y + 1]
                row.append(
                    f'\x1b[38;2;{top[0]};{top[1]};{top[2]}m'
                    f'\x1b[48;2;{bottom[0]};{bottom[1]};{bottom[2]}m▀'
                )
            row.append('\x1b[0m')
            print(''.join(row))
        if resized.height % 2 == 1:
            row = []
            y = resized.height - 1
            for x in range(resized.width):
                top = pixels[x, y]
                row.append(f'\x1b[38;2;{top[0]};{top[1]};{top[2]}m▀')
            row.append('\x1b[0m')
            print(''.join(row))
        print()


def parse_group_file(path: Path) -> Tuple[str, str, str]:
    stem = path.stem
    if '__' not in stem or '_' not in stem:
        raise ValueError(f'Unsupported raw data filename: {path.name}')

    group, remainder = stem.split('__', maxsplit=1)
    label, index = remainder.rsplit('_', maxsplit=1)
    if len(label) != 6 or not label.isdigit():
        raise ValueError(f'Unsupported label in raw data filename: {path.name}')
    if not index.isdigit():
        raise ValueError(f'Unsupported index in raw data filename: {path.name}')

    return group, label, index


def collect_group_files(raw_dir: Path, group: str) -> List[Path]:
    paths = sorted(raw_dir.glob(f'{group}__*.png'))
    if not paths:
        raise FileNotFoundError(f'No files found for group {group} in {raw_dir}')
    return paths


def collect_latest_group(raw_dir: Path) -> str:
    paths = sorted(raw_dir.glob('*__*.png'))
    if not paths:
        raise FileNotFoundError(f'No labeled files found in {raw_dir}')
    latest_path = max(paths, key=lambda path: (path.stat().st_mtime_ns, path.name))
    latest_group, _, _ = parse_group_file(latest_path)
    return latest_group


def collect_group_labels(paths: Iterable[Path]) -> List[str]:
    return sorted({parse_group_file(path)[1] for path in paths})


def preview_group(paths: List[Path], preview_index: int):
    if preview_index < 0 or preview_index >= len(paths):
        raise IndexError(f'Preview index {preview_index} out of range for {len(paths)} files')
    render_image_in_terminal(paths[preview_index].read_bytes())
    print(f'Previewing {paths[preview_index].name}')


def rename_group_files(raw_dir: Path, group: str, old_label: str, new_label: str, dry_run: bool):
    if len(new_label) != 6 or not new_label.isdigit():
        raise ValueError(f'New label must be a 6-digit string, got {new_label!r}')

    group_paths = collect_group_files(raw_dir, group)
    rename_pairs = []

    for path in group_paths:
        parsed_group, parsed_label, index = parse_group_file(path)
        if parsed_group != group or parsed_label != old_label:
            continue
        target = raw_dir / f'{group}__{new_label}_{index}.png'
        rename_pairs.append((path, target))

    if not rename_pairs:
        raise FileNotFoundError(f'No files found for group {group} with label {old_label}')

    conflicts = [target.name for _, target in rename_pairs if target.exists()]
    if conflicts:
        raise FileExistsError(
            'Refusing to overwrite existing files for the target label: '
            + ', '.join(conflicts[:5])
            + ('...' if len(conflicts) > 5 else '')
        )

    for source, target in rename_pairs:
        print(f'{source.name} -> {target.name}')
        if not dry_run:
            source.rename(target)


def main():
    parser = argparse.ArgumentParser(description='Relabel one grouped captcha batch in /data.')
    parser.add_argument(
        'group_or_label',
        nargs='?',
        help='Group id, or the new label when targeting the latest group.',
    )
    parser.add_argument('new_label', nargs='?', help='Correct 6-digit label to write into filenames.')
    parser.add_argument('--raw-dir', default='/data', help='Directory containing the collected PNG files.')
    parser.add_argument(
        '--latest',
        '--edit-latest',
        dest='latest',
        action='store_true',
        help='Relabel the most recently submitted captcha group. This is already the default when no group id is given.',
    )
    parser.add_argument(
        '--old-label',
        help='Current 6-digit label to replace. Required only when the group has mixed labels.',
    )
    parser.add_argument(
        '--preview-index',
        type=int,
        default=0,
        help='Which render in the group to preview before renaming.',
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Print the planned renames without changing any files.',
    )
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    if args.latest:
        if args.new_label is not None:
            raise ValueError('When using --latest, pass at most one positional argument for the new label.')
        group = collect_latest_group(raw_dir)
        new_label = args.group_or_label
    elif args.new_label is not None:
        group = args.group_or_label
        new_label = args.new_label
    else:
        group = collect_latest_group(raw_dir)
        new_label = args.group_or_label

    if group is None:
        raise ValueError('A group id could not be resolved.')

    group_paths = collect_group_files(raw_dir, group)
    group_labels = collect_group_labels(group_paths)

    preview_group(group_paths, args.preview_index)
    print(f'Found {len(group_paths)} files for group {group}')
    print(f'Labels present: {", ".join(group_labels)}')

    if args.old_label is not None:
        old_label = args.old_label
    elif len(group_labels) == 1:
        old_label = group_labels[0]
    else:
        raise ValueError('Group has mixed labels; pass --old-label to select which label to replace.')

    if new_label is None:
        new_label = input('Enter the correct 6-digit label: ').strip()

    rename_group_files(raw_dir, group, old_label, new_label, args.dry_run)


if __name__ == '__main__':
    main()
