from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent


def resolve_repo_path(path_str: str) -> Path:
    path = Path(path_str).expanduser()
    if path.is_absolute():
        return path
    return REPO_ROOT / path
