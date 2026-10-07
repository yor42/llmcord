"""Reject tracked runtime files without displaying their contents."""
from __future__ import annotations

import fnmatch
from pathlib import Path, PurePosixPath
import subprocess


PRIVATE_DIRECTORIES = {'data', 'backups', 'secrets', '.secrets', '.ssh', '.aws',
                       '.venv', '.venv314', '.test-artifacts', '.nicegui', 'logs'}
PRIVATE_NAMES = (
    'config.yaml', 'config.local.yaml', 'config.*.local.yaml',
    'credentials.json', 'credentials.*.json', 'service-account*.json',
    '*.db', '*.db-*', '*.db.*', '*.sqlite', '*.sqlite-*', '*.sqlite.*',
    '*.sqlite3', '*.sqlite3-*', '*.sqlite3.*', '*.key', '*.pem', '*.p12',
    '*.pfx', '*.crt', '*.cer', 'id_rsa*', 'id_ecdsa*', 'id_dsa*',
    'id_ed25519*', 'id_xmss*', '*.log',
)


def private_path(path: str) -> bool:
    parts = PurePosixPath(path).parts
    name = parts[-1].lower()
    # Existing directory placeholders contain only ignore rules.
    if name == '.gitignore':
        return False
    if any(part.lower() in PRIVATE_DIRECTORIES for part in parts[:-1]):
        return True
    if name == '.env' or name.startswith('.env.'):
        return not (name == '.env.example' or name.endswith('.example'))
    return any(fnmatch.fnmatchcase(name, pattern) for pattern in PRIVATE_NAMES)


def is_sqlite(path: Path) -> bool:
    with path.open('rb') as file:
        return file.read(16) == b'SQLite format 3\x00'


def main() -> None:
    paths = subprocess.check_output(['git', 'ls-files', '-z']).decode().split('\x00')
    ignored = set(subprocess.check_output([
        'git', 'ls-files', '--cached', '--ignored', '--exclude-standard', '-z',
    ]).decode().split('\x00'))
    rejected = []
    for path in filter(None, paths):
        file = Path(path)
        if (private_path(path) or
                (path in ignored and file.name != '.gitignore') or
                (file.is_file() and is_sqlite(file))):
            rejected.append(path)
    if rejected:
        print('Remove these private/runtime files from version control:')
        for path in rejected:
            print(f'  {path}')
        raise SystemExit(1)
    print('PASS: tracked files contain no private runtime paths or SQLite databases')


if __name__ == '__main__':
    main()
