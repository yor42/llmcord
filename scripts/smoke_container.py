"""Run inside a built image with no network and only synthetic credentials."""
from __future__ import annotations

import http.client
from io import BytesIO
import os
from pathlib import Path
import shutil
import socket
import sqlite3
import subprocess
import sys
import tempfile
import time


def main() -> None:
    root = Path.cwd()
    sys.path.insert(0, str(root))
    for path in ('llmcord.py', 'web_main.py', 'migrate.py', 'config-gemini.yaml',
                 'llmcord_core/lore_drag.js', 'llmcord_core/web.py'):
        if not (root / path).is_file():
            raise RuntimeError(f'Application asset missing from image: {path}')
    for path in ('.env', 'config.yaml', '.git', 'data', 'secrets', '.secrets'):
        if (root / path).exists():
            raise RuntimeError(f'Private/local path unexpectedly included in image: {path}')

    # Exercise native libraries rather than only checking that packages installed.
    from PIL import Image
    from py_mini_racer import MiniRacer
    import llmcord
    import web_main
    from llmcord_core.config import load_settings

    assert callable(llmcord.main) and callable(web_main.main)
    with MiniRacer() as js:
        assert js.eval('new RegExp("courier", "i").test("Courier")') is True
    image = BytesIO()
    Image.new('RGB', (2, 2)).save(image, format='PNG')
    Image.open(BytesIO(image.getvalue())).verify()

    with tempfile.TemporaryDirectory() as directory:
        work = Path(directory)
        shutil.copyfile(root / 'config-gemini.yaml', work / 'config.yaml')
        with socket.socket() as sock:
            sock.bind(('127.0.0.1', 0))
            port = sock.getsockname()[1]
        database = work / 'smoke.sqlite3'
        env = {**os.environ, 'DISCORD_BOT_TOKEN': 'smoke-placeholder',
               'DISCORD_CLIENT_ID': '1', 'DISCORD_CLIENT_SECRET': 'smoke-placeholder',
               'GEMINI_API_KEY': 'smoke-placeholder', 'DISCORD_GUILD_ID': '1',
               'WEB_BASE_URL': 'https://smoke.example.invalid', 'WEB_HOST': '127.0.0.1',
               'WEB_PORT': str(port), 'LLMCORD_DATABASE_PATH': str(database)}
        os.environ.update(env)
        settings = load_settings(work / 'config.yaml')
        assert settings.profile('dialogue').provider == 'compatible'
        subprocess.run([sys.executable, str(root / 'migrate.py')], cwd=work,
                       env=env, check=True, timeout=30)
        with sqlite3.connect(database) as connection:
            assert connection.execute('PRAGMA integrity_check').fetchone()[0] == 'ok'
            assert connection.execute("SELECT name FROM sqlite_master WHERE name='characters'").fetchone()

        with (work / 'web.log').open('w+') as log:
            process = subprocess.Popen([sys.executable, str(root / 'web_main.py')],
                                       cwd=work, env=env, stdout=log, stderr=log)
            try:
                for _ in range(150):
                    if process.poll() is not None:
                        raise RuntimeError('Dashboard exited during startup')
                    try:
                        connection = http.client.HTTPConnection('127.0.0.1', port, timeout=1)
                        connection.request('GET', '/admin/')
                        response = connection.getresponse()
                        body = response.read().decode()
                        connection.close()
                        if response.status == 200 and 'Sign in with Discord' in body:
                            break
                    except OSError:
                        pass
                    time.sleep(0.2)
                else:
                    raise RuntimeError('Dashboard login page did not become ready')
            except Exception:
                log.seek(0)
                print(log.read())
                raise
            finally:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=10)
    print('PASS: image assets, native libraries, configuration, migrations, and dashboard startup')


if __name__ == '__main__':
    main()
