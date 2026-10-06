"""Install (but do not start) native user services for this checkout."""
from pathlib import Path
import subprocess


def main():
    project = Path(__file__).resolve().parents[1]
    if any(char in str(project) for char in (' ', '\n', '%', '"')):
        raise SystemExit('Service installation requires a checkout path without spaces or systemd metacharacters')
    target = Path.home() / '.config/systemd/user'
    target.mkdir(parents=True, exist_ok=True)
    for name in ('llmcord-bot.service', 'llmcord-web.service'):
        content = (project / 'deploy' / name).read_text().replace('@PROJECT@', str(project))
        (target / name).write_text(content)
    subprocess.run(['systemctl', '--user', 'daemon-reload'], check=True)
    print('Installed bot and web services. Add credentials to .env before enabling them.')


if __name__ == '__main__':
    main()
