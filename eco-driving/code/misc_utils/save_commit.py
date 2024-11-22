import os
import subprocess
from pathlib import Path


def _save_text(path, string):
    with open(path, 'w') as f:
        f.write(string)


def _shell(cmd, wait=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE):
    stdout = stdout or subprocess.DEVNULL
    stderr = stderr or subprocess.DEVNULL
    if not isinstance(cmd, str):
        cmd = ' '.join(cmd)
    process = subprocess.Popen(cmd, shell=True, stdout=stdout, stderr=stderr)
    if not wait:
        return process
    out, err = process.communicate()
    return out.decode().rstrip('\n') if out else '', err.decode().rstrip('\n') if err else ''


def save_commit(repo_root: Path, output_wd: Path):
    cwd = os.getcwd()
    os.chdir(str(repo_root.absolute()))
    status = _shell('git status')[0]
    base_commit = _shell('git rev-parse HEAD')[0]
    diff = _shell('git diff %s' % base_commit)[0]
    os.chdir(cwd)

    save_dir = output_wd / 'commit'
    save_dir.mkdir(exist_ok=True, parents=True)

    _save_text(save_dir / 'hash.txt', base_commit)
    _save_text(save_dir / 'diff.txt', diff)
    _save_text(save_dir / 'status.txt', status)