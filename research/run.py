"""Config-driven torchrun launcher. Dry run needs only Python's standard library."""
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import platform
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def load_config(path):
    c = json.loads(Path(path).read_text(encoding='utf-8'))
    args = c['train']
    # Validate flags against the actual upstream CLI, without importing CUDA packages.
    tree = ast.parse((ROOT/'train.py').read_text(encoding='utf-8'))
    flags = {arg.value[2:] for n in ast.walk(tree) if isinstance(n, ast.Call)
             and isinstance(n.func, ast.Attribute) and n.func.attr == 'add_argument'
             for arg in n.args if isinstance(arg, ast.Constant) and isinstance(arg.value, str)
             and arg.value.startswith('--')}
    unknown = set(args) - flags
    if unknown:
        raise ValueError(f'Unknown train.py flags: {unknown}')
    if args.get('research_mask', 'vanilla') not in ('vanilla', 'fork', 'random'):
        raise ValueError('Invalid masking method')
    if args['enable_thinking'] != 'true' or args['sync_ref_model'] is not False:
        raise ValueError('Team baseline requires ON/ON and frozen teacher')
    if c['gpus'] * args['per_device_train_batch_size'] * args['gradient_accumulation_steps'] != 64:
        raise ValueError('Effective batch must be 64')
    if not (ROOT/args['fsdp_config']).is_file():
        raise ValueError('FSDP config missing')
    return c


def freeze():
    """pip-freeze-style package list; works in uv venvs that ship without pip."""
    from importlib.metadata import distributions
    return ''.join(sorted(f"{d.metadata['Name']}=={d.version}\n" for d in distributions()))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True)
    p.add_argument('--run-id', required=True)
    p.add_argument('--seed', type=int)
    p.add_argument('--dry-run', action='store_true')
    a = p.parse_args()
    os.chdir(ROOT)
    c = load_config(a.config)
    if not a.run_id or any(x not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_' for x in a.run_id):
        p.error('run-id must contain only letters, numbers, hyphen or underscore')
    if a.seed is not None:
        c['train']['seed'] = a.seed
    run = ROOT/'runs'/a.run_id
    if run.exists():
        raise FileExistsError(f'Never overwrite an existing run: {run}')
    c['train']['output_dir'] = str(run/'checkpoint')
    cmd = [sys.executable, '-m', 'torch.distributed.run', '--standalone',
           '--nproc_per_node', str(c['gpus']), 'train.py']
    for k, v in c['train'].items():
        cmd += ['--'+k, str(v).lower() if isinstance(v, bool) else str(v)]
    print(shlex.join(cmd))
    if a.dry_run:
        return
    if not Path(c['train']['train_file_path']).is_file():
        raise FileNotFoundError('Build training parquet first')
    run.mkdir(parents=True)
    manifest = {'config': c, 'command': cmd, 'python': platform.python_version(),
                'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                'git_dirty': bool(subprocess.check_output(['git', 'status', '--porcelain'], text=True)),
                'dataset_sha256': sha256(c['train']['train_file_path']),
                'environment': freeze()}
    # Only allowlisted environment is recorded; credentials are never captured.
    manifest['cuda_visible_devices'] = os.environ.get('CUDA_VISIBLE_DEVICES')
    (run/'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    with (run/'hardware.txt').open('w', encoding='utf-8') as f:
        subprocess.run(['nvidia-smi'], stdout=f, stderr=subprocess.STDOUT, check=True)
    status = subprocess.run(cmd).returncode
    (run/'status.json').write_text(json.dumps({'exit_code': status}), encoding='utf-8')
    raise SystemExit(status)


if __name__ == '__main__':
    main()
