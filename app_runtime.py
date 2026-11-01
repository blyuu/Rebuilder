"""Installation/runtime configuration, independent of cwd and GUI imports."""
from __future__ import annotations
import json
import os
import subprocess
from pathlib import Path

VERSION = '0.14.0'
ENVIRONMENTS = {'gui': '.venv-gui', 'recon': '.venv', 'video': '.venv-video',
                'depth': '.venv-depth', 'gsplat': '.venv-gsplat'}


def install_root() -> Path:
    return Path(os.environ.get('RECONX_HOME', Path(__file__).resolve().parents[2])).resolve()


def read_config(root: Path) -> dict:
    path = Path(root) / 'config/application.json'
    config = {'schema_version': 1, 'runtimes': {}, 'models': {}, 'external': {}}
    if path.is_file():
        config.update(json.loads(path.read_text(encoding='utf-8')))
    override = Path(root) / 'config/local.json'
    if override.is_file():
        local = json.loads(override.read_text(encoding='utf-8'))
        for key in ('runtimes', 'models', 'external'):
            config[key].update(local.get(key, {}))
    if config['schema_version'] != 1:
        raise ValueError('Unsupported application configuration schema')
    for key in ('runtimes', 'models', 'external'):
        if not isinstance(config[key], dict) or any(not isinstance(v, str) or not v.strip() for v in config[key].values()):
            raise ValueError(f'Invalid {key} configuration; expected path strings')
    return config


def resolve(root: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else Path(root) / path).resolve()


def runtime(root: Path, kind: str) -> Path:
    config = read_config(root)
    if kind not in ENVIRONMENTS: raise ValueError(f'Unknown runtime: {kind}')
    configured = config['runtimes'].get(kind)
    path = resolve(root, configured) if configured else Path(root)/ENVIRONMENTS[kind]/'Scripts/python.exe'
    if not path.is_file():
        raise FileNotFoundError(f'{kind} runtime missing: {path}. Install its runtime pack or configure config/local.json.')
    return path


def model_path(name: str, fallback: str) -> str:
    value = read_config(install_root())['models'].get(name)
    if value:
        path = resolve(install_root(), value)
        if not path.is_dir(): raise FileNotFoundError(f'Missing {name} model: {path}. Set its model directory in Home / Runtime assets.')
        return str(path)
    if os.environ.get('RECONX_PACKAGED') == '1':
        raise FileNotFoundError(f'Model {name} is not configured. Select a downloaded model directory in Home / Runtime assets; no automatic download is performed.')
    return fallback


def inventory(root: Path) -> dict:
    config = read_config(root)
    runtimes = {}
    for kind in ENVIRONMENTS:
        try: runtimes[kind] = {'available': True, 'path': str(runtime(root, kind))}
        except FileNotFoundError as exc: runtimes[kind] = {'available': False, 'reason': str(exc)}
    models = {}
    for name, version, estimate in [('depth_anything_v2_small', 'Small-hf', 100_000_000),
                                     ('marigold_v1_1', '1.1 fp16', 3_500_000_000),
                                     ('segformer', '489d5cd81a0b59fab9b7ea758d3548ebe99677da', 15_000_000)]:
        path = resolve(root, config['models'][name]) if name in config['models'] else None
        weights = list(path.rglob('*.safetensors')) + list(path.rglob('*.bin')) if path and path.is_dir() else []
        models[name] = {'version': version, 'runtime': 'depth', 'path': str(path) if path else None,
                        'available': bool(weights), 'estimated_storage_bytes': estimate,
                        'action': 'Select an existing model directory; review upstream license before downloading.'}
    external = {k: {'path': str(resolve(root,v)), 'available': resolve(root,v).is_file()} for k,v in config['external'].items()}
    return {'version': VERSION, 'root': str(root), 'runtimes': runtimes, 'models': models, 'external': external}


def child_environment(root: Path) -> dict[str,str]:
    env = dict(os.environ)
    env.update(RECONX_HOME=str(root), PYTHONPATH=str(root/'src'), PYTHONNOUSERSITE='1',
               PYTHONUTF8='1', PYTHONUNBUFFERED='1')
    env.pop('PYTHONHOME', None)
    if (root/'config/application.json').is_file():
        env.update(RECONX_PACKAGED='1', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    return env


def gpu_status() -> dict:
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=name,driver_version,memory.total,memory.used,utilization.gpu',
                                 '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=5,
                                creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        return {'available': result.returncode == 0, 'devices_csv': result.stdout.strip(), 'error': result.stderr.strip()}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {'available': False, 'error': str(exc)}
