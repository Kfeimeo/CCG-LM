import ast
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

import pytest

from build_train_notebook import CELLS, write_notebook


def source(marker):
    return next(s for kind, s in CELLS if kind == "code" and marker in s)


def namespace(tmp_path):
    ns = {}
    exec(source('REPO_URL ='), ns)
    ns.update(WORKSPACE_DIR=str(tmp_path / 'workspace with spaces'),
              PROJECT_DIR=str(tmp_path / 'cloned source'), STORAGE_DIR=str(tmp_path / 'persistent data'))
    return ns


def project(folder):
    (folder / 'models').mkdir(parents=True)
    (folder / 'configs').mkdir()
    for relative in ('train.py', 'models/model.py', 'configs/smoke.yaml'):
        (folder / relative).write_text('# fixture\n', encoding='utf-8')


def test_standalone_notebook_schema_and_no_outputs(tmp_path):
    notebook = json.loads(write_notebook(tmp_path / 'outside' / 'Train.ipynb').read_text(encoding='utf-8'))
    assert notebook['nbformat'] == 4
    for cell in notebook['cells']:
        if cell['cell_type'] == 'code':
            ast.parse(cell['source'])
            assert cell['execution_count'] is None and not cell['outputs']


@pytest.mark.skipif(shutil.which('git') is None, reason='git unavailable')
def test_actual_offline_clone_and_reuse_with_separate_storage(tmp_path, monkeypatch):
    remote = tmp_path / 'origin repo'
    project(remote)
    subprocess.run(['git', 'init', '-b', 'main', str(remote)], check=True, capture_output=True)
    subprocess.run(['git', '-C', str(remote), 'add', '.'], check=True)
    subprocess.run(['git', '-C', str(remote), '-c', 'user.name=Notebook Test', '-c', 'user.email=notebook@example.invalid',
                    'commit', '-m', 'fixture'], check=True, capture_output=True)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, 'path', list(sys.path))
    ns = namespace(tmp_path)
    ns['REPO_URL'] = str(remote)  # local transport avoids Git-for-Windows MSYS shell IPC in sandbox
    exec(source('LAUNCH_DIR ='), ns)
    assert ns['PROJECT_ROOT'] == (tmp_path / 'cloned source').resolve()
    assert ns['storage_path']('cache/test') == (tmp_path / 'persistent data/cache/test').resolve()
    assert ns['project_path']('configs/smoke.yaml').is_file()
    assert not ns['STORAGE_ROOT'].is_relative_to(ns['PROJECT_ROOT'])
    assert Path.cwd() == ns['PROJECT_ROOT']
    # A second run must not contact the remote or overwrite an existing checkout.
    ns['REPO_URL'] = 'file:///does-not-exist'
    exec(source('LAUNCH_DIR ='), ns)
    assert ns['is_project'](ns['PROJECT_ROOT'])


def test_refuses_nonempty_nonproject_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    ns = namespace(tmp_path)
    destination = Path(ns['PROJECT_DIR'])
    destination.mkdir()
    keep = destination / 'keep.txt'
    keep.write_text('keep')
    with pytest.raises(FileExistsError):
        exec(source('LAUNCH_DIR ='), ns)
    assert keep.read_text() == 'keep'


def test_hf_environment_token_stays_out_of_notebook_and_output(tmp_path, monkeypatch, capsys):
    value = 'test-only-environment-credential'
    monkeypatch.setenv('HF_TOKEN', value)
    ns = namespace(tmp_path)
    ns.update(PROJECT_ROOT=tmp_path, IN_COLAB=False)
    exec(source('HF_TOKEN = os.environ'), ns)
    assert ns['HF_TOKEN'] == value
    assert value not in capsys.readouterr().out
    saved = write_notebook(tmp_path / 'Train.ipynb').read_text(encoding='utf-8')
    assert value not in saved
    assert not (tmp_path / '.hf_token').exists()


def test_colab_secrets_and_hidden_input_do_not_print_or_persist(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv('HF_TOKEN', raising=False)
    value = 'test-only-colab-credential'
    userdata = SimpleNamespace(get=lambda key: value if key == 'HF_TOKEN' else None,
                               SecretNotFoundError=KeyError, NotebookAccessError=PermissionError, TimeoutException=TimeoutError)
    monkeypatch.setitem(sys.modules, 'google.colab', SimpleNamespace(userdata=userdata))
    ns = namespace(tmp_path)
    ns.update(PROJECT_ROOT=tmp_path, IN_COLAB=True)
    exec(source('HF_TOKEN = os.environ'), ns)
    assert ns['HF_TOKEN'] == value and os.environ['HF_TOKEN'] == value
    assert value not in capsys.readouterr().out
    monkeypatch.delenv('HF_TOKEN')
    import getpass
    monkeypatch.setattr(getpass, 'getpass', lambda prompt: 'test-only-hidden-credential')
    ns.update(IN_COLAB=False, PROMPT_FOR_HF_TOKEN=True)
    exec(source('HF_TOKEN = os.environ'), ns)
    assert ns['HF_TOKEN'] == 'test-only-hidden-credential'
    assert ns['HF_TOKEN'] not in capsys.readouterr().out
    assert not (tmp_path / '.hf_token').exists()


def data_namespace(tmp_path):
    from utils import atomic_json, load_config
    ns = namespace(tmp_path)
    ns.update(PROJECT_ROOT=Path(__file__).resolve().parents[1], STORAGE_ROOT=tmp_path / 'data outside source',
              DATA_SOURCE='existing', CACHE_DIR=None, RESUME_PATH=None, VERIFY_CACHE=True,
              CFG=load_config('configs/v3.yaml'), RUN_ROOT=tmp_path / 'results',
              atomic_json=atomic_json, display=lambda value: None, HF_TOKEN=None)
    ns['storage_path'] = lambda value: (ns['STORAGE_ROOT'] / value).resolve()
    return ns


def test_bundled_cache_relocated_outside_source_and_hashes_verified(tmp_path):
    ns = data_namespace(tmp_path)
    exec(source('import zipfile'), ns)
    assert ns['DATA_READY']
    assert (ns['CACHE_ROOT'] / 'manifest.json').is_file()
    assert ns['CACHE_ROOT'].is_relative_to(ns['STORAGE_ROOT'])
    assert ns['CFG']['data']['cache_dir'] == str(ns['CACHE_ROOT'])
    assert (ns['RUN_ROOT'] / 'notebook_config.json').is_file()


def test_cache_archive_rejects_traversal_before_extraction(tmp_path):
    import zipfile
    ns = data_namespace(tmp_path)
    ns['PROJECT_ROOT'] = tmp_path / 'bad source'
    ns['PROJECT_ROOT'].mkdir()
    with zipfile.ZipFile(ns['PROJECT_ROOT'] / 'precision_study_cache.zip', 'w') as archive:
        archive.writestr('cache/tinystories_integration/../escape.txt', 'invalid')
    with pytest.raises(ValueError):
        exec(source('import zipfile'), ns)
    assert not (ns['STORAGE_ROOT'] / 'cache/escape.txt').exists()
