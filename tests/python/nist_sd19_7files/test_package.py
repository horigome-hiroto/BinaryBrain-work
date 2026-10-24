"""Exercise the seven-file distribution in isolation from the earlier package."""
import ast
import json
from pathlib import Path
import subprocess
import sys
import zipfile

REPO = Path(__file__).resolve().parents[3]
PACKAGE = REPO / 'samples/python/nist_sd19_7files'
ORIGINAL = REPO / 'samples/python/nist_sd19'
FILES = {
    'g_research_miou_replicate_fixed.py', 'fpga_magazin_network_samepadding.py',
    'experiment_config.py', 'experiment_common.py', 'experiment_dataset.py',
    'experiment_metrics.py', 'experiment_lut.py',
}


def test_seven_file_distribution(tmp_path):
    archive = REPO / 'downloads/nist_sd19_7files.zip'
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        assert len(z.namelist()) == 7
        assert {Path(n).name for n in z.namelist()} == FILES
        z.extractall(tmp_path)
    isolated = tmp_path / 'nist_sd19_7files'
    for name in FILES:
        assert (isolated / name).read_bytes() == (PACKAGE / name).read_bytes()
        for node in ast.walk(ast.parse((isolated / name).read_text())):
            if isinstance(node, ast.ImportFrom):
                assert node.module not in {'config', 'download_sd19', 'smoke', 'camera', 'benchmark_loader', 'run'}
    # Re-run all existing data, metric, retained-loop, and wiring checks against
    # the extracted seven-file ZIP, not the full earlier package.
    tests = (ORIGINAL / 'tests/test_sd19.py').read_text()
    tests = tests.replace('from config import ', 'from experiment_config import ')
    tests = tests.replace('str(Path(__file__).resolve().parents[1])', repr(str(isolated)))
    tests = tests.replace("(Path(__file__).parents[1]/'provenance.json')", repr(str(ORIGINAL / 'provenance.json')))
    # The original expression calls read_text on the path, so retain a Path.
    tests = tests.replace(repr(str(ORIGINAL / 'provenance.json')) + '.read_text()',
                          'Path(' + repr(str(ORIGINAL / 'provenance.json')) + ').read_text()')
    suite = tmp_path / 'test_extracted.py'
    suite.write_text(tests)
    result = subprocess.run([sys.executable, '-m', 'pytest', str(suite), '-q'],
                            cwd=isolated, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert '9 passed' in result.stdout


def test_training_and_network_definitions_unchanged():
    """Consolidation only changes CLI dispatch and helper locations."""
    names = {
        'experiment_common.py': {'_train_epoch_whole', '_train_epoch_tail',
            'evaluate_segmentation', 'train_whole_model', 'train_tail_model', 'distillation_layer'},
        'g_research_miou_replicate_fixed.py': {'create_network', 'run_experiment'},
        'fpga_magazin_network_samepadding.py': {'create_network', 'run_experiment'},
        'experiment_dataset.py': {'preprocess', '_settings', 'make_data_loaders'},
    }
    for filename, functions in names.items():
        def nodes(path):
            return {n.name: ast.dump(n, include_attributes=False)
                    for n in ast.parse(path.read_text()).body
                    if isinstance(n, ast.FunctionDef) and n.name in functions}
        assert nodes(PACKAGE / filename) == nodes(ORIGINAL / filename)


def test_integrated_download_and_cli_modes(tmp_path):
    # Use a local small ZIP through a mocked HTTPS response. No large download
    # and no BinaryBrain runtime are needed to verify the integrated download.
    script = r'''
import hashlib, io, json, sys, zipfile
from pathlib import Path
from unittest.mock import patch
from PIL import Image
sys.path.insert(0, sys.argv[1])
import experiment_dataset as dataset
from experiment_config import build_arg_parser
root = Path(sys.argv[2])
image = io.BytesIO()
Image.new('L', (128, 128), 255).save(image, format='PNG')
archive = io.BytesIO()
with zipfile.ZipFile(archive, 'w') as z:
    for i in range(10):
        z.writestr('by_class/%02x/hsf_0/a.png' % (48+i), image.getvalue())
    z.writestr('by_class/41/hsf_0/letter.png', image.getvalue())
raw = archive.getvalue()
class Response(io.BytesIO):
    headers = {'Content-Length': str(len(raw))}
with patch.object(dataset.urllib.request, 'urlopen', return_value=Response(raw)) as get:
    dataset.download_sd19(root)
    assert get.call_args.args[0] == dataset.SD19_DOWNLOAD_URL
    assert get.call_args.kwargs['timeout'] == 60
assert len(list(root.rglob('*.png'))) == 10
assert not (root/'by_class/41').exists()
assert json.loads((root/'download.json').read_text())['sha256'] == hashlib.sha256(raw).hexdigest()
# A second call reuses existing sources rather than downloading again.
with patch.object(dataset.urllib.request, 'urlopen', side_effect=AssertionError('Unexpected redownload')):
    dataset.download_sd19(root)
for driver, kind in [('fpga_magazin_network_samepadding','fpga_magazine'),
                     ('g_research_miou_replicate_fixed','g_research')]:
    import importlib
    module = importlib.import_module(driver)
    import experiment_common as common
    for args, name in [(['--smoke-test'], 'run_smoke_test'),
                       (['--camera-run-dir','saved-run'], 'run_camera_preview'),
                       (['--benchmark-loader'], 'run_loader_benchmark')]:
        with patch.object(sys, 'argv', [driver]+args), patch.object(common, name) as action, patch.object(module, 'run_experiment') as train:
            module.main()
            assert action.call_count == 1
            assert action.call_args.kwargs['model_kind'] == kind
            train.assert_not_called()
'''
    result = subprocess.run([sys.executable, '-c', script, str(PACKAGE), str(tmp_path / 'raw')],
                            cwd=tmp_path, text=True, capture_output=True)
    assert result.returncode == 0, result.stdout + result.stderr
