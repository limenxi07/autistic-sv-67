import ast
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from tools.package_vision_v3 import package

ROOT = Path(__file__).resolve().parents[1]
EXPORT = ROOT / "artifacts/v3/models/selected-export"


class PackageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.folder = Path(cls.temporary.name)
        cls.source = package(ROOT, EXPORT, cls.folder)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_package_under_decimal_2mb(self):
        self.assertLess(self.source.stat().st_size, 1_600_000)
        ast.parse(self.source.read_text(), feature_version=(3, 10))
        other = package(ROOT, EXPORT, self.folder / "repeat")
        self.assertEqual(other.read_bytes(), self.source.read_bytes())

    def run_worker(self, mode):
        worker = r"""
import builtins, io, json, socket, sys
from pathlib import Path
from grid_adventure.grid import GridState, step
from grid_adventure.env import ImageObservation
from grid_adventure.step import Action
from tests.planner_p1_fixtures import corridor
code=compile('from grid_adventure.grid import GridState\nfrom grid_adventure.env import ImageObservation\n'+Path(sys.argv[1]).read_text(),'<submission>','exec')
state=corridor()
if sys.argv[2]=='image':
    import torch, numpy
    from PIL import Image
    from tests.vision_v3_fixtures import vision_observation
    observations=[vision_observation(state),vision_observation(step(state,Action.RIGHT))]
original_import=builtins.__import__
def checked_import(name,*args,**kwargs):
    if name.split('.')[0] in ('agents','tools','tests'):
        raise AssertionError('local runtime dependency: '+name)
    return original_import(name,*args,**kwargs)
def denied(*args,**kwargs): raise AssertionError('external file or network access')
builtins.__import__=checked_import
builtins.open=io.open=denied
socket.socket=socket.create_connection=denied
ns={'__name__':'unregistered_v3'}
exec(code,ns)
a=ns['Agent']()
if sys.argv[2]=='image':
    loads=[]
    original_load=torch.jit.load
    def memory_load(buffer,*args,**kwargs):
        assert isinstance(buffer,io.BytesIO)
        loads.append(True)
        return original_load(buffer,*args,**kwargs)
    torch.jit.load=memory_load
    assert [a.step(o) for o in observations]==[Action.RIGHT,Action.RIGHT]
    assert loads==[True]
    print(json.dumps({'embedded':True,'unregistered':True,'actions':2}))
else:
    assert a.step(state)==Action.RIGHT
    assert 'torch' not in sys.modules
    print(json.dumps({'torch_loaded':False,'unregistered':True}))
"""
        result = subprocess.run(
            [sys.executable, "-c", worker, str(self.source), mode],
            cwd=ROOT,
            env={**os.environ, "MPLCONFIGDIR": "/private/tmp/v3-mpl"},
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
        return json.loads(result.stdout.splitlines()[-1])

    def test_source_runs_with_prelude_without_module_registration(self):
        self.assertTrue(self.run_worker("structured")["unregistered"])

    def test_task1_does_not_load_vision_model(self):
        self.assertFalse(self.run_worker("structured")["torch_loaded"])

    def test_packaged_agent_works_without_assets_or_network(self):
        self.assertEqual(self.run_worker("image")["actions"], 2)

    def test_model_load_is_from_embedded_bytes(self):
        self.assertTrue(self.run_worker("image")["embedded"])
