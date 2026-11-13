import unittest

from tools.train_vision_v4 import CellCNN
from tools.train_vision_v3 import CLASSES
from tools.train_vision_v3 import CellCNN as ParentCNN


class ModelTests(unittest.TestCase):
    def test_original_architecture_state_dict_and_fixed_class_order(self):
        model = CellCNN()
        self.assertEqual(sum(p.numel() for p in model.parameters()), 104526)
        model.load_state_dict(ParentCNN().state_dict(), strict=True)
        from agents.vision_v4_bridge import CLASSES as RUNTIME_CLASSES

        self.assertEqual(tuple(CLASSES), tuple(RUNTIME_CLASSES))
