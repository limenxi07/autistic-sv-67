"""Exercise archive restoration and the historical source boundary."""

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tools import restore_archives
from tools.historical_sources import historical_bytes


class ReproducibilityTests(unittest.TestCase):
    def test_restore_and_preserve_existing_files(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "tools").mkdir()
            (root / "docs/reproducibility").mkdir(parents=True)
            data = b"example archive bytes"
            (root / "part").write_bytes(data)
            digest = hashlib.sha256(data).hexdigest()
            record = {
                "path": "archive",
                "sha256": digest,
                "parts": [{"path": "part", "bytes": len(data), "sha256": digest}],
            }
            (root / "docs/reproducibility/archive-parts.json").write_text(
                json.dumps([record])
            )
            unrelated = root / "archive.restoring"
            unrelated.write_bytes(b"other invocation")
            with patch.object(
                restore_archives, "__file__", str(root / "tools/restore_archives.py")
            ):
                restore_archives.main()
                self.assertEqual((root / "archive").read_bytes(), data)
                self.assertEqual(unrelated.read_bytes(), b"other invocation")
                (root / "archive").write_bytes(b"different")
                with self.assertRaises(ValueError):
                    restore_archives.main()
                self.assertEqual((root / "archive").read_bytes(), b"different")
                (root / "archive").unlink()
                (root / "part").write_bytes(b"bad chunk")
                with self.assertRaises(ValueError):
                    restore_archives.main()
                self.assertFalse((root / "archive").exists())

    def test_original_source_and_current_source_are_distinct(self):
        original = historical_bytes("agents/p3.py")
        current = historical_bytes("agents/hierarchical_p3.py")
        self.assertNotEqual(original, current)
        self.assertEqual(
            hashlib.sha256(original).hexdigest(),
            "d487728413d4ec7d24f7a8cbf52f4769f6830653db1549d22c82afb1254bc2f4",
        )


if __name__ == "__main__":
    unittest.main()
