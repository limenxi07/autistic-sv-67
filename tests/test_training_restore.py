"""Archive restoration must preserve bytes without overwriting source files."""

import gzip
import hashlib
import io
import tarfile
import tempfile
import unittest
from pathlib import Path

from tools.restore_training import extract_verified


class RestoreTrainingTests(unittest.TestCase):
    def fixture(self, root, name="artifacts/v4/data/corpora/example.bin"):
        archive = root / "data.tar.gz"
        data = b"training example"
        with tarfile.open(archive, "w:gz") as tar:
            entry = tarfile.TarInfo(name)
            entry.size = len(data)
            tar.addfile(entry, io.BytesIO(data))
        manifest = {
            name: {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
        }
        return archive, manifest

    def test_restores_verified_members_and_rejects_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive, manifest = self.fixture(root)
            output = root / "restored"
            output.mkdir()
            self.assertEqual(
                extract_verified(archive, manifest, output, ("artifacts/v4/data/",)), 1
            )
            self.assertEqual(
                (output / next(iter(manifest))).read_bytes(), b"training example"
            )
            with self.assertRaises(FileExistsError):
                extract_verified(archive, manifest, output, ("artifacts/v4/data/",))

    def test_rejects_path_escape_and_corrupt_content(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive, manifest = self.fixture(root, "../escape")
            with self.assertRaises(ValueError):
                extract_verified(archive, manifest, root / "restored", ("../",))
            archive, manifest = self.fixture(root)
            manifest[next(iter(manifest))]["sha256"] = "0" * 64
            with self.assertRaises(ValueError):
                extract_verified(
                    archive, manifest, root / "restored", ("artifacts/v4/data/",)
                )

    def test_reports_missing_expected_members(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive, manifest = self.fixture(root)
            manifest["artifacts/v4/data/missing"] = {"bytes": 1, "sha256": "0" * 64}
            with self.assertRaisesRegex(ValueError, "Missing"):
                extract_verified(
                    archive, manifest, root / "restored", ("artifacts/v4/data/",)
                )

    def test_reads_consecutive_gzip_members(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive, manifest = self.fixture(root)
            raw = gzip.decompress(archive.read_bytes())
            archive.write_bytes(gzip.compress(raw[:400]) + gzip.compress(raw[400:]))
            self.assertEqual(
                extract_verified(
                    archive, manifest, root / "restored", ("artifacts/v4/data/",)
                ),
                1,
            )
