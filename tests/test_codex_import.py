from __future__ import annotations

import builtins
from pathlib import Path

import tools.codex_import as codex_import


def test_import_skills_closes_each_source_file(tmp_path: Path, monkeypatch) -> None:
    source_root = tmp_path / "home" / ".codex" / "skills"
    source = source_root / "fixture-skill"
    source.mkdir(parents=True)
    (source / "SKILL.md").write_bytes(b"fixture")
    destination = tmp_path / "repo"
    readers = []

    class TrackedReader:
        def __init__(self, handle):
            self._handle = handle
            self.closed = False

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            self._handle.close()
            self.closed = True

        def read(self):
            return self._handle.read()

    def tracked_open(*args, **kwargs):
        reader = TrackedReader(builtins.open(*args, **kwargs))
        readers.append(reader)
        return reader

    monkeypatch.setattr(codex_import, "HOME", tmp_path / "home")
    monkeypatch.setattr(codex_import, "SKILLS_TO_IMPORT", ["fixture-skill"])
    monkeypatch.setattr(codex_import, "open", tracked_open, raising=False)

    manifest = codex_import.import_skills(destination)

    assert manifest[0]["file_count"] == 1
    assert (destination / ".codex" / "skills" / "fixture-skill" / "SKILL.md").read_bytes() == b"fixture"
    assert readers and all(reader.closed for reader in readers)
