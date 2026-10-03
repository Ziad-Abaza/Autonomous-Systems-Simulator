"""Tests for the track library service + thumbnails (headless)."""
import json
import os
import time

import pygame
import pytest

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
pygame.init()

from sim_project.library import TrackLibrary
from sim_project.serializer import EnvironmentProject
from sim_project.presets import create_oval_circuit, create_serpentine_track
from sim_ui.thumbnails import render_track_thumbnail, thumbnail_for_file


@pytest.fixture()
def lib(tmp_path):
    return TrackLibrary(str(tmp_path / "tracks"))


def _make(name="Test Track"):
    p = create_oval_circuit()
    p.name = name
    return p


class TestScan:
    def test_scan_empty(self, lib):
        assert lib.scan() == []

    def test_scan_extracts_metadata(self, lib):
        path = lib.create(_make("Oval One"))
        assets = lib.scan()
        assert len(assets) == 1
        a = assets[0]
        assert a.name == "Oval One"
        assert a.point_count == 8
        assert a.is_closed and a.length_m > 100
        assert not a.broken and not a.readonly
        assert a.modified > 0

    def test_broken_file_tolerated(self, lib):
        bad = os.path.join(lib.root, "broken.sim.json")
        with open(bad, "w") as f:
            f.write("{not json")
        assets = lib.scan()
        assert len(assets) == 1 and assets[0].broken
        assert assets[0].error

    def test_unique_names_on_create(self, lib):
        p1 = lib.create(_make("Dup"))
        p2 = lib.create(_make("Dup"))
        assert p1 != p2 and os.path.exists(p1) and os.path.exists(p2)


class TestOps:
    def test_duplicate(self, lib):
        src = lib.create(_make("Original"))
        dup = lib.duplicate(src)
        assert os.path.exists(dup)
        names = {a.name for a in lib.scan()}
        assert "Original (copy)" in names

    def test_rename_keeps_file(self, lib):
        path = lib.create(_make("Old Name"))
        lib.rename(path, "New Name")
        assert os.path.exists(path)
        assert lib.scan()[0].name == "New Name"

    def test_delete_removes_file_and_meta(self, lib):
        path = lib.create(_make())
        lib.set_favorite(path, True)
        lib.delete(path)
        assert not os.path.exists(path)
        assert lib._load_sidecar() == {}

    def test_favorite_and_description_sidecar(self, lib):
        path = lib.create(_make())
        lib.set_favorite(path, True)
        lib.set_description(path, "fast oval")
        a = lib.scan()[0]
        assert a.favorite and a.description == "fast oval"
        # .sim.json itself untouched by sidecar data
        with open(path) as f:
            assert "favorite" not in f.read()

    def test_recents_order(self, lib):
        p1 = lib.create(_make("A"))
        time.sleep(0.02)
        p2 = lib.create(_make("B"))
        lib.mark_opened(p1)
        time.sleep(0.02)
        lib.mark_opened(p2)
        rec = lib.recents()
        assert [os.path.basename(r.path) for r in rec] == [
            os.path.basename(p2), os.path.basename(p1)]

    def test_readonly_blocks_writes(self, tmp_path):
        ro_dir = tmp_path / "presets"
        ro_dir.mkdir()
        lib = TrackLibrary(str(ro_dir), readonly=True)
        with pytest.raises(PermissionError):
            lib.create(_make())
        with pytest.raises(PermissionError):
            lib.delete("x")


class TestThumbnails:
    def test_render_closed_track(self):
        road = create_oval_circuit().road_def.to_dict()
        surf = render_track_thumbnail(road)
        assert surf.get_size() == (192, 112)
        # non-empty pixels drawn
        px = pygame.surfarray.array3d(surf)
        assert (px != px[0, 0]).any()

    def test_render_open_track_and_empty(self):
        road = {"is_closed": False, "control_points": [
            {"x": 0, "y": 0, "width": 10}, {"x": 50, "y": 20, "width": 10}]}
        surf = render_track_thumbnail(road)
        assert surf.get_size() == (192, 112)
        empty = render_track_thumbnail({"control_points": []})
        assert empty.get_size() == (192, 112)

    def test_file_cache(self, lib, tmp_path):
        path = lib.create(_make())
        cache = lib.thumb_path_for(path)
        s1 = thumbnail_for_file(path, cache_path=cache)
        assert s1 and os.path.exists(cache)
        s2 = thumbnail_for_file(path, cache_path=cache)
        assert s2 is not None

    def test_missing_file_returns_none(self):
        assert thumbnail_for_file("nope.sim.json") is None
