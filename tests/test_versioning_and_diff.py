"""
Unit tests for Environment Versioning, structural hashing (fingerprint),
and Configuration Diff between versions.
"""

import pytest

from sim_env.versioning import (
    EnvironmentVersion,
    EnvironmentVersionManager,
    ConfigurationDiffReport
)
from sim_project.presets import create_oval_circuit


def test_semantic_versioning_increments():
    ver = EnvironmentVersion(1, 0, 0)
    assert ver.to_string() == "1.0.0"

    ver.increment_patch()
    assert ver.to_string() == "1.0.1"

    ver.increment_minor()
    assert ver.to_string() == "1.1.0"

    ver.increment_major()
    assert ver.to_string() == "2.0.0"


def test_structural_fingerprint_determinism():
    proj1 = create_oval_circuit()
    proj2 = create_oval_circuit()

    fp1 = EnvironmentVersionManager.compute_fingerprint(proj1.to_dict())
    fp2 = EnvironmentVersionManager.compute_fingerprint(proj2.to_dict())

    assert len(fp1) == 64  # SHA-256 hex
    assert fp1 == fp2      # Identical configurations have identical fingerprint

    # Modify one parameter in proj2
    proj2.road_def.default_friction = 1.8
    fp3 = EnvironmentVersionManager.compute_fingerprint(proj2.to_dict())
    assert fp1 != fp3      # Changed configuration yields different fingerprint


def test_configuration_diff_generation():
    proj_v1 = create_oval_circuit()
    dict_v1 = proj_v1.to_dict()

    # Create v2 with reward change and observation change
    proj_v2 = create_oval_circuit()
    proj_v2.reward_config.weight_centering = 1.5
    proj_v2.observation_schema.include_camera_rgb = True
    dict_v2 = proj_v2.to_dict()

    diff_report = EnvironmentVersionManager.compare_configurations(
        dict_a=dict_v1,
        dict_b=dict_v2,
        ver_a="1.0.0",
        ver_b="1.1.0"
    )

    assert diff_report.identical is False
    assert diff_report.version_a == "1.0.0"
    assert diff_report.version_b == "1.1.0"

    text = diff_report.format_text()
    assert "REWARD" in text
    assert "OBSERVATION" in text
    assert "weight_centering" in text
    assert "include_camera_rgb" in text
