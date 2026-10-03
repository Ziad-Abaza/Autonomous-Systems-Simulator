"""
Environment Versioning and Configuration Diff Engine.
Calculates deterministic structural fingerprints (SHA-256), manages semantic
versioning (Major.Minor.Patch), detects changed subsystems, and computes
detailed configuration diffs between environment iterations.
"""

from __future__ import annotations
import json
import hashlib
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Optional, Tuple


@dataclass
class EnvironmentVersion:
    major: int = 1
    minor: int = 0
    patch: int = 0

    def to_string(self) -> str:
        return f"{self.major}.{self.minor}.{self.patch}"

    def increment_patch(self) -> None:
        self.patch += 1

    def increment_minor(self) -> None:
        self.minor += 1
        self.patch = 0

    def increment_major(self) -> None:
        self.major += 1
        self.minor = 0
        self.patch = 0

    @classmethod
    def from_string(cls, version_str: str) -> EnvironmentVersion:
        parts = version_str.strip().split(".")
        try:
            major = int(parts[0]) if len(parts) > 0 else 1
            minor = int(parts[1]) if len(parts) > 1 else 0
            patch = int(parts[2]) if len(parts) > 2 else 0
            return cls(major, minor, patch)
        except Exception:
            return cls(1, 0, 0)


@dataclass
class SubsystemDiff:
    subsystem: str
    has_changes: bool
    changes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ConfigurationDiffReport:
    version_a: str
    version_b: str
    hash_a: str
    hash_b: str
    identical: bool
    subsystem_diffs: List[SubsystemDiff] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version_a": self.version_a,
            "version_b": self.version_b,
            "hash_a": self.hash_a,
            "hash_b": self.hash_b,
            "identical": self.identical,
            "subsystems": [s.to_dict() for s in self.subsystem_diffs]
        }

    def format_text(self) -> str:
        if self.identical:
            return f"Environment {self.version_a} == {self.version_b}: Configurations are identical."

        lines = [f"Environment Diff: v{self.version_a} ({self.hash_a[:8]}) -> v{self.version_b} ({self.hash_b[:8]}):"]
        for s in self.subsystem_diffs:
            if s.has_changes:
                lines.append(f"  [{s.subsystem.upper()}]")
                for c in s.changes:
                    lines.append(f"    * {c}")
            else:
                lines.append(f"  [{s.subsystem.upper()}] unchanged")
        return "\n".join(lines)


class EnvironmentVersionManager:
    """
    Manages structural hashing, version increments, and configuration diffing.
    """
    @classmethod
    def compute_fingerprint(cls, data: Dict[str, Any]) -> str:
        """
        Computes deterministic SHA-256 fingerprint of environment data dict,
        excluding transient or metadata fields like 'timestamp'.
        """
        clean_data = cls._strip_metadata(data)
        serialized = json.dumps(clean_data, sort_keys=True)
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    @staticmethod
    def _strip_metadata(data: Any) -> Any:
        if isinstance(data, dict):
            return {
                k: EnvironmentVersionManager._strip_metadata(v)
                for k, v in data.items()
                if k not in ("version", "timestamp", "last_saved", "author",
                             "environment_version", "schema_version", "fingerprint")
            }
        elif isinstance(data, list):
            return [EnvironmentVersionManager._strip_metadata(item) for item in data]
        return data

    @classmethod
    def compare_configurations(
        cls,
        dict_a: Dict[str, Any],
        dict_b: Dict[str, Any],
        ver_a: str = "1.0.0",
        ver_b: str = "1.1.0"
    ) -> ConfigurationDiffReport:
        """
        Produces detailed structured comparison between two environment project dictionaries.
        """
        hash_a = cls.compute_fingerprint(dict_a)
        hash_b = cls.compute_fingerprint(dict_b)

        if hash_a == hash_b:
            return ConfigurationDiffReport(
                version_a=ver_a,
                version_b=ver_b,
                hash_a=hash_a,
                hash_b=hash_b,
                identical=True,
                subsystem_diffs=[]
            )

        diffs: List[SubsystemDiff] = []

        # 1. Track comparison
        t_a = dict_a.get("road_definition", {})
        t_b = dict_b.get("road_definition", {})
        t_changes = []
        if len(t_a.get("control_points", [])) != len(t_b.get("control_points", [])):
            t_changes.append(f"Control point count: {len(t_a.get('control_points', []))} -> {len(t_b.get('control_points', []))}")
        if t_a.get("is_closed") != t_b.get("is_closed"):
            t_changes.append(f"Closed loop: {t_a.get('is_closed')} -> {t_b.get('is_closed')}")
        diffs.append(SubsystemDiff("Track", len(t_changes) > 0, t_changes))

        # 2. Observation comparison (Phase 3 agent space or legacy schema)
        obs_changes = []
        agent_a = dict_a.get("agent", {})
        agent_b = dict_b.get("agent", {})
        
        if "observation_space" in agent_a or "observation_space" in agent_b:
            ch_a = {c.get("name"): c for c in agent_a.get("observation_space", {}).get("channels", [])}
            ch_b = {c.get("name"): c for c in agent_b.get("observation_space", {}).get("channels", [])}
            for name in set(ch_a.keys()).union(ch_b.keys()):
                if name not in ch_a:
                    obs_changes.append(f"+ channel '{name}'")
                elif name not in ch_b:
                    obs_changes.append(f"- channel '{name}'")
                else:
                    en_a, en_b = ch_a[name].get("enabled"), ch_b[name].get("enabled")
                    if en_a != en_b:
                        obs_changes.append(f"channel '{name}' enabled: {en_a} -> {en_b}")

        # Check legacy observation_schema fields
        obs_a = dict_a.get("observation_schema", {})
        obs_b = dict_b.get("observation_schema", {})
        for k in set(obs_a.keys()).union(obs_b.keys()):
            va, vb = obs_a.get(k), obs_b.get(k)
            if va != vb:
                obs_changes.append(f"{k}: {va} -> {vb}")
        diffs.append(SubsystemDiff("Observation", len(obs_changes) > 0, obs_changes))

        # 3. Action comparison (Phase 3 agent space or legacy config)
        act_changes = []
        if "action_space" in agent_a or "action_space" in agent_b:
            as_a = agent_a.get("action_space", {})
            as_b = agent_b.get("action_space", {})
            if as_a.get("space_type") != as_b.get("space_type"):
                act_changes.append(f"Space type: {as_a.get('space_type')} -> {as_b.get('space_type')}")
            ch_a = {c.get("name"): c for c in as_a.get("channels", [])}
            ch_b = {c.get("name"): c for c in as_b.get("channels", [])}
            for name in set(ch_a.keys()).union(ch_b.keys()):
                if name not in ch_a:
                    act_changes.append(f"+ action channel '{name}'")
                elif name not in ch_b:
                    act_changes.append(f"- action channel '{name}'")
                else:
                    for field in ("min_val", "max_val", "dead_zone", "rate_limit"):
                        va, vb = ch_a[name].get(field), ch_b[name].get(field)
                        if va != vb:
                            act_changes.append(f"channel '{name}' {field}: {va} -> {vb}")

        act_a = dict_a.get("action_config", {})
        act_b = dict_b.get("action_config", {})
        if act_a.get("type") != act_b.get("type"):
            act_changes.append(f"Space type: {act_a.get('type')} -> {act_b.get('type')}")
        if act_a.get("continuous_low") != act_b.get("continuous_low"):
            act_changes.append(f"Continuous low bounds: {act_a.get('continuous_low')} -> {act_b.get('continuous_low')}")
        if act_a.get("continuous_high") != act_b.get("continuous_high"):
            act_changes.append(f"Continuous high bounds: {act_a.get('continuous_high')} -> {act_b.get('continuous_high')}")
        diffs.append(SubsystemDiff("Action", len(act_changes) > 0, act_changes))

        # 4. Reward comparison (Phase 3 reward function or legacy config)
        rc_changes = []
        if "reward_function" in agent_a or "reward_function" in agent_b:
            comps_a = {c.get("component_id"): c for c in agent_a.get("reward_function", {}).get("components", [])}
            comps_b = {c.get("component_id"): c for c in agent_b.get("reward_function", {}).get("components", [])}
            for cid in set(comps_a.keys()).union(comps_b.keys()):
                if cid not in comps_a:
                    rc_changes.append(f"+ component '{cid}' (weight {comps_b[cid].get('weight')})")
                elif cid not in comps_b:
                    rc_changes.append(f"- component '{cid}'")
                else:
                    wa, wb = comps_a[cid].get("weight"), comps_b[cid].get("weight")
                    if wa != wb:
                        rc_changes.append(f"component '{cid}' weight: {wa} -> {wb}")
                    ea, eb = comps_a[cid].get("enabled"), comps_b[cid].get("enabled")
                    if ea != eb:
                        rc_changes.append(f"component '{cid}' enabled: {ea} -> {eb}")

        rc_a = dict_a.get("reward_config", {})
        rc_b = dict_b.get("reward_config", {})
        for k in set(rc_a.keys()).union(rc_b.keys()):
            va, vb = rc_a.get(k), rc_b.get(k)
            if va != vb:
                rc_changes.append(f"{k}: {va} -> {vb}")
        diffs.append(SubsystemDiff("Reward", len(rc_changes) > 0, rc_changes))

        # 5. Termination comparison (Phase 3 rules or legacy config)
        term_changes = []
        if "termination_rules" in agent_a or "termination_rules" in agent_b:
            rules_a = {r.get("rule_id"): r for r in agent_a.get("termination_rules", {}).get("rules", [])}
            rules_b = {r.get("rule_id"): r for r in agent_b.get("termination_rules", {}).get("rules", [])}
            for rid in set(rules_a.keys()).union(rules_b.keys()):
                if rid not in rules_a:
                    term_changes.append(f"+ termination rule '{rid}'")
                elif rid not in rules_b:
                    term_changes.append(f"- termination rule '{rid}'")
                else:
                    ea, eb = rules_a[rid].get("enabled"), rules_b[rid].get("enabled")
                    if ea != eb:
                        term_changes.append(f"rule '{rid}' enabled: {ea} -> {eb}")

        term_a = dict_a.get("termination_config", {})
        term_b = dict_b.get("termination_config", {})
        for k in set(term_a.keys()).union(term_b.keys()):
            va, vb = term_a.get(k), term_b.get(k)
            if va != vb:
                term_changes.append(f"{k}: {va} -> {vb}")
        diffs.append(SubsystemDiff("Termination", len(term_changes) > 0, term_changes))

        # 6. Scenario & Entities comparison
        ent_a = dict_a.get("entities", [])
        ent_b = dict_b.get("entities", [])
        scen_changes = []
        if len(ent_a) != len(ent_b):
            scen_changes.append(f"World entity count: {len(ent_a)} -> {len(ent_b)}")
        
        # Scenario Definition comparison
        s_def_a = dict_a.get("scenario_def", {})
        s_def_b = dict_b.get("scenario_def", {})
        for prop in ("name", "weather", "time_of_day", "surface_friction_mult", "ambient_light"):
            va, vb = s_def_a.get(prop), s_def_b.get(prop)
            if va is not None and vb is not None and va != vb:
                scen_changes.append(f"Scenario {prop}: {va} -> {vb}")

        diffs.append(SubsystemDiff("Entities & Scenario", len(scen_changes) > 0, scen_changes))

        return ConfigurationDiffReport(
            version_a=ver_a,
            version_b=ver_b,
            hash_a=hash_a,
            hash_b=hash_b,
            identical=False,
            subsystem_diffs=diffs
        )
