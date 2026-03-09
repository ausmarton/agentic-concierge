"""Tests for externalized specialist templates loaded from YAML.

Verifies:
- Shipped default YAML files load correctly and match v1 values.
- User YAML override modifies existing templates.
- New user templates are discovered and loadable.
- Cache invalidation via reload_templates().
- Malformed YAML falls back gracefully.
- Role descriptions resolve correctly from YAML role keys.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import yaml

from agentic_concierge.infrastructure.specialists.dynamic_pack import (
    PackTemplate,
    pack_templates,
    reload_templates,
)
from agentic_concierge.infrastructure.specialists.finish_schemas import FINISH_SCHEMAS
from agentic_concierge.infrastructure.specialists.prompts import ROLE_DESCRIPTIONS


@pytest.fixture(autouse=True)
def _clean_caches():
    """Ensure each test starts and ends with clean caches."""
    reload_templates()
    yield
    os.environ.pop("XDG_CONFIG_HOME", None)
    reload_templates()


# ---------------------------------------------------------------------------
# Shipped YAML files exist and load correctly
# ---------------------------------------------------------------------------


class TestShippedDefaults:

    def test_template_yaml_files_exist(self):
        from agentic_concierge.config.external import shipped_defaults_dir
        templates_dir = shipped_defaults_dir() / "templates"
        assert templates_dir.is_dir()
        expected = {"engineering.yaml", "research.yaml", "enterprise_research.yaml"}
        actual = {f.name for f in templates_dir.iterdir() if f.suffix == ".yaml"}
        assert expected == actual

    def test_three_templates_loaded(self):
        tpls = pack_templates()
        assert set(tpls.keys()) == {"engineering", "research", "enterprise_research"}

    def test_all_templates_are_pack_template_instances(self):
        for tpl in pack_templates().values():
            assert isinstance(tpl, PackTemplate)

    def test_engineering_tools(self):
        tpl = pack_templates()["engineering"]
        assert set(tpl.tool_names) == {"shell", "read_file", "write_file", "list_files", "run_tests"}

    def test_engineering_quality_gates(self):
        assert "tests_verified" in pack_templates()["engineering"].quality_gates

    def test_engineering_finish_schema(self):
        tpl = pack_templates()["engineering"]
        assert tpl.finish_schema is FINISH_SCHEMAS["code"]

    def test_research_tools(self):
        tpl = pack_templates()["research"]
        assert "web_search" in tpl.tool_names
        assert "fetch_url" in tpl.tool_names

    def test_research_no_quality_gates(self):
        assert pack_templates()["research"].quality_gates == []

    def test_research_finish_schema(self):
        tpl = pack_templates()["research"]
        assert tpl.finish_schema is FINISH_SCHEMAS["quick_answer"]

    def test_enterprise_research_has_cross_run_search(self):
        tpl = pack_templates()["enterprise_research"]
        assert "cross_run_search" in tpl.tool_names

    def test_enterprise_research_finish_schema(self):
        tpl = pack_templates()["enterprise_research"]
        assert tpl.finish_schema is FINISH_SCHEMAS["enterprise_report"]


# ---------------------------------------------------------------------------
# Role description resolution
# ---------------------------------------------------------------------------


class TestRoleDescriptions:

    def test_role_descriptions_registry_has_all_roles(self):
        assert "engineering" in ROLE_DESCRIPTIONS
        assert "research" in ROLE_DESCRIPTIONS
        assert "enterprise_research" in ROLE_DESCRIPTIONS

    def test_engineering_role_description_populated(self):
        tpl = pack_templates()["engineering"]
        assert "software engineering" in tpl.role_description.lower()

    def test_research_role_description_populated(self):
        tpl = pack_templates()["research"]
        assert "research" in tpl.role_description.lower()

    def test_enterprise_role_description_populated(self):
        tpl = pack_templates()["enterprise_research"]
        assert "enterprise" in tpl.role_description.lower()


# ---------------------------------------------------------------------------
# User override
# ---------------------------------------------------------------------------


class TestUserOverride:

    def test_override_changes_template_tools(self, tmp_path):
        templates_dir = tmp_path / "concierge" / "templates"
        templates_dir.mkdir(parents=True)
        override = {
            "template_id": "engineering",
            "tools": ["shell", "read_file", "write_file", "list_files", "run_tests", "web_search"],
            "role": "engineering",
            "finish_schema": "code",
            "quality_gates": ["tests_verified"],
        }
        (templates_dir / "engineering.yaml").write_text(yaml.dump(override))
        os.environ["XDG_CONFIG_HOME"] = str(tmp_path)
        reload_templates()

        tpl = pack_templates()["engineering"]
        assert "web_search" in tpl.tool_names

    def test_new_user_template_discovered(self, tmp_path):
        templates_dir = tmp_path / "concierge" / "templates"
        templates_dir.mkdir(parents=True)
        custom = {
            "template_id": "data_analysis",
            "tools": ["shell", "read_file", "write_file"],
            "role": "engineering",
            "finish_schema": "general",
            "quality_gates": [],
        }
        (templates_dir / "data_analysis.yaml").write_text(yaml.dump(custom))
        os.environ["XDG_CONFIG_HOME"] = str(tmp_path)
        reload_templates()

        tpls = pack_templates()
        assert "data_analysis" in tpls
        assert tpls["data_analysis"].tool_names == ["shell", "read_file", "write_file"]
        # Original templates still present
        assert "engineering" in tpls
        assert "research" in tpls

    def test_override_finish_schema(self, tmp_path):
        templates_dir = tmp_path / "concierge" / "templates"
        templates_dir.mkdir(parents=True)
        override = {
            "template_id": "research",
            "tools": ["web_search", "fetch_url", "read_file", "list_files"],
            "role": "research",
            "finish_schema": "research_report",
            "quality_gates": [],
        }
        (templates_dir / "research.yaml").write_text(yaml.dump(override))
        os.environ["XDG_CONFIG_HOME"] = str(tmp_path)
        reload_templates()

        tpl = pack_templates()["research"]
        assert tpl.finish_schema is FINISH_SCHEMAS["research_report"]


# ---------------------------------------------------------------------------
# Cache invalidation
# ---------------------------------------------------------------------------


class TestCacheInvalidation:

    def test_reload_clears_cache(self, tmp_path):
        tpls1 = pack_templates()
        assert "engineering" in tpls1

        # Add a new template
        templates_dir = tmp_path / "concierge" / "templates"
        templates_dir.mkdir(parents=True)
        custom = {
            "template_id": "custom_spec",
            "tools": ["shell"],
            "role": "engineering",
            "finish_schema": "general",
        }
        (templates_dir / "custom_spec.yaml").write_text(yaml.dump(custom))
        os.environ["XDG_CONFIG_HOME"] = str(tmp_path)
        reload_templates()

        tpls2 = pack_templates()
        assert "custom_spec" in tpls2


# ---------------------------------------------------------------------------
# Error resilience
# ---------------------------------------------------------------------------


class TestErrorResilience:

    def test_malformed_yaml_uses_fallback(self, tmp_path):
        templates_dir = tmp_path / "concierge" / "templates"
        templates_dir.mkdir(parents=True)
        # Write invalid YAML that overwrites engineering
        (templates_dir / "engineering.yaml").write_text("not: [valid: yaml")
        os.environ["XDG_CONFIG_HOME"] = str(tmp_path)
        reload_templates()

        # Should still have templates (shipped defaults + valid overrides)
        tpls = pack_templates()
        assert len(tpls) >= 2  # at least research and enterprise_research

    def test_unknown_role_skips_template(self, tmp_path):
        templates_dir = tmp_path / "concierge" / "templates"
        templates_dir.mkdir(parents=True)
        bad = {
            "template_id": "bad_role",
            "tools": ["shell"],
            "role": "nonexistent_role",
            "finish_schema": "general",
        }
        (templates_dir / "bad_role.yaml").write_text(yaml.dump(bad))
        os.environ["XDG_CONFIG_HOME"] = str(tmp_path)
        reload_templates()

        tpls = pack_templates()
        assert "bad_role" not in tpls
        # Valid templates still present
        assert "engineering" in tpls

    def test_unknown_finish_schema_defaults_to_general(self, tmp_path):
        templates_dir = tmp_path / "concierge" / "templates"
        templates_dir.mkdir(parents=True)
        tpl = {
            "template_id": "bad_schema",
            "tools": ["shell"],
            "role": "engineering",
            "finish_schema": "nonexistent_schema",
        }
        (templates_dir / "bad_schema.yaml").write_text(yaml.dump(tpl))
        os.environ["XDG_CONFIG_HOME"] = str(tmp_path)
        reload_templates()

        tpls = pack_templates()
        assert "bad_schema" in tpls
        assert tpls["bad_schema"].finish_schema is FINISH_SCHEMAS["general"]

    def test_missing_template_id_skipped(self, tmp_path):
        templates_dir = tmp_path / "concierge" / "templates"
        templates_dir.mkdir(parents=True)
        bad = {
            "tools": ["shell"],
            "role": "engineering",
            "finish_schema": "general",
        }
        (templates_dir / "no_id.yaml").write_text(yaml.dump(bad))
        os.environ["XDG_CONFIG_HOME"] = str(tmp_path)
        reload_templates()

        tpls = pack_templates()
        assert "no_id" not in tpls
        # Valid templates still present
        assert "engineering" in tpls
