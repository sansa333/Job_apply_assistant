import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from app.agent.skill_runtime import (
    JOB_APPLICATION_SKILL,
    SkillRegistry,
    SkillSession,
    auto_activate_for_request,
    create_skill_session,
)
from app.agent.tools import build_job_tools
from app.schemas import AgentRequest, ApplicationIntent


class _UnusedService:
    def analyze_scoped_fit(self, req):  # pragma: no cover - guard should prevent this call
        raise AssertionError("domain tool ran before its Skill was active")


class SkillRuntimeTests(unittest.TestCase):
    @staticmethod
    def _write_skill(root: Path, name: str, frontmatter: str, body: str = "Use structured states.\n") -> None:
        skill_root = root / name
        skill_root.mkdir(parents=True, exist_ok=True)
        (skill_root / "SKILL.md").write_text(
            f"---\n{frontmatter}\n---\n\n{body}",
            encoding="utf-8",
        )

    def _registry(self, root: Path) -> SkillRegistry:
        skill_root = root / JOB_APPLICATION_SKILL
        references = skill_root / "references"
        references.mkdir(parents=True)
        (skill_root / "SKILL.md").write_text(
            "---\n"
            f"name: {JOB_APPLICATION_SKILL}\n"
            "description: Controlled job workflow\n"
            "---\n\n"
            "Use structured states. Read [routing](references/routing.md).\n",
            encoding="utf-8",
        )
        (references / "routing.md").write_text("Use status and next_action.", encoding="utf-8")
        return SkillRegistry(root)

    def test_reference_can_only_be_read_after_declared_skill_activation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            session = SkillSession(self._registry(Path(tmp)))

            with self.assertRaisesRegex(ValueError, "not active"):
                session.read_reference(JOB_APPLICATION_SKILL, "routing.md")

            session.activate(JOB_APPLICATION_SKILL)
            self.assertEqual(session.read_reference(JOB_APPLICATION_SKILL, "routing.md"), "Use status and next_action.")

    def test_yaml_metadata_controls_requirements_and_always_loading(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_skill(
                root,
                "ready-skill",
                "name: ready-skill\n"
                "description: >\n"
                "  Supports YAML metadata and always-on loading.\n"
                "metadata:\n"
                "  nanobot:\n"
                "    always: true\n"
                "    requires:\n"
                "      bins: [test-skill-cli]\n"
                "      env: [TEST_SKILL_ENV]",
            )
            registry = SkillRegistry(root)
            skill = registry.catalog()[0]
            self.assertTrue(skill.always)
            self.assertEqual(skill.requires_bins, ("test-skill-cli",))
            self.assertEqual(skill.requires_env, ("TEST_SKILL_ENV",))

            with patch("app.agent.skill_runtime.shutil.which", return_value=None), patch.dict(os.environ, {}, clear=True):
                available, reason = registry.availability(skill)
                self.assertFalse(available)
                self.assertIn("CLI: test-skill-cli", reason)
                self.assertIn("ENV: TEST_SKILL_ENV", reason)
                with self.assertRaisesRegex(ValueError, "requirements unavailable"):
                    SkillSession(registry).activate("ready-skill")

            with patch("app.agent.skill_runtime.shutil.which", return_value="/bin/test-skill-cli"), patch.dict(
                os.environ, {"TEST_SKILL_ENV": "present"}, clear=True
            ):
                self.assertEqual(registry.always_skill_names(), ["ready-skill"])
                fake_settings = SimpleNamespace(
                    skills_dir=root,
                    agent_workspace_skills_dir=None,
                    agent_allow_workspace_skill_overrides=False,
                    agent_approved_workspace_skills="",
                    agent_disabled_skills="",
                    agent_always_on_skills="",
                )
                with patch("app.agent.skill_runtime.settings", fake_settings):
                    session = create_skill_session()
                self.assertTrue(session.is_active("ready-skill"))

    def test_disabled_skill_is_hidden_and_cannot_be_activated(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._registry(root)
            registry = SkillRegistry(root, disabled_skills={JOB_APPLICATION_SKILL})

            self.assertEqual(registry.catalog(), [])
            with self.assertRaisesRegex(ValueError, "Unknown skill"):
                SkillSession(registry).activate(JOB_APPLICATION_SKILL)
            activated = auto_activate_for_request(
                SkillSession(registry),
                AgentRequest(goal="生成申请包", intent=ApplicationIntent.APPLICATION_PACKAGE),
            )
            self.assertEqual(activated, [])

    def test_workspace_skill_requires_explicit_approval_to_override_trusted_skill(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            trusted_root = root / "trusted"
            workspace_root = root / "workspace"
            self._write_skill(
                trusted_root,
                "example",
                "name: example\ndescription: Trusted Skill",
                "trusted instructions\n",
            )
            self._write_skill(
                workspace_root,
                "example",
                "name: example\ndescription: Workspace Skill",
                "workspace instructions\n",
            )

            default_registry = SkillRegistry(
                trusted_root,
                workspace_skills_dir=workspace_root,
                allow_workspace_overrides=False,
                approved_workspace_skills={"example"},
            )
            self.assertEqual(default_registry.get("example").source, "trusted")

            unapproved_registry = SkillRegistry(
                trusted_root,
                workspace_skills_dir=workspace_root,
                allow_workspace_overrides=True,
                approved_workspace_skills=set(),
            )
            self.assertEqual(unapproved_registry.get("example").source, "trusted")

            approved_registry = SkillRegistry(
                trusted_root,
                workspace_skills_dir=workspace_root,
                allow_workspace_overrides=True,
                approved_workspace_skills={"example"},
            )
            skill = approved_registry.get("example")
            self.assertEqual(skill.source, "workspace")
            self.assertEqual(skill.body, "workspace instructions")

    def test_job_intent_auto_activates_the_job_workflow(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            session = SkillSession(self._registry(Path(tmp)))
            activated = auto_activate_for_request(
                session,
                AgentRequest(goal="生成申请包", intent=ApplicationIntent.APPLICATION_PACKAGE),
            )

        self.assertEqual(activated, [JOB_APPLICATION_SKILL])
        self.assertTrue(session.is_active(JOB_APPLICATION_SKILL))

    def test_domain_tool_returns_machine_readable_activation_requirement(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            session = SkillSession(self._registry(Path(tmp)))
            tools = {tool.name: tool for tool in build_job_tools(_UnusedService(), session)}
            raw = tools["analyze_job_fit"].invoke(
                {"company_name": "Acme", "job_title": "RAG Engineer", "resume_text": "Python"}
            )

        result = json.loads(raw)
        self.assertEqual(result["status"], "skill_not_active")
        self.assertEqual(result["next_action"], "activate_skill")


if __name__ == "__main__":
    unittest.main()
