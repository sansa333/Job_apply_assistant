from __future__ import annotations

import json
import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field
import yaml

from app.config import settings
from app.schemas import AgentRequest, ApplicationIntent


JOB_APPLICATION_SKILL = "job-application-workflow"
_REFERENCE_PATTERN = re.compile(r"\]\(references/([A-Za-z0-9_.-]+\.md)\)")
_JOB_INTENTS = {
    ApplicationIntent.FIT_ANALYSIS,
    ApplicationIntent.COVER_LETTER,
    ApplicationIntent.APPLICATION_EMAIL,
    ApplicationIntent.INTERVIEW_PREP,
    ApplicationIntent.APPLICATION_PACKAGE,
}
_JOB_KEYWORDS = (
    "岗位",
    "求职",
    "投递",
    "申请包",
    "求职信",
    "面试准备",
    "cover letter",
    "application package",
    "job match",
)


@dataclass(frozen=True)
class SkillDefinition:
    name: str
    description: str
    root: Path
    body: str
    references: tuple[str, ...]
    source: str
    requires_bins: tuple[str, ...]
    requires_env: tuple[str, ...]
    always: bool


class SkillRegistry:
    """Discover trusted Skills with optional, explicitly approved workspace overrides."""

    def __init__(
        self,
        skills_dir: Path,
        *,
        workspace_skills_dir: Path | None = None,
        allow_workspace_overrides: bool = False,
        approved_workspace_skills: set[str] | None = None,
        disabled_skills: set[str] | None = None,
    ):
        self.skills_dir = skills_dir
        self.workspace_skills_dir = workspace_skills_dir
        self.allow_workspace_overrides = allow_workspace_overrides
        self.approved_workspace_skills = approved_workspace_skills or set()
        self.disabled_skills = disabled_skills or set()
        self._skills = self._discover()

    def _discover(self) -> dict[str, SkillDefinition]:
        skills = self._discover_from_dir(self.skills_dir, source="trusted")
        if not self.allow_workspace_overrides or not self.workspace_skills_dir:
            return skills

        workspace_skills = self._discover_from_dir(self.workspace_skills_dir, source="workspace")
        for name, skill in workspace_skills.items():
            if name in self.approved_workspace_skills:
                skills[name] = skill
        return skills

    def _discover_from_dir(self, root: Path, *, source: str) -> dict[str, SkillDefinition]:
        if not root.exists():
            return {}

        skills: dict[str, SkillDefinition] = {}
        for skill_md in sorted(root.glob("*/SKILL.md")):
            skill = self._parse_skill_file(skill_md, source=source)
            if skill is not None and skill.name not in self.disabled_skills:
                skills[skill.name] = skill
        return skills

    def _parse_skill_file(self, skill_md: Path, *, source: str) -> SkillDefinition | None:
        text = skill_md.read_text(encoding="utf-8")
        parsed = self._parse_skill(text)
        if parsed is None:
            return None
        name, description, body, frontmatter = parsed
        if not name or not description:
            return None

        metadata = self._skill_metadata(frontmatter)
        requires = frontmatter.get("requires", metadata.get("requires", {}))
        requires_bins, requires_env = self._parse_requirements(requires)
        always = self._as_bool(frontmatter.get("always", metadata.get("always", False)))
        references = tuple(
            reference
            for reference in _REFERENCE_PATTERN.findall(body)
            if (skill_md.parent / "references" / reference).is_file()
        )
        return SkillDefinition(
            name=name,
            description=description,
            root=skill_md.parent,
            body=body.strip(),
            references=references,
            source=source,
            requires_bins=requires_bins,
            requires_env=requires_env,
            always=always,
        )

    @staticmethod
    def _parse_skill(text: str) -> tuple[str, str, str, dict[str, Any]] | None:
        if not text.startswith("---"):
            return None
        match = re.match(r"^---\s*\r?\n(.*?)\r?\n---\s*\r?\n?", text, re.DOTALL)
        if match is None:
            return None
        try:
            frontmatter = yaml.safe_load(match.group(1))
        except yaml.YAMLError:
            return None
        if not isinstance(frontmatter, dict):
            return None
        name = str(frontmatter.get("name") or "").strip()
        description = str(frontmatter.get("description") or "").strip()
        return name, description, text[match.end() :], frontmatter

    @staticmethod
    def _skill_metadata(frontmatter: dict[str, Any]) -> dict[str, Any]:
        raw = frontmatter.get("metadata", {})
        if isinstance(raw, str):
            try:
                raw = json.loads(raw)
            except json.JSONDecodeError:
                return {}
        if not isinstance(raw, dict):
            return {}
        for key in ("nanobot", "openclaw"):
            nested = raw.get(key)
            if isinstance(nested, dict):
                return nested
        return raw

    @staticmethod
    def _parse_requirements(raw: Any) -> tuple[tuple[str, ...], tuple[str, ...]]:
        if not isinstance(raw, dict):
            return (), ()

        def values(key: str) -> tuple[str, ...]:
            value = raw.get(key, [])
            if isinstance(value, str):
                value = [value]
            if not isinstance(value, list):
                return ()
            return tuple(item.strip() for item in value if isinstance(item, str) and item.strip())

        return values("bins"), values("env")

    @staticmethod
    def _as_bool(value: Any) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "on"}
        return False

    def catalog(self) -> list[SkillDefinition]:
        return list(self._skills.values())

    def get(self, name: str) -> SkillDefinition:
        try:
            skill = self._skills[name]
        except KeyError as exc:
            raise ValueError(f"Unknown skill: {name}") from exc
        available, reason = self.availability(skill)
        if not available:
            raise ValueError(f"Skill requirements unavailable: {reason}")
        return skill

    def availability(self, skill: SkillDefinition) -> tuple[bool, str]:
        missing_bins = [command for command in skill.requires_bins if not shutil.which(command)]
        missing_env = [name for name in skill.requires_env if not os.environ.get(name)]
        missing = [*(f"CLI: {command}" for command in missing_bins), *(f"ENV: {name}" for name in missing_env)]
        return not missing, ", ".join(missing)

    def always_skill_names(self) -> list[str]:
        return [
            skill.name
            for skill in self.catalog()
            if skill.always and self.availability(skill)[0]
        ]


class SkillSession:
    """Request-scoped active skills and their permitted reference files."""

    def __init__(self, registry: SkillRegistry):
        self.registry = registry
        self._active: dict[str, SkillDefinition] = {}
        self._always_on: set[str] = set()

    @property
    def active_names(self) -> list[str]:
        return list(self._active)

    def activate(self, name: str, *, always_on: bool = False) -> SkillDefinition:
        skill = self.registry.get(name)
        self._active[name] = skill
        if always_on:
            self._always_on.add(name)
        return skill

    def is_active(self, name: str) -> bool:
        return name in self._active

    def read_reference(self, skill_name: str, reference_name: str) -> str:
        skill = self._active.get(skill_name)
        if skill is None:
            raise ValueError(f"Skill is not active: {skill_name}")
        if reference_name not in skill.references:
            raise ValueError(f"Reference is not declared by active skill: {reference_name}")
        return (skill.root / "references" / reference_name).read_text(encoding="utf-8")

    def catalog_context(self) -> str:
        skills = self.registry.catalog()
        if not skills:
            return "没有发现可用 Skill。"
        entries = []
        for skill in skills:
            available, reason = self.registry.availability(skill)
            source = "工作区已批准覆盖" if skill.source == "workspace" else "受信任项目目录"
            availability = "可激活" if available else f"不可用（缺少 {reason}）"
            entries.append(f"- {skill.name}: {skill.description}（{source}；{availability}）")
        rendered_entries = "\n".join(entries)
        return (
            "可用按需 Skill（需要专门流程时先调用 activate_skill）：\n"
            f"{rendered_entries}"
        )

    def active_context(self) -> str:
        if not self._active:
            return "当前没有已激活的按需 Skill。"
        blocks = []
        for skill in self._active.values():
            mode = "常驻" if skill.name in self._always_on else "已激活"
            references = ", ".join(skill.references) or "无"
            blocks.append(
                f"[{mode} Skill: {skill.name}]\n{skill.body}\n"
                f"可按需读取的 reference: {references}"
            )
        return "\n\n".join(blocks)


class ActivateSkillInput(BaseModel):
    skill_name: str = Field(..., description="从可用 Skill 目录中选择的 Skill 名称")


class ReadSkillReferenceInput(BaseModel):
    skill_name: str = Field(..., description="已激活的 Skill 名称")
    reference_name: str = Field(..., description="Skill 声明的 reference 文件名")


def create_skill_session() -> SkillSession:
    approved_workspace_skills = _comma_separated_names(settings.agent_approved_workspace_skills)
    registry = SkillRegistry(
        settings.skills_dir,
        workspace_skills_dir=settings.agent_workspace_skills_dir,
        allow_workspace_overrides=settings.agent_allow_workspace_skill_overrides,
        approved_workspace_skills=approved_workspace_skills,
        disabled_skills=_comma_separated_names(settings.agent_disabled_skills),
    )
    session = SkillSession(registry)
    for name in registry.always_skill_names():
        session.activate(name, always_on=True)
    for name in settings.agent_always_on_skills.split(","):
        name = name.strip()
        if name and name not in registry.disabled_skills:
            session.activate(name, always_on=True)
    return session


def _comma_separated_names(value: str) -> set[str]:
    return {name.strip() for name in value.split(",") if name.strip()}


def auto_activate_for_request(session: SkillSession, req: AgentRequest) -> list[str]:
    if req.intent in _JOB_INTENTS:
        return _activate_if_available(session, JOB_APPLICATION_SKILL)

    has_job_context = bool(req.company_name or req.job_title or req.jd_text or req.resume_text)
    goal = req.goal.lower()
    if has_job_context or any(keyword in goal for keyword in _JOB_KEYWORDS):
        return _activate_if_available(session, JOB_APPLICATION_SKILL)
    return []


def _activate_if_available(session: SkillSession, name: str) -> list[str]:
    """Keep automatic routing safe when a deployment disables a domain Skill."""
    try:
        session.activate(name)
    except ValueError:
        return []
    return [name]


def build_skill_runtime_tools(session: SkillSession) -> list[StructuredTool]:
    def activate_skill(skill_name: str) -> str:
        try:
            skill = session.activate(skill_name)
            return json.dumps(
                {
                    "status": "activated",
                    "skill_name": skill.name,
                    "instructions": skill.body,
                    "references": list(skill.references),
                },
                ensure_ascii=False,
            )
        except ValueError as exc:
            message = str(exc)
            return json.dumps(
                {
                    "status": "skill_unavailable" if message.startswith("Skill requirements unavailable") else "unknown_skill",
                    "message": message,
                    "available_skills": [skill.name for skill in session.registry.catalog()],
                },
                ensure_ascii=False,
            )

    def read_skill_reference(skill_name: str, reference_name: str) -> str:
        try:
            return json.dumps(
                {
                    "status": "ok",
                    "skill_name": skill_name,
                    "reference_name": reference_name,
                    "content": session.read_reference(skill_name, reference_name),
                },
                ensure_ascii=False,
            )
        except ValueError as exc:
            return json.dumps({"status": "reference_unavailable", "message": str(exc)}, ensure_ascii=False)

    return [
        StructuredTool.from_function(
            func=activate_skill,
            name="activate_skill",
            description="激活可用 Skill，并加载其操作规程、可用 reference 与工具约束。",
            args_schema=ActivateSkillInput,
        ),
        StructuredTool.from_function(
            func=read_skill_reference,
            name="read_skill_reference",
            description="读取已激活 Skill 明确声明的参考资料。",
            args_schema=ReadSkillReferenceInput,
        ),
    ]
