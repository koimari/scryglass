#!/usr/bin/env python3
"""Import Codex CLI session data related to this repo into docs/codex/.

Scans ~/.codex/sessions, ~/.codex/archived_sessions, and
~/.codex/memories/rollout_summaries for rollouts whose working directory is
this repo (or its Codex worktrees), distills each into a compact markdown
record (metadata, user prompts, final agent message, tool/file touches,
Codex's own rollout summary when available), and writes:

  docs/codex/README.md        index + usage
  docs/codex/INDEX.md         sortable session table with topic tags
  docs/codex/manifest.json    machine-readable session index
  docs/codex/sessions/<date>_<slug>.md   per-session records

Related-project sessions (league-combat-calculator, lol-strength-analysis,
parlay-risk-sim) are included in a separate section when --related is set.

Usage:
  python tools/codex_import.py [--repo /Users/river/scryglass] [--related]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

HOME = Path.home()
SESSIONS = HOME / ".codex" / "sessions"
ARCHIVED = HOME / ".codex" / "archived_sessions"
SUMMARIES = HOME / ".codex" / "memories" / "rollout_summaries"

USER_MSG_MARKERS = ('"type":"user_message"', '"role":"user"', '"role":"assistant"',
                    '"type":"function_call"', '"type":"task_complete"', '"cwd"',
                    '"session_meta"', '"type":"agent_message"')
TOOL_FILE_RE = re.compile(r'"(?:file_path|path|output_path|output_file|locator|filename)":\s*"([^"]+)"')

TOPIC_RULES = [
    ("draft", re.compile(r"draft|ban.?pick|pick.?ban", re.I)),
    ("ratings", re.compile(r"rating|elo", re.I)),
    ("tier-list", re.compile(r"tier ?list|tierlist", re.I)),
    ("champion-atoms", re.compile(r"atom|ontology|champion representation|mechanic", re.I)),
    ("calculator-bridge", re.compile(r"league-combat-calculator|combat calculator|lcc", re.I)),
    ("grubs", re.compile(r"grubs|void ?grub", re.I)),
    ("live", re.compile(r"live|totals|over/under|kills", re.I)),
    ("market", re.compile(r"bet|market|odds|bookmaker", re.I)),
    ("frontend", re.compile(r"frontend|ui|interface|atlas|page|component|vercel", re.I)),
    ("data-warehouse", re.compile(r"warehouse|elixir|parquet|etl|refresh|ingest", re.I)),
    ("leaguepedia", re.compile(r"leaguepedia|wiki|oracle|mechanics", re.I)),
    ("evaluation", re.compile(r"evaluation|benchmark|calibration|holdout|remand|audit|review", re.I)),
    ("deploy", re.compile(r"deploy|publish|pack|cdn|vercel", re.I)),
    ("replay", re.compile(r"rofl|replay", re.I)),
]


def topic_tags(text: str) -> list[str]:
    tags = []
    for tag, rx in TOPIC_RULES:
        if rx.search(text):
            tags.append(tag)
    return tags


def slugify(text: str, maxlen: int = 56) -> str:
    text = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return text[:maxlen].rstrip("-") or "session"


def iter_rollouts() -> list[Path]:
    paths = []
    for base in (SESSIONS, ARCHIVED):
        if base.exists():
            paths.extend(sorted(base.rglob("*.jsonl")))
    return paths


def parse_line(line: str) -> dict | None:
    try:
        return json.loads(line)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None


class RolloutExtractor:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.session_id: str | None = None
        self.started_at: str | None = None
        self.model_provider: str | None = None
        self.cli_version: str | None = None
        self.cwd: str | None = None
        self.user_messages: list[str] = []
        self.assistant_parts: list[str] = []
        self.tool_calls: Counter = Counter()
        self.file_touches: list[str] = []
        self.error: str | None = None
        self.completed: bool = False
        self.summary: str | None = None

    def run(self) -> None:
        seen_user: set[str] = set()
        with self.path.open("r", errors="replace") as fh:
            for line in fh:
                if not any(m in line for m in USER_MSG_MARKERS):
                    continue
                obj = parse_line(line)
                if obj is None:
                    continue
                t = obj.get("type")
                p = obj.get("payload") or {}
                if t == "session_meta":
                    self.session_id = p.get("session_id") or p.get("id")
                    self.started_at = p.get("timestamp")
                    self.model_provider = p.get("model_provider")
                    self.cli_version = p.get("cli_version")
                elif t == "turn_context":
                    if self.cwd is None:
                        self.cwd = p.get("cwd")
                elif t == "event_msg":
                    pt = p.get("type")
                    if pt == "user_message":
                        msg = str(p.get("message") or "").strip()
                        if msg and msg not in seen_user:
                            seen_user.add(msg)
                            self.user_messages.append(msg)
                    elif pt == "task_complete":
                        self.completed = True
                        err = p.get("error")
                        if err:
                            self.error = str(err.get("message") or err)
                    elif pt == "agent_message":
                        msg = str(p.get("message") or "").strip()
                        if msg:
                            self.assistant_parts.append(msg)
                elif t == "response_item":
                    pt = p.get("type")
                    role = p.get("role")
                    if pt == "message" and role in ("user", "assistant"):
                        text = ""
                        for c in p.get("content") or []:
                            if c.get("type") in ("input_text", "output_text", "refusal"):
                                text += str(c.get("text") or c.get("refusal") or "")
                        text = text.strip()
                        if not text:
                            continue
                        if role == "user":
                            # skip harness boilerplate (AGENTS.md instructions, tool results)
                            if text.startswith("# AGENTS.md") or text.startswith("<INSTRUCTIONS>"):
                                continue
                            if text not in seen_user:
                                seen_user.add(text)
                                self.user_messages.append(text)
                        else:
                            self.assistant_parts.append(text)
                    elif pt == "function_call":
                        name = p.get("name")
                        if name:
                            self.tool_calls[name] += 1
                        args = p.get("arguments")
                        if isinstance(args, str):
                            for m in TOOL_FILE_RE.finditer(args):
                                fp = m.group(1)
                                if fp.startswith(("/Users/river/scryglass", "/Users/river/.codex/worktrees")):
                                    self.file_touches.append(fp)

    def summary_record(self) -> dict:
        user_text = "\n\n".join(self.user_messages)
        assistant_text = "\n\n".join(self.assistant_parts)
        first_user = self.user_messages[0] if self.user_messages else ""
        last_assistant = self.assistant_parts[-1] if self.assistant_parts else ""
        return {
            "path": str(self.path),
            "session_id": self.session_id,
            "started_at": self.started_at,
            "cwd": self.cwd,
            "model_provider": self.model_provider,
            "cli_version": self.cli_version,
            "completed": self.completed,
            "error": self.error,
            "user_message_count": len(self.user_messages),
            "assistant_char_count": len(assistant_text),
            "first_user_message": first_user[:600],
            "last_assistant_message": last_assistant[:1200],
            "tools": dict(self.tool_calls.most_common()),
            "file_touches": sorted(set(self.file_touches))[:60],
            "tags": topic_tags(first_user + "\n" + last_assistant),
        }


def find_summary(rollout_path: Path) -> str | None:
    if not SUMMARIES.exists():
        return None
    for f in SUMMARIES.iterdir():
        try:
            txt = f.read_text(errors="replace")
        except OSError:
            continue
        if f"rollout_path: {rollout_path}" in txt:
            return txt
    return None


def render_markdown(rec: dict, summary: str | None) -> str:
    lines = []
    lines.append(f"# {rec['first_user_message'][:100] or 'Codex session'}")
    lines.append("")
    lines.append("| | |")
    lines.append("|---|---|")
    lines.append(f"| Session | `{rec['session_id'] or '?'}` |")
    lines.append(f"| Started | {rec['started_at'] or '?'} |")
    lines.append(f"| CWD | `{rec['cwd'] or '?'}` |")
    lines.append(f"| Model provider | {rec['model_provider'] or '?'} |")
    lines.append(f"| CLI | {rec['cli_version'] or '?'} |")
    lines.append(f"| Completed | {rec['completed']} |")
    if rec["error"]:
        lines.append(f"| Error | {rec['error'][:200]} |")
    lines.append(f"| Rollout | `{rec['path']}` |")
    lines.append("")
    lines.append(f"Tags: {', '.join(rec['tags']) or '—'}")
    lines.append("")
    if summary:
        lines.append("## Codex rollout summary")
        lines.append("")
        lines.append("```text")
        lines.append(summary.strip()[:12000])
        lines.append("```")
        lines.append("")
    lines.append("## User prompts")
    lines.append("")
    if rec["user_message_count"]:
        # first user message in full, the rest truncated
        lines.append("### 1")
        lines.append("")
        lines.append(rec["first_user_message"])
        lines.append("")
        if rec["user_message_count"] > 1:
            lines.append(f"*(+{rec['user_message_count'] - 1} further user messages; see rollout JSONL)*")
            lines.append("")
    else:
        lines.append("_(none extracted)_")
        lines.append("")
    lines.append("## Final agent message")
    lines.append("")
    lines.append(rec["last_assistant_message"] or "_(none extracted)_")
    lines.append("")
    lines.append("## Tools and files")
    lines.append("")
    if rec["tools"]:
        lines.append("| Tool | Calls |")
        lines.append("|---|---|")
        for name, count in sorted(rec["tools"].items()):
            lines.append(f"| {name} | {count} |")
        lines.append("")
    if rec["file_touches"]:
        lines.append("Files touched in-repo (sampled):")
        lines.append("")
        for fp in rec["file_touches"]:
            lines.append(f"- `{fp}`")
        lines.append("")
    return "\n".join(lines)


SKILLS_TO_IMPORT = [
    "league-wiki-query",      # Scryglass wiki vault SQLite queries (repo-aware defaults)
    "query-grid-research",    # private Scryglass GRID integration research + capability catalog
    "who-wins-this-game",     # Scryglass Draft Score / lineup strength prediction script
    "frontend-skill",         # visually strong web UI work (Scryglass surfaces)
    "frontend-design",        # distinctive visual design guidance
    "write-website-copy",     # Koi house voice for public copy
    "pdf",                    # PDF generation/inspection (articles, reports)
    "playwright",             # browser automation/QA for the public app
]
SKILL_EXCLUDE_DIRS = {"cdn", ".playwright", ".git", "__pycache__", "node_modules"}


def import_skills(repo: Path) -> list[dict]:
    """Copy curated user skills from ~/.codex/skills into <repo>/.codex/skills.

    Returns a manifest list describing what was copied (name, source, file
    count, total bytes, excluded dirs).
    """
    src_root = HOME / ".codex" / "skills"
    dst_root = repo / ".codex" / "skills"
    dst_root.mkdir(parents=True, exist_ok=True)
    manifest = []
    for name in SKILLS_TO_IMPORT:
        src = src_root / name
        if not src.is_dir():
            print(f"WARN skill not found: {name}", file=sys.stderr)
            continue
        dst = dst_root / name
        if dst.exists():
            import shutil
            shutil.rmtree(dst)
        n_files = 0
        n_bytes = 0
        excluded = []
        for dp, dirs, files in os.walk(src):
            keep_dirs = [d for d in dirs if d not in SKILL_EXCLUDE_DIRS]
            for d in dirs:
                if d in SKILL_EXCLUDE_DIRS:
                    excluded.append(os.path.relpath(os.path.join(dp, d), src))
            dirs[:] = keep_dirs
            rel = os.path.relpath(dp, src)
            target = dst if rel == "." else dst / rel
            target.mkdir(parents=True, exist_ok=True)
            for f in files:
                with open(os.path.join(dp, f), "rb") as source_file:
                    data = source_file.read()
                (target / f).write_bytes(data)
                n_files += 1
                n_bytes += len(data)
        manifest.append({
            "name": name,
            "source": str(src),
            "file_count": n_files,
            "bytes": n_bytes,
            "excluded_dirs": sorted(set(excluded)),
        })
        print(f"skill {name}: {n_files} files, {n_bytes/1024:.0f}KB")
    return manifest


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", default=str(Path(__file__).resolve().parents[1]))
    ap.add_argument("--related", action="store_true", help="include related-project sessions")
    ap.add_argument("--out", default=None, help="output dir (default: <repo>/docs/codex)")
    ap.add_argument("--skills", action="store_true", help="also copy curated Codex skills into <repo>/.codex/skills")
    args = ap.parse_args()

    repo = Path(args.repo).resolve()
    out = Path(args.out) if args.out else repo / "docs" / "codex"
    out.mkdir(parents=True, exist_ok=True)
    (out / "sessions").mkdir(parents=True, exist_ok=True)

    related_prefixes = (
        "/Users/river/Projects/league-combat-calculator",
        "/Users/river/Projects/lol-strength-analysis",
        "/Users/river/parlay-risk-sim",
    )

    records = []
    skipped = 0
    for rp in iter_rollouts():
        rel = False
        # cheap prefilter: peek at first lines for session_meta/turn_context
        with rp.open("r", errors="replace") as fh:
            head = [fh.readline() for _ in range(3)]
        head_text = "".join(head)
        if args.repo in head_text or "/.codex/worktrees/" in head_text:
            rel = True
        if args.related and any(p in head_text for p in related_prefixes):
            rel = True
        if not rel:
            skipped += 1
            continue

        ex = RolloutExtractor(rp)
        try:
            ex.run()
        except Exception as exc:  # noqa: BLE001 - one bad rollout must not kill the import
            print(f"WARN extract failed {rp}: {exc}", file=sys.stderr)
            continue
        rec = ex.summary_record()
        rec["summary"] = find_summary(rp)
        records.append(rec)

    records.sort(key=lambda r: r["started_at"] or "")
    print(f"imported {len(records)} rollouts ({skipped} skipped)")

    # per-session files
    for rec in records:
        date = (rec["started_at"] or "unknown")[:10]
        base = rec["first_user_message"] or rec["last_assistant_message"] or (rec["session_id"] or "session")
        slug = slugify(base)
        suffix = (rec["session_id"] or "none")[:8]
        fname = f"{date}_{slug}_{suffix}.md"
        path = out / "sessions" / fname
        path.write_text(render_markdown(rec, rec["summary"]))

    # manifest
    manifest = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "repo": str(repo),
        "count": len(records),
        "sessions": [
            {k: v for k, v in rec.items() if k != "summary"}
            for rec in records
        ],
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))

    # INDEX.md
    lines = ["# Codex session index", ""]
    lines.append(f"Generated {manifest['generated_at']} — {len(records)} sessions.")
    lines.append("")
    lines.append("| Date | Tags | First user message | Session |")
    lines.append("|---|---|---|---|")
    for rec in records:
        date = (rec["started_at"] or "?")[:10]
        tags = ",".join(rec["tags"]) or "-"
        msg = rec["first_user_message"].replace("|", "/")[:80]
        sid = rec["session_id"] or "?"
        lines.append(f"| {date} | {tags} | {msg} | `{sid}` |")
    (out / "INDEX.md").write_text("\n".join(lines) + "\n")

    # README.md
    readme = [
        "# Codex import", "",
        "Distilled records of Codex CLI sessions whose working directory was this repo",
        "(or its Codex worktrees), imported by `tools/codex_import.py`.",
        "",
        "- `INDEX.md` — sortable table of every imported session with topic tags.",
        "- `manifest.json` — machine-readable index (metadata, user prompts, final",
        "  agent messages, tools, file touches).",
        "- `sessions/` — one markdown record per session, including Codex's own",
        "  rollout summary when one exists.",
        "",
        "Raw JSONL rollouts remain at `~/.codex/sessions/**` and are not copied into",
        "this repo (privacy + size). Re-run the importer after Codex work to refresh:",
        "",
        "```bash",
        "python tools/codex_import.py --repo /Users/river/scryglass",
        "```",
        "",
    ]
    (out / "README.md").write_text("\n".join(readme) + "\n")

    if args.skills:
        skills_manifest = import_skills(repo)
        skills_doc = [
            "# Codex skills import", "",
            f"Generated {manifest['generated_at']} — copied from `~/.codex/skills` into",
            "`.codex/skills/` in this repo.", "",
            "| Skill | Files | Size | Excluded |",
            "|---|---|---|---|",
        ]
        for s in skills_manifest:
            skills_doc.append(
                f"| {s['name']} | {s['file_count']} | {s['bytes']/1024:.0f}KB | "
                f"{', '.join(s['excluded_dirs']) or '—'} |"
            )
        skills_doc += [
            "",
            "Heavy/irrelevant skills (`visor-mcp`, `sora`, `hatch-pet`, `tldraw-offline`,",
            "`coast-cli-skill`, `.system/*`, `vendor_imports/*`) were intentionally not",
            "copied. Playwright browser binaries under `cdn/` and its local cache were",
            "excluded; the skill re-downloads them when needed.",
            "",
        ]
        (out / "skills-import.md").write_text("\n".join(skills_doc) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
