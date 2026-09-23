"""Read-only artifact/configuration audit. Does not claim live deployment success."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from app.readiness import domestic_readiness


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = {"checked_at": datetime.now(timezone.utc).isoformat(), **domestic_readiness(),
              "gates_not_proven_by_this_readonly_audit": ["Fresh source crawl and posting-status verification", "Live LLM provider smoke test",
                  "Human review of retrieval labels and generated materials", "Load test and backup/restore rehearsal",
                  "Tenant identity/authorization before any public multi-user launch"]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
