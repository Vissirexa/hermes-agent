"""Summarize a `lintlang scan --format json` report for a GitHub step summary.

The raw JSON is one object per scanned file (~780 files in this repo's
agent config/prompt surface as of Phase 1a) — too long to paste into a step
summary directly. This collapses it to:
  - verdict counts (PASS / REVIEW / FAIL)
  - finding counts by severity
  - the CRITICAL/HIGH findings, in full (these are the ones worth a human
    actually reading; everything else is in the uploaded artifact)

Advisory only: this script always exits 0. Whether lintlang's own verdict
should ever gate CI is a decision for a later phase (see the header comment
in .github/workflows/lintlang.yml).
"""

from __future__ import annotations

import collections
import json
import sys


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: lintlang_summary.py <report.json>", file=sys.stderr)
        return 2

    with open(sys.argv[1], encoding="utf-8") as fh:
        report = json.load(fh)

    verdicts: collections.Counter[str] = collections.Counter()
    severities: collections.Counter[str] = collections.Counter()
    notable: list[tuple[str, dict]] = []

    for entry in report:
        verdicts[entry.get("verdict", "UNKNOWN")] += 1
        for finding in entry.get("structural_findings", []):
            sev = finding.get("severity", "info")
            severities[sev] += 1
            if sev in ("critical", "high"):
                notable.append((entry["file"], finding))

    print("## lintlang — agent config/prompt scan")
    print()
    print(f"Files scanned: **{len(report)}**")
    print()
    print("| Verdict | Count |")
    print("|---|---|")
    for verdict in ("PASS", "REVIEW", "FAIL"):
        print(f"| {verdict} | {verdicts.get(verdict, 0)} |")
    print()
    print("| Severity | Count |")
    print("|---|---|")
    for sev in ("critical", "high", "medium", "low", "info"):
        print(f"| {sev} | {severities.get(sev, 0)} |")
    print()

    if notable:
        print(f"### CRITICAL/HIGH findings ({len(notable)})")
        print()
        for file_path, finding in notable:
            desc = finding.get("description", "")
            pattern = f"{finding.get('pattern_id', '?')} {finding.get('pattern_name', '')}"
            print(f"- **{file_path}** — `{pattern}` ({finding.get('severity')}): {desc}")
        print()
    else:
        print("No CRITICAL/HIGH findings.")
        print()

    print(
        "Full per-file report (including MEDIUM/LOW/INFO) is attached as "
        "the `lintlang-report` artifact on this run. This job is advisory "
        "only — see `.github/workflows/lintlang.yml` for why."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
