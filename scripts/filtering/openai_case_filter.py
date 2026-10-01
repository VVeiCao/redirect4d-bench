#!/usr/bin/env python3
"""Filter candidate clips with the OpenAI API.

No API key is stored in this repository. Set OPENAI_API_KEY in your shell.
Input is JSONL. Each row is sent as text together with the prompt file.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True, help="Candidate JSONL.")
    parser.add_argument("--output", type=Path, required=True, help="Filtered JSONL with model decisions.")
    parser.add_argument("--prompt", type=Path, default=Path("configs/openai/filter_prompt.txt"))
    parser.add_argument("--model", default=os.environ.get("OPENAI_FILTER_MODEL", "gpt-4.1-mini"))
    parser.add_argument("--limit", type=int)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not os.environ.get("OPENAI_API_KEY") and not args.dry_run:
        raise RuntimeError("OPENAI_API_KEY is not set")

    prompt = args.prompt.read_text()
    rows = []
    for line in args.input.read_text().splitlines():
        if not line.strip():
            continue
        rows.append(json.loads(line))
        if args.limit and len(rows) >= args.limit:
            break

    if args.dry_run:
        print(f"[dry-run] would filter {len(rows)} rows with model={args.model}")
        return

    from openai import OpenAI

    client = OpenAI()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as f:
        for row in rows:
            response = client.responses.create(
                model=args.model,
                input=[
                    {
                        "role": "system",
                        "content": prompt,
                    },
                    {
                        "role": "user",
                        "content": json.dumps(row, ensure_ascii=False),
                    },
                ],
                text={"format": {"type": "json_object"}},
            )
            decision = json.loads(response.output_text)
            out = dict(row)
            out["openai_filter"] = decision
            f.write(json.dumps(out, ensure_ascii=False, sort_keys=True) + "\n")
            print(f"{row.get('id') or row.get('case') or row.get('video_id')}: {decision}")


if __name__ == "__main__":
    main()
