import argparse
import json
import os
import sys
from pathlib import Path

from lmad.config import DEFAULT_MODEL, DEFAULT_MAX_ROUNDS, GROQ_API_KEY
from lmad.models.schemas import CaseInput
from lmad.orchestrator.debate import DebateOrchestrator
from lmad.utils.display import print_case_header, print_round, print_judgment


def load_case_file(file_path: str) -> CaseInput:
    path = Path(file_path)
    if not path.exists():
        print(f"[Error] Case file not found: {file_path}", file=sys.stderr)
        sys.exit(1)

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return CaseInput(**data)
    except Exception as e:
        print(f"[Error] Failed to parse case file '{file_path}': {e}", file=sys.stderr)
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Legal Multi-Agent Debate (LMAD) System powered by Groq",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--case",
        "-c",
        type=str,
        default="examples/criminal_fraud.json",
        help="Path to the JSON case file containing hypothesis, for_evidence, and against_evidence",
    )
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default=DEFAULT_MODEL,
        help="Groq model name to use for Advocate, Defender, and Judge",
    )
    parser.add_argument(
        "--rounds",
        "-r",
        type=int,
        default=None,
        help="Optional maximum round cap. By default, rounds adjust automatically according to the evidence count.",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="debate_result.json",
        help="Path to save the resulting debate transcript and verdict JSON",
    )

    args = parser.parse_args()

    # Verify Groq API Key
    api_key = os.getenv("GROQ_API_KEY", GROQ_API_KEY)
    if not api_key:
        print("\n[Error] GROQ_API_KEY is not set!", file=sys.stderr)
        print("Please set GROQ_API_KEY in your environment or create a .env file based on .env.example:", file=sys.stderr)
        print("  GROQ_API_KEY=gsk_your_actual_key_here\n", file=sys.stderr)
        sys.exit(1)

    case_input = load_case_file(args.case)
    print_case_header(case_input=case_input, model_name=args.model, max_rounds=args.rounds)

    orchestrator = DebateOrchestrator(
        case_input=case_input,
        max_rounds=args.rounds,
        model=args.model,
    )

    rounds, judgment, full_transcript = orchestrator.run()

    for r in rounds:
        print_round(r)

    print_judgment(judgment, historical_outcome=case_input.historical_outcome)

    # Save to output file
    output_data = {
        "case": case_input.model_dump(),
        "model": args.model,
        "rounds": [r.model_dump() for r in rounds],
        "judgment": judgment.model_dump(),
        "full_transcript": full_transcript,
    }

    try:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2)
        print(f"[Info] Complete debate record saved to: {args.output}")
    except Exception as e:
        print(f"[Warning] Could not save output to '{args.output}': {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
