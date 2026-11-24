"""
ContractNLI -> LMAD Debate Bridge
Links Tier 1 (NyayaSetu DCP-LRN Neural Clause Retriever & Stance Classifier)
to Tier 2 (Legal Multi-Agent Debate Layer powered by Groq).
Does NOT modify any existing LMAD codebase.
"""
import os
import sys
import json
import argparse
from pathlib import Path
import torch
import numpy as np

# Ensure root of PAPER is in sys.path
PAPER_ROOT = Path(__file__).resolve().parent
if str(PAPER_ROOT) not in sys.path:
    sys.path.insert(0, str(PAPER_ROOT))

from lmad.models.schemas import CaseInput, Clause
from lmad.orchestrator.debate import DebateOrchestrator
from lmad.utils.display import print_case_header, print_round, print_judgment
from lmad.config import DEFAULT_MODEL

# Load Tier-1 inference logic
from infer_contract import NyayaSetuInference

DEV_PATH = Path("/Users/sarah/.gemini/antigravity/scratch/data/dev.json")
if not DEV_PATH.exists():
    DEV_PATH = Path("/Users/sarah/Downloads/contract-nli/dev.json")
CHECKPOINT_PATH = PAPER_ROOT / "best_dcp_lrn_full.pt"

def extract_case_from_contractnli(doc_id=3, hyp_id="nda-11", top_k=5):
    """
    Extracts a real contract from ContractNLI dev.json, runs Tier 1 NyayaSetu
    to retrieve and partition clauses, and formats it as a CaseInput.
    """
    print(f"--> Loading ContractNLI document #{doc_id} and hypothesis '{hyp_id}'...")
    with open(DEV_PATH, "r", encoding="utf-8") as f:
        dev_data = json.load(f)
        
    doc = next((d for d in dev_data["documents"] if d["id"] == doc_id), None)
    if not doc:
        raise ValueError(f"Document ID {doc_id} not found in {DEV_PATH}")
        
    labels = dev_data["labels"]
    if hyp_id not in labels:
        raise ValueError(f"Hypothesis '{hyp_id}' not found in labels.")
        
    hypothesis_text = labels[hyp_id]["hypothesis"]
    
    # Extract contract clauses
    text = doc["text"]
    clauses = [text[s[0]:s[1]].strip() for s in doc["spans"]]
    print(f"--> Contract has {len(clauses)} clauses. Running Tier-1 DCP-LRN...")
    
    # Run Tier 1 model
    engine = NyayaSetuInference(checkpoint_path=CHECKPOINT_PATH)
    results = engine.analyze(clauses, hypothesis_text, top_k=top_k)
    
    for_evidence = []
    against_evidence = []
    
    for r in results:
        cid = f"clause_{r['clause_index']}"
        clause_obj = Clause(
            id=cid,
            title=f"Section {r['clause_index']} (Relevance: {r['relevance_score']:.1%})",
            text=r['clause_text']
        )
        if "ENTAILMENT" in r['predicted_stance']:
            for_evidence.append(clause_obj)
        else:
            against_evidence.append(clause_obj)
            
    # Ground truth comparison
    ground_truth = "Unknown"
    ann_set = doc["annotation_sets"][0]["annotations"].get(hyp_id)
    if ann_set:
        ground_truth = ann_set.get("choice", "NotMentioned")
        print(f"--> Ground Truth Annotation from ContractNLI: '{ground_truth}' (Spans: {ann_set.get('spans', [])})")

    # If one pool is empty, ensure at least one contextual clause is present for debate balance
    if not against_evidence:
        against_evidence.append(Clause(
            id="defense_general_scope",
            title="General Limitation & Scope Defense",
            text="The contract provisions do not explicitly extend the prohibition beyond what is strictly and unambiguously stated in the text."
        ))
    if not for_evidence:
        for_evidence.append(Clause(
            id="affirmative_general_intent",
            title="General Intent & Protection Objective",
            text="The overall intent and purpose of this agreement is to comprehensively protect proprietary disclosures and enforce strict non-disclosure."
        ))
        
    case_input = CaseInput(
        case_name=f"ContractNLI Doc #{doc_id} ({hyp_id})",
        hypothesis=hypothesis_text,
        historical_outcome=ground_truth,
        standard_of_proof="Preponderance of the evidence",
        verdict_mode="entails_does_not_entail",
        for_evidence=for_evidence,
        against_evidence=against_evidence
    )
    
    return case_input

def main():
    parser = argparse.ArgumentParser(description="Run LMAD Debate on real ContractNLI cases using Tier 1")
    parser.add_argument("--doc", "-d", type=int, default=3, help="ContractNLI Document ID from dev.json")
    parser.add_argument("--hyp", "-y", type=str, default="nda-11", help="ContractNLI Hypothesis ID (e.g. nda-11, nda-16, nda-5)")
    parser.add_argument("--model", "-m", type=str, default=DEFAULT_MODEL, help="Groq model name")
    parser.add_argument("--output", "-o", type=str, default="contractnli_debate_result.json", help="Output file path")
    args = parser.parse_args()
    
    # 1. Build CaseInput via Tier 1
    case_input = extract_case_from_contractnli(doc_id=args.doc, hyp_id=args.hyp, top_k=5)
    
    # 2. Display Header
    print_case_header(case_input=case_input, model_name=args.model, max_rounds=None)
    
    # 3. Launch Debate Orchestrator
    def on_turn(phase, agent, data):
        print(f"  ✓ Turn Finished: {phase} by {agent}")

    orchestrator = DebateOrchestrator(
        case_input=case_input,
        max_rounds=None,
        model=args.model,
        on_turn_complete=on_turn
    )
    
    rounds, judgment, full_transcript = orchestrator.run()
    
    # 4. Display Results
    for r in rounds:
        print_round(r)
        
    print_judgment(judgment, historical_outcome=case_input.historical_outcome)
    
    # 5. Save Output
    output_data = {
        "case": case_input.model_dump(),
        "model": args.model,
        "rounds": [r.model_dump() for r in rounds],
        "judgment": judgment.model_dump(),
        "full_transcript": full_transcript
    }
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)
    print(f"\n[Info] Full debate transcript & judgment saved to: {args.output}")

if __name__ == "__main__":
    main()
