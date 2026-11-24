import json
import logging
import time
import re
from typing import Optional
from groq import Groq
from lmad.config import DEFAULT_MODEL, GROQ_API_KEY, VERDICT_MODES
from lmad.models.schemas import CaseInput, JudgmentOutput, FidelityAssessment
from lmad.models.prompts import JUDGE_SYSTEM_PROMPT, get_judge_evaluation_prompt

logger = logging.getLogger(__name__)


class JudgeAgent:
    """Judge and Adjudicator agent evaluating the entire debate and returning calibrated probabilities."""

    def __init__(self, model: Optional[str] = None, api_key: Optional[str] = None):
        self.model = model or DEFAULT_MODEL
        self.client = Groq(api_key=api_key or GROQ_API_KEY)
        self.system_prompt = JUDGE_SYSTEM_PROMPT

    def _call_groq_json(self, messages: list, max_retries: int = 8) -> dict:
        """Call Groq API with JSON mode and auto-pause retry on rate limits (strictly adhering to assigned model)."""
        last_error = None
        for attempt in range(1, max_retries + 1):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    response_format={"type": "json_object"},
                    temperature=0.1,  # Lower temperature for judicial objectivity
                )
                content = response.choices[0].message.content
                return json.loads(content)
            except Exception as e:
                err_msg = str(e)
                last_error = e
                # Check for Groq rate limit (429 / TPM / RPM)
                if "429" in err_msg or "rate_limit" in err_msg.lower():
                    match = re.search(r"try again in ([0-9.]+)s", err_msg)
                    wait_time = float(match.group(1)) + 2.0 if match else 25.0
                    print(f"\n[Groq Rate Limit] Judge attempt {attempt}/{max_retries}: Rate limit reached. Pausing {wait_time:.1f}s before retrying on {self.model}...")
                    time.sleep(wait_time)
                elif attempt < max_retries and any(code in err_msg for code in ["500", "502", "503", "504", "timeout"]):
                    print(f"\n[Groq Server Error] Judge attempt {attempt}/{max_retries}: {e}. Pausing 5s before retrying...")
                    time.sleep(5)
                else:
                    logger.error("Judge call to model '%s' failed: %s", self.model, e)
                    if attempt == max_retries:
                        raise RuntimeError(f"Judge Groq API call failed on model '{self.model}' after {max_retries} attempts: {last_error}")
                    time.sleep(5)

        raise RuntimeError(f"All Judge attempts failed on model '{self.model}'. Last error: {last_error}")

    def evaluate_debate(self, case_input: CaseInput, full_transcript: str) -> JudgmentOutput:
        """Evaluate the full debate transcript and output calibrated verdict and probabilities."""
        verdict_labels = VERDICT_MODES.get(
            case_input.verdict_mode,
            VERDICT_MODES["guilty_not_guilty"],
        )
        pos_label, neg_label = verdict_labels

        prompt = get_judge_evaluation_prompt(
            hypothesis=case_input.hypothesis,
            for_clauses=case_input.for_evidence,
            against_clauses=case_input.against_evidence,
            full_transcript=full_transcript,
            standard_of_proof=case_input.standard_of_proof,
            verdict_mode=case_input.verdict_mode,
            positive_label=pos_label,
            negative_label=neg_label,
        )

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt},
        ]

        raw = self._call_groq_json(messages)

        # Extract and normalize probabilities
        p_pos = float(raw.get("probability_positive", 0.5))
        p_neg = float(raw.get("probability_negative", 0.5))

        # Ensure bounds and normalization
        total_p = p_pos + p_neg
        if total_p > 0:
            p_pos = round(p_pos / total_p, 4)
            p_neg = round(1.0 - p_pos, 4)
        else:
            p_pos, p_neg = 0.5, 0.5

        # Extract verdict
        verdict = raw.get("verdict", pos_label if p_pos >= 0.5 else neg_label)

        # Fidelity
        fidelity_raw = raw.get("fidelity_assessment", {})
        fidelity = FidelityAssessment(
            advocate_fidelity=fidelity_raw.get("advocate_fidelity", "No specific remarks."),
            defender_fidelity=fidelity_raw.get("defender_fidelity", "No specific remarks."),
        )

        decisive = raw.get("decisive_clauses", [])
        reasoning = raw.get("judicial_reasoning", "No detailed opinion provided.")

        return JudgmentOutput(
            verdict=verdict,
            verdict_type=case_input.verdict_mode,
            probability_positive=p_pos,
            probability_negative=p_neg,
            fidelity_assessment=fidelity,
            decisive_clauses=decisive,
            judicial_reasoning=reasoning,
        )
