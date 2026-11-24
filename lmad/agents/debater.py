import json
import logging
import time
import re
from typing import List, Optional
from groq import Groq
from lmad.config import DEFAULT_MODEL, GROQ_API_KEY
from lmad.models.schemas import Clause, StatementOutput, RebuttalOutput
from lmad.models.prompts import (
    ADVOCATE_SYSTEM_PROMPT,
    DEFENDER_SYSTEM_PROMPT,
    get_advocate_statement_prompt,
    get_defender_rebuttal_prompt,
    get_defender_statement_prompt,
    get_advocate_rebuttal_prompt,
)

logger = logging.getLogger(__name__)


class DebaterAgent:
    """Agent representing either the Advocate or Defender in the multi-agent legal debate."""

    def __init__(self, role: str, model: Optional[str] = None, api_key: Optional[str] = None):
        """
        role: 'Advocate' or 'Defender'
        model: Groq model name (defaults to config.DEFAULT_MODEL)
        """
        if role not in ("Advocate", "Defender"):
            raise ValueError(f"Invalid role '{role}'. Must be 'Advocate' or 'Defender'.")
        self.role = role
        self.model = model or DEFAULT_MODEL
        self.client = Groq(api_key=api_key or GROQ_API_KEY)
        self.system_prompt = (
            ADVOCATE_SYSTEM_PROMPT if self.role == "Advocate" else DEFENDER_SYSTEM_PROMPT
        )

    def _call_groq_json(self, messages: list, max_retries: int = 8) -> dict:
        """Call Groq API with JSON mode and auto-pause retry on rate limits (strictly adhering to assigned model)."""
        last_error = None
        for attempt in range(1, max_retries + 1):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    response_format={"type": "json_object"},
                    temperature=0.3,
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
                    print(f"\n[Groq Rate Limit] Attempt {attempt}/{max_retries} for {self.role}: Rate limit reached. Pausing {wait_time:.1f}s before retrying on {self.model}...")
                    time.sleep(wait_time)
                elif attempt < max_retries and any(code in err_msg for code in ["500", "502", "503", "504", "timeout"]):
                    print(f"\n[Groq Server Error] Attempt {attempt}/{max_retries} for {self.role}: {e}. Pausing 5s before retrying...")
                    time.sleep(5)
                else:
                    logger.error("Call to model '%s' failed for %s: %s", self.model, self.role, e)
                    if attempt == max_retries:
                        raise RuntimeError(f"Groq API call failed for {self.role} on model '{self.model}' after {max_retries} attempts: {last_error}")
                    time.sleep(5)

        raise RuntimeError(f"All attempts failed for {self.role} on model '{self.model}'. Last error: {last_error}")

    def generate_statement(
        self,
        hypothesis: str,
        unused_clauses: List[Clause],
        transcript_history: str,
        round_number: int,
    ) -> StatementOutput:
        """Generate a leading statement by selecting and grouping similar clauses."""
        if self.role == "Advocate":
            prompt = get_advocate_statement_prompt(
                hypothesis=hypothesis,
                unused_for_clauses=unused_clauses,
                transcript_history=transcript_history,
                round_number=round_number,
            )
        else:
            prompt = get_defender_statement_prompt(
                hypothesis=hypothesis,
                unused_against_clauses=unused_clauses,
                transcript_history=transcript_history,
                round_number=round_number,
            )

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt},
        ]

        raw_json = self._call_groq_json(messages)

        # Fallback validation to ensure fields exist
        grouped_ids = raw_json.get("grouped_clause_ids", [])
        if not grouped_ids and unused_clauses:
            grouped_ids = [unused_clauses[0].id]

        quotes = raw_json.get("exact_document_statements", [])
        if not quotes and unused_clauses:
            quotes = [unused_clauses[0].text]

        argument = raw_json.get("argument", "No explicit argument generated.")

        return StatementOutput(
            grouped_clause_ids=grouped_ids,
            exact_document_statements=quotes,
            argument=argument,
        )

    def generate_rebuttal(
        self,
        hypothesis: str,
        opponent_statement: StatementOutput,
        transcript_history: str,
        round_number: int,
    ) -> RebuttalOutput:
        """Generate a cross-examination rebuttal to the opponent's statement."""
        if self.role == "Defender":
            prompt = get_defender_rebuttal_prompt(
                hypothesis=hypothesis,
                advocate_statement=opponent_statement.model_dump(),
                transcript_history=transcript_history,
                round_number=round_number,
            )
        else:
            prompt = get_advocate_rebuttal_prompt(
                hypothesis=hypothesis,
                defender_statement=opponent_statement.model_dump(),
                transcript_history=transcript_history,
                round_number=round_number,
            )

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt},
        ]

        raw_json = self._call_groq_json(messages)

        fidelity = raw_json.get("textual_fidelity_check", "No fidelity remarks provided.")
        rebuttal = raw_json.get("substantive_rebuttal", "No substantive rebuttal provided.")

        return RebuttalOutput(
            textual_fidelity_check=fidelity,
            substantive_rebuttal=rebuttal,
        )
