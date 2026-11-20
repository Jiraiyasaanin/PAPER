import json
import logging
from typing import List, Optional
from groq import Groq
from lmad.config import DEFAULT_MODEL, FALLBACK_MODELS, GROQ_API_KEY
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

    def _call_groq_json(self, messages: list) -> dict:
        """Call Groq API with JSON mode and automatic model fallback on error."""
        candidate_models = [self.model] + [m for m in FALLBACK_MODELS if m != self.model]
        last_error = None

        for model_name in candidate_models:
            try:
                response = self.client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    response_format={"type": "json_object"},
                    temperature=0.3,
                )
                content = response.choices[0].message.content
                return json.loads(content)
            except Exception as e:
                logger.warning("Call to model '%s' failed: %s. Trying next candidate...", model_name, e)
                last_error = e

        raise RuntimeError(f"All Groq model attempts failed for {self.role}. Last error: {last_error}")

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
