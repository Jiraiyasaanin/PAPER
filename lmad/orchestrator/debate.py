import logging
from typing import Callable, List, Optional, Tuple
from lmad.models.schemas import (
    CaseInput,
    RoundResult,
    PhaseExchange,
    JudgmentOutput,
    StatementOutput,
)
from lmad.orchestrator.clause_tracker import ClauseTracker
from lmad.agents.debater import DebaterAgent
from lmad.agents.judge import JudgeAgent
from lmad.config import DEFAULT_MAX_ROUNDS

logger = logging.getLogger(__name__)


class DebateOrchestrator:
    """Coordinates the 2-phase multi-round adversarial debate loop and delivers the case to the Judge."""

    def __init__(
        self,
        case_input: CaseInput,
        max_rounds: Optional[int] = None,
        model: Optional[str] = None,
        on_turn_complete: Optional[Callable[[str, str, dict], None]] = None,
    ):
        """
        case_input: Case data including hypothesis, for_evidence, and against_evidence
        max_rounds: Optional round cap. If None, dynamically determined by evidence pool size.
        model: Groq model name for all agents (Advocate, Defender, Judge)
        on_turn_complete: Optional hook (phase_name, agent_name, payload) for live logging
        """
        self.case_input = case_input
        # Automatically calculate rounds based on evidence count if not specified
        max_evidence = max(len(case_input.for_evidence), len(case_input.against_evidence))
        self.max_rounds = max_rounds if max_rounds is not None else max(1, max_evidence)
        self.tracker = ClauseTracker(
            for_clauses=case_input.for_evidence,
            against_clauses=case_input.against_evidence,
        )
        self.advocate = DebaterAgent(role="Advocate", model=model)
        self.defender = DebaterAgent(role="Defender", model=model)
        self.judge = JudgeAgent(model=model)
        self.on_turn_complete = on_turn_complete

        self.rounds: List[RoundResult] = []
        self.transcript_lines: List[str] = []

    def _append_transcript(self, text: str) -> None:
        self.transcript_lines.append(text)

    def get_full_transcript(self) -> str:
        return "\n\n".join(self.transcript_lines)

    def run(self) -> Tuple[List[RoundResult], JudgmentOutput, str]:
        """Executes the complete debate and returns (rounds, judgment, full_transcript)."""
        logger.info("Starting debate on hypothesis: '%s'", self.case_input.hypothesis)

        for round_num in range(1, self.max_rounds + 1):
            # Check termination before starting the round
            if self.tracker.all_evidence_presented():
                logger.info(
                    "Termination condition satisfied: Both For and Against agents have presented all their evidence. Ending debate."
                )
                self._append_transcript(
                    "\n[DEBATE TERMINATION: Both Advocate and Defender have presented all evidence clauses from their pools.]"
                )
                break

            unused_for = self.tracker.get_unused_for()
            unused_against = self.tracker.get_unused_against()

            self._append_transcript(f"==================================================")
            self._append_transcript(f"ROUND {round_num}")
            self._append_transcript(f"==================================================")

            # ------------------------------------------------------------------
            # PHASE A: Advocate Leads Statement -> Defender Rebuts
            # ------------------------------------------------------------------
            exchange_a: Optional[PhaseExchange] = None

            if unused_for:
                logger.info("Executing Round %d - Phase A (Advocate Leading)...", round_num)
                stmt_a = self.advocate.generate_statement(
                    hypothesis=self.case_input.hypothesis,
                    unused_clauses=unused_for,
                    transcript_history=self.get_full_transcript(),
                    round_number=round_num,
                )
                self.tracker.mark_for_used(stmt_a.grouped_clause_ids)

                phase_a_stmt_text = (
                    f"[ROUND {round_num} - PHASE A: ADVOCATE STATEMENT]\n"
                    f"Grouped Clauses: {stmt_a.grouped_clause_ids}\n"
                    f"Exact Document Statements:\n"
                    + "\n".join(f"  - \"{q}\"" for q in stmt_a.exact_document_statements)
                    + f"\nArgument: {stmt_a.argument}"
                )
                self._append_transcript(phase_a_stmt_text)

                if self.on_turn_complete:
                    self.on_turn_complete("Phase A Statement", "Advocate", stmt_a.model_dump())

                # Defender creates rebuttal
                reb_a = self.defender.generate_rebuttal(
                    hypothesis=self.case_input.hypothesis,
                    opponent_statement=stmt_a,
                    transcript_history=self.get_full_transcript(),
                    round_number=round_num,
                )

                phase_a_reb_text = (
                    f"[ROUND {round_num} - PHASE A: DEFENDER REBUTTAL]\n"
                    f"Fidelity Check: {reb_a.textual_fidelity_check}\n"
                    f"Substantive Rebuttal: {reb_a.substantive_rebuttal}"
                )
                self._append_transcript(phase_a_reb_text)

                if self.on_turn_complete:
                    self.on_turn_complete("Phase A Rebuttal", "Defender", reb_a.model_dump())

                exchange_a = PhaseExchange(
                    phase="Phase A",
                    leading_agent="Advocate",
                    statement=stmt_a,
                    rebuttal=reb_a,
                )
            else:
                logger.info("Round %d - Phase A skipped: Advocate has exhausted evidence pool.", round_num)
                self._append_transcript(
                    f"[ROUND {round_num} - PHASE A SKIPPED: Advocate has presented all evidence clauses. Skipping to Phase B.]"
                )

            # ------------------------------------------------------------------
            # PHASE B: Defender Leads Statement -> Advocate Rebuts
            # ------------------------------------------------------------------
            unused_against = self.tracker.get_unused_against()
            exchange_b: Optional[PhaseExchange] = None

            if unused_against:
                logger.info("Executing Round %d - Phase B (Defender Leading)...", round_num)
                stmt_b = self.defender.generate_statement(
                    hypothesis=self.case_input.hypothesis,
                    unused_clauses=unused_against,
                    transcript_history=self.get_full_transcript(),
                    round_number=round_num,
                )
                self.tracker.mark_against_used(stmt_b.grouped_clause_ids)

                phase_b_stmt_text = (
                    f"[ROUND {round_num} - PHASE B: DEFENDER STATEMENT]\n"
                    f"Grouped Clauses: {stmt_b.grouped_clause_ids}\n"
                    f"Exact Document Statements:\n"
                    + "\n".join(f"  - \"{q}\"" for q in stmt_b.exact_document_statements)
                    + f"\nDefense Argument: {stmt_b.argument}"
                )
                self._append_transcript(phase_b_stmt_text)

                if self.on_turn_complete:
                    self.on_turn_complete("Phase B Statement", "Defender", stmt_b.model_dump())

                # Advocate creates rebuttal
                reb_b = self.advocate.generate_rebuttal(
                    hypothesis=self.case_input.hypothesis,
                    opponent_statement=stmt_b,
                    transcript_history=self.get_full_transcript(),
                    round_number=round_num,
                )

                phase_b_reb_text = (
                    f"[ROUND {round_num} - PHASE B: ADVOCATE REBUTTAL]\n"
                    f"Fidelity Check: {reb_b.textual_fidelity_check}\n"
                    f"Substantive Rebuttal: {reb_b.substantive_rebuttal}"
                )
                self._append_transcript(phase_b_reb_text)

                if self.on_turn_complete:
                    self.on_turn_complete("Phase B Rebuttal", "Advocate", reb_b.model_dump())

                exchange_b = PhaseExchange(
                    phase="Phase B",
                    leading_agent="Defender",
                    statement=stmt_b,
                    rebuttal=reb_b,
                )
            else:
                logger.info("Round %d - Phase B skipped: Defender has exhausted evidence pool.", round_num)
                self._append_transcript(
                    f"[ROUND {round_num} - PHASE B SKIPPED: Defender has presented all evidence clauses.]"
                )

            if exchange_a is not None or exchange_b is not None:
                self.rounds.append(
                    RoundResult(
                        round_number=round_num,
                        phase_a=exchange_a,
                        phase_b=exchange_b,
                    )
                )

            # ------------------------------------------------------------------
            # TERMINATION CONDITION: All evidence presented by both agents
            # ------------------------------------------------------------------
            if self.tracker.all_evidence_presented():
                logger.info(
                    "Termination condition satisfied: Both For and Against agents have presented all their evidence. Ending debate."
                )
                self._append_transcript(
                    "\n[DEBATE TERMINATION: Both Advocate and Defender have presented all evidence clauses from their pools.]"
                )
                break

        # ------------------------------------------------------------------
        # FINAL ADJUDICATION
        # ------------------------------------------------------------------
        logger.info("Delivering full debate transcript to Judge for adjudication...")
        judgment = self.judge.evaluate_debate(
            case_input=self.case_input,
            full_transcript=self.get_full_transcript(),
        )

        return self.rounds, judgment, self.get_full_transcript()
