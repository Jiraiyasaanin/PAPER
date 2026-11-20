from typing import List, Optional
from pydantic import BaseModel, Field


class Clause(BaseModel):
    id: str = Field(description="Unique identifier for the clause, e.g., 'for_1' or 'against_2'")
    title: str = Field(description="Short title or section heading")
    text: str = Field(description="Full text or statutory language of the clause")


class CaseInput(BaseModel):
    case_name: Optional[str] = Field(
        default=None,
        description="Formal name of the case (e.g., 'United States v. Holmes')",
    )
    hypothesis: str = Field(description="The central legal hypothesis/charge being evaluated")
    historical_outcome: Optional[str] = Field(
        default=None,
        description="Actual historical real-world verdict for benchmarking and accuracy comparison",
    )
    for_evidence: List[Clause] = Field(
        default_factory=list,
        description="Pool of clauses supporting the hypothesis (accessible strictly by the Advocate)",
    )
    against_evidence: List[Clause] = Field(
        default_factory=list,
        description="Pool of clauses opposing/defending the hypothesis (accessible strictly by the Defender)",
    )
    standard_of_proof: str = Field(
        default="Beyond a reasonable doubt",
        description="Standard of proof (e.g., 'Beyond a reasonable doubt' or 'Preponderance of evidence')",
    )
    verdict_mode: str = Field(
        default="guilty_not_guilty",
        description="Verdict framing: 'guilty_not_guilty', 'entails_does_not_entail', or 'liable_not_liable'",
    )


class StatementOutput(BaseModel):
    grouped_clause_ids: List[str] = Field(
        description="IDs of 1 to 3 closely related clauses selected from the agent's available pool",
    )
    exact_document_statements: List[str] = Field(
        description="Exact, verbatim excerpts from the selected clauses (no paraphrasing)",
    )
    argument: str = Field(
        description="Legal interpretation applying the exact quoted text to prove or refute the hypothesis",
    )


class RebuttalOutput(BaseModel):
    textual_fidelity_check: str = Field(
        description="Critique evaluating whether the opponent understood the quotes or distorted/stretched their plain meaning",
    )
    substantive_rebuttal: str = Field(
        description="Substantive counter-argument explaining why the opponent's claim fails under legal rules or exceptions",
    )


class PhaseExchange(BaseModel):
    phase: str = Field(description="'Phase A' or 'Phase B'")
    leading_agent: str = Field(description="'Advocate' or 'Defender'")
    statement: StatementOutput
    rebuttal: RebuttalOutput


class RoundResult(BaseModel):
    round_number: int
    phase_a: Optional[PhaseExchange] = Field(default=None, description="Phase A exchange (None if skipped)")
    phase_b: Optional[PhaseExchange] = Field(default=None, description="Phase B exchange (None if skipped)")


class FidelityAssessment(BaseModel):
    advocate_fidelity: str = Field(description="Assessment of Advocate's adherence to verbatim clause text")
    defender_fidelity: str = Field(description="Assessment of Defender's adherence to verbatim clause text")


class JudgmentOutput(BaseModel):
    verdict: str = Field(description="Final decision: e.g. 'Guilty' or 'Not Guilty'")
    verdict_type: str = Field(description="The verdict mode used (e.g. 'guilty_not_guilty')")
    probability_positive: float = Field(
        ge=0.0,
        le=1.0,
        description="Calibrated probability of the positive verdict (e.g., P(Guilty) or P(Entails))",
    )
    probability_negative: float = Field(
        ge=0.0,
        le=1.0,
        description="Calibrated probability of the negative verdict (e.g., P(Not Guilty) or P(Does Not Entail))",
    )
    fidelity_assessment: FidelityAssessment
    decisive_clauses: List[str] = Field(
        description="List of clause IDs that ultimately tipped the balance of judgment",
    )
    judicial_reasoning: str = Field(
        description="Comprehensive judicial opinion explaining why one side prevailed",
    )
