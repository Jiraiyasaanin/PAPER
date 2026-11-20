import pytest
from unittest.mock import MagicMock, patch
from lmad.models.schemas import (
    Clause,
    CaseInput,
    StatementOutput,
    RebuttalOutput,
    JudgmentOutput,
    FidelityAssessment,
)
from lmad.orchestrator.clause_tracker import ClauseTracker
from lmad.orchestrator.debate import DebateOrchestrator
from lmad.models.prompts import (
    get_advocate_statement_prompt,
    get_defender_rebuttal_prompt,
    get_defender_statement_prompt,
    get_advocate_rebuttal_prompt,
    get_judge_evaluation_prompt,
)


@pytest.fixture
def sample_case():
    return CaseInput(
        hypothesis="The accused committed breach of trust under Section 406.",
        standard_of_proof="Beyond a reasonable doubt",
        verdict_mode="guilty_not_guilty",
        for_evidence=[
            Clause(id="for_1", title="Statute", text="Whoever converts property commits breach."),
            Clause(id="for_2", title="Transfer", text="Transferred $50k to personal wallet."),
            Clause(id="for_3", title="Admission", text="Admitted transfer via chat."),
        ],
        against_evidence=[
            Clause(id="against_1", title="Authorization", text="Authorized to hedge funds."),
            Clause(id="against_2", title="Repayment", text="Full amount returned within 48 hours."),
        ],
    )


def test_clause_tracker_isolation_and_lifecycle(sample_case):
    tracker = ClauseTracker(
        for_clauses=sample_case.for_evidence,
        against_clauses=sample_case.against_evidence,
    )

    # Initial state
    assert len(tracker.get_unused_for()) == 3
    assert len(tracker.get_unused_against()) == 2
    assert tracker.has_remaining_evidence() is True

    # Mark some For clauses used
    tracker.mark_for_used(["for_1", "for_2"])
    assert len(tracker.get_unused_for()) == 1
    assert tracker.get_unused_for()[0].id == "for_3"
    assert len(tracker.get_unused_against()) == 2

    # Mark some Against clauses used
    tracker.mark_against_used(["against_1"])
    assert len(tracker.get_unused_against()) == 1
    assert tracker.get_unused_against()[0].id == "against_2"

    # Mark all remaining
    tracker.mark_for_used(["for_3"])
    tracker.mark_against_used(["against_2"])
    assert len(tracker.get_unused_for()) == 0
    assert len(tracker.get_unused_against()) == 0
    assert tracker.has_remaining_evidence() is False


def test_prompt_formatting(sample_case):
    adv_prompt = get_advocate_statement_prompt(
        hypothesis=sample_case.hypothesis,
        unused_for_clauses=sample_case.for_evidence,
        transcript_history="",
        round_number=1,
    )
    assert sample_case.hypothesis in adv_prompt
    assert "for_1" in adv_prompt
    assert "exact_document_statements" in adv_prompt

    def_reb_prompt = get_defender_rebuttal_prompt(
        hypothesis=sample_case.hypothesis,
        advocate_statement={
            "grouped_clause_ids": ["for_1"],
            "exact_document_statements": ["Whoever converts property commits breach."],
            "argument": "Claimant proved conversion.",
        },
        transcript_history="",
        round_number=1,
    )
    assert "textual_fidelity_check" in def_reb_prompt
    assert "substantive_rebuttal" in def_reb_prompt

    judge_prompt = get_judge_evaluation_prompt(
        hypothesis=sample_case.hypothesis,
        for_clauses=sample_case.for_evidence,
        against_clauses=sample_case.against_evidence,
        full_transcript="Sample transcript",
        standard_of_proof=sample_case.standard_of_proof,
        verdict_mode="guilty_not_guilty",
        positive_label="Guilty",
        negative_label="Not Guilty",
    )
    assert "Guilty" in judge_prompt
    assert "Not Guilty" in judge_prompt
    assert "probability_positive" in judge_prompt


def test_judgment_output_validation():
    judgment = JudgmentOutput(
        verdict="Guilty",
        verdict_type="guilty_not_guilty",
        probability_positive=0.85,
        probability_negative=0.15,
        fidelity_assessment=FidelityAssessment(
            advocate_fidelity="Accurate verbatim quotes",
            defender_fidelity="Mild reach on exceptions",
        ),
        decisive_clauses=["for_1", "for_2"],
        judicial_reasoning="The evidence established conversion without valid defense.",
    )
    assert judgment.verdict == "Guilty"
    assert round(judgment.probability_positive + judgment.probability_negative, 2) == 1.0


@patch("lmad.agents.debater.DebaterAgent._call_groq_json")
@patch("lmad.agents.judge.JudgeAgent._call_groq_json")
def test_full_orchestrator_mock_run(mock_judge_call, mock_debater_call, sample_case):
    # Mock debater returns
    mock_debater_call.side_effect = [
        # Round 1 - Phase A Statement (Advocate)
        {
            "grouped_clause_ids": ["for_1", "for_2"],
            "exact_document_statements": ["Quote for_1", "Quote for_2"],
            "argument": "Advocate argument round 1.",
        },
        # Round 1 - Phase A Rebuttal (Defender)
        {
            "textual_fidelity_check": "Quote fidelity verified.",
            "substantive_rebuttal": "Defender counter to Phase A.",
        },
        # Round 1 - Phase B Statement (Defender)
        {
            "grouped_clause_ids": ["against_1"],
            "exact_document_statements": ["Quote against_1"],
            "argument": "Defender argument round 1.",
        },
        # Round 1 - Phase B Rebuttal (Advocate)
        {
            "textual_fidelity_check": "Defender quoted accurately.",
            "substantive_rebuttal": "Advocate counter to Phase B.",
        },
    ]

    # Mock judge return
    mock_judge_call.return_value = {
        "verdict": "Guilty",
        "verdict_type": "guilty_not_guilty",
        "probability_positive": 0.75,
        "probability_negative": 0.25,
        "fidelity_assessment": {
            "advocate_fidelity": "High",
            "defender_fidelity": "Moderate",
        },
        "decisive_clauses": ["for_1"],
        "judicial_reasoning": "Guilt established beyond reasonable doubt.",
    }

    orchestrator = DebateOrchestrator(
        case_input=sample_case,
        max_rounds=1,
    )

    rounds, judgment, transcript = orchestrator.run()

    assert len(rounds) == 1
    r1 = rounds[0]
    assert r1.round_number == 1
    assert r1.phase_a.statement.grouped_clause_ids == ["for_1", "for_2"]
    assert r1.phase_b.statement.grouped_clause_ids == ["against_1"]

    assert judgment.verdict == "Guilty"
    assert judgment.probability_positive == 0.75
    assert judgment.probability_negative == 0.25
    assert "ROUND 1" in transcript


@patch("lmad.agents.debater.DebaterAgent._call_groq_json")
@patch("lmad.agents.judge.JudgeAgent._call_groq_json")
def test_debate_termination_when_all_evidence_presented(mock_judge_call, mock_debater_call):
    # Case where each side has exactly 1 evidence clause
    simple_case = CaseInput(
        hypothesis="Breach occurred.",
        standard_of_proof="Preponderance",
        verdict_mode="guilty_not_guilty",
        for_evidence=[Clause(id="for_single", title="For", text="Violation text.")],
        against_evidence=[Clause(id="against_single", title="Against", text="Defense text.")],
    )

    mock_debater_call.side_effect = [
        # Round 1 Phase A
        {"grouped_clause_ids": ["for_single"], "exact_document_statements": ["Violation text."], "argument": "Proven."},
        {"textual_fidelity_check": "Accurate.", "substantive_rebuttal": "Rebuttal A."},
        # Round 1 Phase B
        {"grouped_clause_ids": ["against_single"], "exact_document_statements": ["Defense text."], "argument": "Excused."},
        {"textual_fidelity_check": "Accurate.", "substantive_rebuttal": "Rebuttal B."},
    ]

    mock_judge_call.return_value = {
        "verdict": "Not Guilty",
        "verdict_type": "guilty_not_guilty",
        "probability_positive": 0.40,
        "probability_negative": 0.60,
        "fidelity_assessment": {"advocate_fidelity": "High", "defender_fidelity": "High"},
        "decisive_clauses": ["against_single"],
        "judicial_reasoning": "Defense excused the violation.",
    }

    # Ask for 5 rounds, but all evidence will be presented in Round 1
    orchestrator = DebateOrchestrator(
        case_input=simple_case,
        max_rounds=5,
    )

    rounds, judgment, transcript = orchestrator.run()

    # Even though max_rounds was 5, it MUST terminate after Round 1 because all evidence was presented!
    assert len(rounds) == 1
    assert orchestrator.tracker.all_evidence_presented() is True
    assert "DEBATE TERMINATION" in transcript


@patch("lmad.agents.debater.DebaterAgent._call_groq_json")
@patch("lmad.agents.judge.JudgeAgent._call_groq_json")
def test_phase_skipping_when_agent_runs_out_of_evidence(mock_judge_call, mock_debater_call):
    # Advocate has 2 clauses, Defender has 1 clause
    asym_case = CaseInput(
        hypothesis="Breach occurred.",
        standard_of_proof="Preponderance",
        verdict_mode="guilty_not_guilty",
        for_evidence=[
            Clause(id="for_1", title="For 1", text="Violation text 1."),
            Clause(id="for_2", title="For 2", text="Violation text 2."),
        ],
        against_evidence=[
            Clause(id="against_1", title="Against 1", text="Defense text 1."),
        ],
    )

    mock_debater_call.side_effect = [
        # Round 1 - Phase A (Advocate presents for_1)
        {"grouped_clause_ids": ["for_1"], "exact_document_statements": ["Violation text 1."], "argument": "Proven 1."},
        {"textual_fidelity_check": "Accurate.", "substantive_rebuttal": "Rebuttal 1."},
        # Round 1 - Phase B (Defender presents against_1)
        {"grouped_clause_ids": ["against_1"], "exact_document_statements": ["Defense text 1."], "argument": "Excused 1."},
        {"textual_fidelity_check": "Accurate.", "substantive_rebuttal": "Rebuttal B1."},
        # Round 2 - Phase A (Advocate presents for_2)
        {"grouped_clause_ids": ["for_2"], "exact_document_statements": ["Violation text 2."], "argument": "Proven 2."},
        {"textual_fidelity_check": "Accurate.", "substantive_rebuttal": "Rebuttal 2."},
        # Round 2 - Phase B SHOULD BE SKIPPED because Defender has 0 unused clauses left!
    ]

    mock_judge_call.return_value = {
        "verdict": "Guilty",
        "verdict_type": "guilty_not_guilty",
        "probability_positive": 0.80,
        "probability_negative": 0.20,
        "fidelity_assessment": {"advocate_fidelity": "High", "defender_fidelity": "High"},
        "decisive_clauses": ["for_2"],
        "judicial_reasoning": "Advocate presented further conclusive proof in Round 2.",
    }

    orchestrator = DebateOrchestrator(case_input=asym_case)
    rounds, judgment, transcript = orchestrator.run()

    # Round 1 has both phases
    assert len(rounds) == 2
    assert rounds[0].phase_a is not None
    assert rounds[0].phase_b is not None

    # Round 2 has Phase A, but Phase B is SKIPPED
    assert rounds[1].phase_a is not None
    assert rounds[1].phase_b is None
    assert "PHASE B SKIPPED" in transcript
    assert orchestrator.tracker.all_evidence_presented() is True
