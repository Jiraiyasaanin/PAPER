from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich import box
from lmad.models.schemas import CaseInput, RoundResult, JudgmentOutput

console = Console()


def print_case_header(case_input: CaseInput, model_name: str, max_rounds: int) -> None:
    """Prints a styled header summarizing the case and debate configuration."""
    table = Table(show_header=False, box=box.ROUNDED, expand=True)
    table.add_column("Field", style="bold cyan", width=22)
    table.add_column("Details", style="white")

    if case_input.case_name:
        table.add_row("Case Name", f"[bold yellow]{case_input.case_name}[/bold yellow]")
    table.add_row("Hypothesis", case_input.hypothesis)
    if case_input.historical_outcome:
        table.add_row("Historical Benchmark", f"[italic green]{case_input.historical_outcome}[/italic green]")
    table.add_row("Verdict Mode", case_input.verdict_mode)
    table.add_row("Standard of Proof", case_input.standard_of_proof)
    table.add_row("For Clauses Count", str(len(case_input.for_evidence)))
    table.add_row("Against Clauses Count", str(len(case_input.against_evidence)))
    table.add_row("Model (Groq)", model_name)
    if max_rounds is None:
        table.add_row("Rounds Mode", "Automatic (Terminates when all evidence is presented)")
    else:
        table.add_row("Max Rounds Cap", str(max_rounds))

    panel = Panel(
        table,
        title="[bold yellow]⚖️  LEGAL MULTI-AGENT DEBATE (LMAD) SYSTEM[/bold yellow]",
        border_style="bright_blue",
    )
    console.print()
    console.print(panel)
    console.print()


def print_round(round_result: RoundResult) -> None:
    """Prints formatted output for a single round including Phase A and Phase B."""
    round_num = round_result.round_number
    console.print(
        f"\n[bold magenta]══════════════════════════════════════════════════════════════════════[/bold magenta]"
    )
    console.print(f"[bold magenta]                         ROUND {round_num}                         [/bold magenta]")
    console.print(
        f"[bold magenta]══════════════════════════════════════════════════════════════════════[/bold magenta]\n"
    )

    # ---------------- Phase A ----------------
    if round_result.phase_a:
        pa = round_result.phase_a
        stmt_a = pa.statement
        reb_a = pa.rebuttal

        pa_content = (
            f"[bold underline cyan]ADVOCATE STATEMENT[/bold underline cyan]\n"
            f"[bold]Grouped Clauses:[/bold] {', '.join(stmt_a.grouped_clause_ids)}\n\n"
            f"[bold]Exact Document Statements:[/bold]\n"
            + "\n".join(f"  • [italic green]\"{q}\"[/italic green]" for q in stmt_a.exact_document_statements)
            + f"\n\n[bold]Argument:[/bold]\n{stmt_a.argument}\n\n"
            f"[bold red]────────────────────────────────────────────────────────────[/bold red]\n"
            f"[bold underline red]DEFENDER REBUTTAL[/bold underline red]\n"
            f"[bold]Fidelity Check (Misquotation/Distortion Audit):[/bold]\n"
            f"{reb_a.textual_fidelity_check}\n\n"
            f"[bold]Substantive Counter-Argument:[/bold]\n"
            f"{reb_a.substantive_rebuttal}"
        )

        console.print(
            Panel(
                pa_content,
                title=f"[bold cyan]PHASE A (Advocate Leading)[/bold cyan]",
                border_style="cyan",
                box=box.ROUNDED,
            )
        )
    else:
        console.print(
            Panel(
                "[italic gray]Advocate has already presented all evidence. Phase A skipped.[/italic gray]",
                title=f"[dim cyan]PHASE A (Skipped)[/dim cyan]",
                border_style="dim cyan",
                box=box.ROUNDED,
            )
        )

    # ---------------- Phase B ----------------
    if round_result.phase_b:
        pb = round_result.phase_b
        stmt_b = pb.statement
        reb_b = pb.rebuttal

        pb_content = (
            f"[bold underline red]DEFENDER STATEMENT[/bold underline red]\n"
            f"[bold]Grouped Clauses:[/bold] {', '.join(stmt_b.grouped_clause_ids)}\n\n"
            f"[bold]Exact Document Statements:[/bold]\n"
            + "\n".join(f"  • [italic yellow]\"{q}\"[/italic yellow]" for q in stmt_b.exact_document_statements)
            + f"\n\n[bold]Defense Argument:[/bold]\n{stmt_b.argument}\n\n"
            f"[bold cyan]────────────────────────────────────────────────────────────[/bold cyan]\n"
            f"[bold underline cyan]ADVOCATE REBUTTAL[/bold underline cyan]\n"
            f"[bold]Fidelity Check (Misquotation/Distortion Audit):[/bold]\n"
            f"{reb_b.textual_fidelity_check}\n\n"
            f"[bold]Substantive Counter-Argument:[/bold]\n"
            f"{reb_b.substantive_rebuttal}"
        )

        console.print(
            Panel(
                pb_content,
                title=f"[bold red]PHASE B (Defender Leading)[/bold red]",
                border_style="red",
                box=box.ROUNDED,
            )
        )
    else:
        console.print(
            Panel(
                "[italic gray]Defender has already presented all evidence. Phase B skipped.[/italic gray]",
                title=f"[dim red]PHASE B (Skipped)[/dim red]",
                border_style="dim red",
                box=box.ROUNDED,
            )
        )


def print_judgment(judgment: JudgmentOutput, historical_outcome: str = None) -> None:
    """Prints the final judicial ruling and calibrated probability scores."""
    console.print(
        f"\n[bold gold1]══════════════════════════════════════════════════════════════════════[/bold gold1]"
    )
    console.print(f"[bold gold1]                     FINAL JUDICIAL ADJUDICATION                      [/bold gold1]")
    console.print(
        f"[bold gold1]══════════════════════════════════════════════════════════════════════[/bold gold1]\n"
    )

    verdict_color = "bold green" if "not" in judgment.verdict.lower() else "bold red"
    verdict_text = Text(judgment.verdict.upper(), style=verdict_color)

    p_pos_pct = f"{judgment.probability_positive * 100:.1f}%"
    p_neg_pct = f"{judgment.probability_negative * 100:.1f}%"

    summary_table = Table(box=box.SIMPLE_HEAVY, expand=True)
    summary_table.add_column("Metric", style="bold white", width=28)
    summary_table.add_column("Score / Result", style="bold")

    summary_table.add_row("System Verdict", verdict_text)
    if historical_outcome:
        match_style = "bold green" if judgment.verdict.lower() in historical_outcome.lower() else "bold yellow"
        summary_table.add_row("Actual Historical Verdict", f"[{match_style}]{historical_outcome}[/{match_style}]")
    summary_table.add_row("P(Positive Verdict)", f"[bold red]{p_pos_pct}[/bold red]")
    summary_table.add_row("P(Negative Verdict)", f"[bold green]{p_neg_pct}[/bold green]")
    summary_table.add_row("Decisive Clauses", ", ".join(judgment.decisive_clauses) or "None cited specifically")

    console.print(summary_table)

    fidelity_content = (
        f"[bold]Advocate Fidelity Audit:[/bold]\n{judgment.fidelity_assessment.advocate_fidelity}\n\n"
        f"[bold]Defender Fidelity Audit:[/bold]\n{judgment.fidelity_assessment.defender_fidelity}"
    )
    console.print(Panel(fidelity_content, title="[bold]Textual Fidelity & Grounding Assessment[/bold]", border_style="blue"))

    reasoning_panel = Panel(
        judgment.judicial_reasoning,
        title="[bold gold1]Judicial Opinion & Legal Rationale[/bold gold1]",
        border_style="gold1",
        box=box.ROUNDED,
    )
    console.print(reasoning_panel)
    console.print()
