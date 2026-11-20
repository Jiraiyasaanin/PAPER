# Legal Multi-Agent Debate (LMAD) System

A multi-agent legal reasoning system where two adversarial agents (**Advocate** and **Defender**) debate a legal hypothesis against partitioned clause pools, followed by a formal judicial adjudication by a **Judge** outputting calibrated probabilities ($P(\text{Guilty})$ vs $P(\text{Not Guilty})$).

All agents and the judge are powered by the **Groq API** (defaulting to the fast Qwen model family: `qwen/qwen3.8-27b`, with fallback support).

---

## Architecture & Turn Workflow

### 1. Strict Evidence Pool Isolation & Dynamic Grouping
* **For Pool (`for_evidence`)**: Exclusively accessible to the Advocate.
* **Against Pool (`against_evidence`)**: Exclusively accessible to the Defender.
* **Dynamic Grouping**: On each turn, the agent inspects its remaining unused clauses and groups 1 to 3 very similar clauses addressing the same factual or legal issue into a single claim.

### 2. 2-Phase Round Structure
Each round consists of:
* **Phase A (Advocate Leading)**:
  * **Advocate Statement**: `[Exact Document Quotes]` + `[Legal Claim]`.
  * **Defender Rebuttal**: Cross-examines the Advocate's quotes to detect distortion, overreach, or mischaracterization.
* **Phase B (Defender Leading)**:
  * **Defender Statement**: `[Exact Document Quotes]` + `[Defense Claim]`.
  * **Advocate Rebuttal**: Cross-examines the Defender's quotes and defends the affirmative theory.

### 3. Judicial Adjudication
At the conclusion of the rounds, the **Judge LLM** evaluates the full transcript in Groq JSON mode and delivers:
* **Verdict**: `Guilty` / `Not Guilty`, `Entails` / `Does Not Entail`, or `Liable` / `Not Liable`.
* **Calibrated Probabilities**: E.g., $P(\text{Guilty}) = 0.82$, $P(\text{Not Guilty}) = 0.18$.
* **Fidelity Assessment**: Detailed check on whether either party distorted or misquoted clauses.
* **Decisive Clauses & Judicial Opinion**: Rationale explaining why one side prevailed.

---

## Installation & Setup

1. **Clone & Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure Environment Variables**:
   Copy `.env.example` to `.env`:
   ```bash
   cp .env.example .env
   ```
   Add your Groq API key:
   ```env
   GROQ_API_KEY=gsk_your_actual_groq_api_key_here
   GROQ_MODEL=qwen/qwen3.8-27b
   ```

---

## Running the Debate

Rounds are **completely automatic**: the system dynamically groups similar evidence and runs until all clauses on both sides are fully presented and cross-examined. You do not need to specify the number of rounds.

### 1. Famous Historical Benchmark: Theranos Wire Fraud (*United States v. Holmes*)
Compare the AI debate system's verdict against the actual 2022 federal jury conviction:
```bash
python main.py --case examples/theranos_fraud.json
```

### 2. Default Criminal Breach of Trust Case:
```bash
python main.py --case examples/criminal_fraud.json
```

### 3. Contract Breach Dispute:
```bash
python main.py --case examples/contract_breach.json
```

*(Optional: You can pass `--rounds <N>` only if you wish to enforce an artificial maximum cap on the number of rounds.)*

---

## Running Tests

Run the test suite:
```bash
python -m pytest
```
