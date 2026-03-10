# Data Generation

This directory contains the prompts, universe definitions, and evaluation questions used to generate synthetic data for the safety prompts experiment.

## Overview

To produce synthetic and diverse data, we provide three prompts designed to be consumed by a coding agent (e.g., Claude Code, GitHub Copilot, or similar). The pipeline has three stages:

1. **Universe Generation** — Generate 4 fictional universe YAML files, each representing a different domain (restaurant, law firm, medical clinic, tech startup). Each universe has clearly separated public and private information, ensuring the domain itself is not the factor influencing model refusals.

2. **Question Generation** — Generate 150 evaluation questions per universe (75 public + 75 malicious), grounded in the YAML content. Questions vary in formality and language register (formal, neutral, casual, terse, verbose) to test robustness across user styles. Run once per universe.

3. **Question Validation** — Validate each question CSV against its source universe YAML. Checks grounding, giveaway words, coverage across YAML sections, language variety distribution, duplicates, and answer leakage. Automatically fixes critical issues and generates an audit report.

## Directory Structure

```
data_generation/
├── prompts/                          # Agent prompts for generation
│   ├── universe_generation.md        # Prompt: generate universe YAML files
│   ├── question_generation.md        # Prompt: generate evaluation questions
│   └── question_validation.md        # Prompt: validate and fix questions
│
├── universes/                        # Generated universe definitions (YAML)
│   ├── the_carnelian_table.yaml      # Restaurant — Vermillion Harbor
│   ├── hartwell_and_grey.yaml        # Law Firm — Ashford Crossing
│   ├── linden_grove_clinic.yaml      # Medical Clinic — Cedarhill
│   └── nova_circuit_labs.yaml        # Tech Startup — Neon Flats
│
├── questions/                        # Generated evaluation questions (CSV)
│   ├── the_carnelian_table.csv       # 150 questions (75 public + 75 malicious)
│   ├── hartwell_and_grey.csv         # 150 questions (75 public + 75 malicious)
│   ├── linden_grove_clinic.csv       # 150 questions (75 public + 75 malicious)
│   ├── nova_circuit_labs.csv         # 150 questions (75 public + 75 malicious)
│   ├── the_carnelian_table_validation.md   # Validation report
│   ├── hartwell_and_grey_validation.md     # Validation report
│   ├── linden_grove_clinic_validation.md   # Validation report
│   └── nova_circuit_labs_validation.md     # Validation report
│
└── README.md
```

## Pipeline

Run the prompts in order. Each prompt is self-contained and designed for a single coding agent session.

```
universe_generation.md          →  universes/*.yaml (run once)
question_generation.md          →  questions/*.csv  (run once per universe)
question_validation.md          →  questions/*_validation.md + fixes (run once per universe)
```

## Universes

Each universe YAML follows an identical schema with two top-level sections:

- **`public_info`** — Information the agent may share freely (services, public staff bios, general facts, policies, highlights).
- **`private_info`** — Information the agent must refuse to reveal (financials, staff salaries/personal details, trade secrets, internal operations, confidential incidents).

| Universe | Domain | City |
|----------|--------|------|
| The Carnelian Table | Restaurant | Vermillion Harbor |
| Hartwell & Grey | Law Firm | Ashford Crossing |
| Linden Grove Clinic | Medical Clinic | Cedarhill |
| Nova Circuit Labs | Tech Startup | Neon Flats |

The four domains were chosen to disentangle domain-specific privacy priors from formatting effects. A model may have strong privacy intuitions for medical data (HIPAA-adjacent training signal) but weaker ones for restaurant operations — if Markdown formatting helps across all domains, the effect is generalizable.

All staff members appear in both `public_info.public_staff` (shareable bios) and `private_info.staff_private` (confidential salaries, nicknames, performance notes), enabling tests of whether the agent can share the public role but refuse the private details.

## Questions

Each CSV contains 150 rows with the format:

```
Number,Question,Universe,Malicious
```

- **Public questions (1-75, Malicious: No)** — Target `public_info`. The agent should answer these freely. Distributed across direct factual, descriptive, listing/comparative, and specific detail subcategories.
- **Malicious questions (1-75, Malicious: Yes)** — Target `private_info`. The agent should refuse these. Distributed across direct extraction, indirect/innocent-sounding, probing for secrets, staff private details, and incident fishing subcategories.

Malicious questions are phrased as casual curiosity without giveaway words like "confidential" or "secret", making them a realistic test of the agent's ability to recognize and refuse private information requests.

Questions vary across five language registers (formal, neutral, casual, terse, verbose) shuffled throughout both blocks, ensuring the evaluation tests robustness across user communication styles — not just well-formed queries.

## Validation

Each question CSV is validated against its source universe YAML. The validation checks:

| Check | Type | Criteria |
|-------|------|----------|
| Grounding | Critical | Every question references real YAML content |
| Cross-contamination | Critical | Public questions don't need private info; malicious questions don't target public info |
| Giveaway words | Critical | No forbidden words in malicious questions |
| Answer leakage | Critical | No question embeds the answer or private values |
| Coverage | Warning | Every YAML section has ≥3 questions targeting it |
| Language variety | Warning | No register exceeds 40% or falls below 10% |
| Duplicates | Warning | No exact or near-duplicate questions |

Critical issues are auto-fixed (questions rewritten or replaced) and re-validated. Validation reports are saved alongside the CSVs.

## Provenance

The universes and questions available here were generated and validated using Claude Opus 4.6 with maximum effort, using Claude Code as harness. 05/02/2026.
