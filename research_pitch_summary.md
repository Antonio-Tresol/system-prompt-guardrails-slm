# Does Formatting Matter? Investigating Markdown System Prompts and Safety Behavior in Small Language Models

## What

This research investigates whether **Markdown formatting in system prompts improves instruction following** in the Gemma 3 family of small language models (1B–27B parameters) when deployed as safety-critical RAG agents. Specifically, we test whether models are better at refusing to reveal private information when their system prompt uses structured Markdown (headings, bold, lists) versus identical content in plain text.

The hypothesis is straightforward: if formatting helps the model parse and prioritize instructions, we should see higher refusal rates on private questions under Markdown prompts, with no loss in compliance on public questions. Beyond measuring this behavioral difference, we use sparse autoencoders (SAEs) to look *inside* the model and identify whether formatting changes the internal representations that drive refusal decisions.

## Why

Prompt engineering is central to deploying language models safely, yet most guidance on prompt design is anecdotal. Practitioners widely assume that structured formatting helps models follow instructions, but there is limited empirical evidence — particularly for small models, which are increasingly deployed in resource-constrained or latency-sensitive settings where large frontier models are impractical.

This matters for three reasons:

**Practical impact.** If a simple formatting change measurably improves safety behavior, it is a zero-cost intervention available to every practitioner. If it doesn't, the community can stop cargo-culting Markdown in system prompts and focus on interventions that actually work.

**Mechanistic understanding.** Behavioral evaluations tell us *whether* something works, not *why*. By capturing SAE features at the moment the model decides to refuse or comply, we can identify which internal representations are activated by Markdown but absent in plain text. This connects prompt formatting to specific computational mechanisms inside the model — a contribution to mechanistic interpretability, not just prompt engineering.

**Scaling behavior.** Testing across Gemma 3 model sizes (1B, 4B, 12B, 27B) reveals whether formatting sensitivity is a property of small models that disappears at scale, or a consistent effect across the family. This informs practitioners about which model sizes benefit most from careful prompt design.

## How

### Experimental Setup

We construct a controlled evaluation environment with four fictional organization universes spanning diverse domains (restaurant, law firm, medical clinic, tech startup). Each universe is defined as a YAML file with clearly separated public information (shareable) and private information (must refuse). The domain diversity ensures that any observed effect of formatting is not confounded by domain-specific privacy priors — a model may have strong intuitions about medical privacy from training but weaker ones for restaurant operations.

For each universe, we generate 150 evaluation questions: 75 targeting public information (the agent should answer) and 75 targeting private information (the agent should refuse). Questions use varied language registers — formal, neutral, casual, terse, and verbose — to test robustness across user communication styles. Malicious questions are framed as casual curiosity without giveaway words like "confidential" or "secret," making them a realistic test of the agent's ability to recognize private information requests.

### Agent Architecture

The safety agent is built with LangChain and has access to two tools: a reasoning scratchpad and a knowledge base search that returns document chunks tagged with privacy labels. The agent's system prompt contains identical content across conditions — organization context, tool instructions, and a step-by-step privacy decision workflow — with the only difference being Markdown formatting versus plain text.

A KB Generator Agent pre-generates and caches all knowledge base content so both conditions see identical retrieval results, isolating the formatting variable. Each question is run through both prompt conditions, yielding 1,200 agent runs per model size (600 questions × 2 formats).

### Evaluation

Refusal and compliance are classified by an LLM-as-judge pipeline rather than keyword heuristics, handling edge cases like partial compliance and hedged refusals. A second judge verifies that compliance responses are grounded in the knowledge base sources, with automatic retries for hallucinated responses.

### Mechanistic Analysis

On every run, we extract SAE features at two layers per model (middle ~50% and upper ~85% depth) using Gemma Scope 2 JumpReLU SAEs. The analysis follows a pre-registered three-level plan:

**Level 1** computes the mean activation difference between Markdown and Plain conditions at the decision point (last token before generation) for all 16,384 features per layer. Significance is established via permutation testing with Benjamini-Hochberg FDR correction to control for multiple comparisons. The top surviving features are looked up on Neuronpedia for semantic interpretation.

**Level 2** identifies features that predict successful refusal regardless of formatting, then measures their overlap (Jaccard similarity) with the format-sensitive features from Level 1. High overlap means Markdown activates the same circuits that drive refusals — a mechanistic explanation for any behavioral improvement.

**Level 3** checks whether Markdown produces higher aggregate activation magnitude at the decision point, a coarse measure of whether the model engages more features when processing formatted prompts.

### Expected Contributions

This work contributes an empirical answer to a common but untested assumption in prompt engineering, a mechanistic analysis connecting prompt formatting to internal model representations via SAEs, and a reusable evaluation framework (universes, questions, agent, judge pipeline) for studying prompt design effects on safety behavior. Whether the hypothesis is confirmed or refuted, the result is informative: either formatting is a reliable safety lever, or the community should look elsewhere.
