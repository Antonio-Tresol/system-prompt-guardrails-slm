# Reference Search Guide

This document outlines the key topics that need supporting references for the research paper. Each section describes what we're claiming or doing, why we need a reference, and suggested search terms.

---

## 1. Prompt Formatting Effects on LLM Behavior

**What we claim**: Markdown formatting in system prompts may improve instruction following in small language models.

**What to look for**:
- Studies comparing structured vs. unstructured prompts on LLM task performance
- Evidence that formatting (headers, bold, lists) affects model comprehension or compliance
- Prompt engineering best practices papers that discuss formatting specifically
- Any work on how models process markup syntax tokens internally

**Search terms**: `prompt formatting LLM performance`, `markdown prompts language models`, `structured prompts instruction following`, `system prompt design`, `prompt engineering formatting effects`

**Notes**: This is our core claim. There may not be much direct literature — if so, that strengthens our novelty argument. Tangential work on prompt sensitivity (e.g., prompt order effects, delimiter effects) is also relevant.

---

## 2. LLM Safety and Refusal Behavior

**What we claim**: Safety agents should refuse to reveal private information when instructed via system prompts.

**What to look for**:
- Papers on LLM refusal behavior and alignment
- Work on privacy preservation in RAG systems
- System prompt adherence / instruction hierarchy research
- Jailbreaking and prompt injection defenses (our malicious questions are a soft version of this)
- Studies measuring refusal rates under different conditions

**Search terms**: `LLM refusal behavior alignment`, `privacy RAG retrieval augmented generation`, `system prompt adherence language models`, `instruction hierarchy LLM`, `prompt injection defense`, `LLM safety evaluation benchmarks`

---

## 3. Sparse Autoencoders for Mechanistic Interpretability

**What we claim**: SAE features can reveal internal model representations that differ between conditions.

**What to look for**:
- Gemma Scope / Gemma Scope 2 papers (Google DeepMind)
- SAE interpretability papers (Anthropic, EleutherAI, independent researchers)
- JumpReLU SAE architecture papers
- Work on using SAEs to understand model behavior (not just feature extraction)
- Feature visualization and Neuronpedia-related work
- Any work connecting SAE features to behavioral outcomes (refusal, safety, reasoning)

**Search terms**: `sparse autoencoder mechanistic interpretability`, `Gemma Scope SAE`, `JumpReLU sparse autoencoder`, `SAE feature analysis language models`, `mechanistic interpretability safety`, `dictionary learning neural networks transformers`

**Key authors/groups to check**: Anthropic interpretability team, Google DeepMind (Gemma team), Neel Nanda, Joseph Bloom, Arthur Conmy

---

## 4. LLM-as-Judge Evaluation

**What we claim**: An LLM judge (GLM-4.7-Flash) classifies refusal/compliance more accurately than keyword heuristics.

**What to look for**:
- Papers establishing LLM-as-judge as a valid evaluation methodology
- Studies on judge model agreement with human evaluators
- Work on using smaller/cheaper models as judges
- Any analysis of judge reliability, bias, or failure modes
- Groundedness evaluation in RAG systems

**Search terms**: `LLM as judge evaluation`, `language model automated evaluation`, `LLM judge agreement human`, `groundedness evaluation RAG`, `automated refusal classification`, `G-Eval LLM evaluation`

**Key papers to find**: The original "Judging LLM-as-a-Judge" paper (Zheng et al.), MT-Bench, AlpacaEval methodology papers

---

## 5. RAG Architecture and Evaluation

**What we claim**: We use a RAG-based agent with privacy-labeled document chunks.

**What to look for**:
- Foundational RAG papers (Lewis et al.)
- RAG evaluation frameworks and benchmarks
- Privacy-aware retrieval systems
- Work on document chunk labeling or metadata-driven retrieval
- Agent-based RAG architectures (tool-using agents with retrieval)

**Search terms**: `retrieval augmented generation evaluation`, `RAG privacy information retrieval`, `agent RAG tool use`, `document chunk privacy labeling`, `RAG hallucination evaluation`

---

## 6. Small Language Models as Agents

**What we claim**: We evaluate Gemma 3 (1B-27B) as safety agents, and small models may behave differently from large ones regarding prompt formatting.

**What to look for**:
- Papers on small language model capabilities and limitations
- Gemma 3 technical reports / model cards
- Studies comparing model sizes on instruction following or safety tasks
- Work on tool-calling in small models (limited native support, XML parsing workarounds)
- Scaling laws related to instruction following or safety behavior

**Search terms**: `small language models agents`, `Gemma 3 technical report`, `model size instruction following`, `small LLM tool calling`, `scaling laws safety alignment`, `language model size capability threshold`

---

## 7. Statistical Methods

**What we claim**: We use permutation tests with FDR correction for SAE analysis and standard tests for behavioral comparisons.

**What to look for**:
- Benjamini-Hochberg FDR correction (original paper)
- Permutation tests for high-dimensional comparisons
- Two-proportion z-tests for behavioral experiments
- Mann-Whitney U for non-parametric comparisons
- Statistical methods in ML evaluation papers (what do similar papers cite?)

**Search terms**: `Benjamini Hochberg false discovery rate`, `permutation test high dimensional`, `statistical methods machine learning evaluation`, `multiple comparison correction neuroscience` (neuroscience faces similar multiple comparison issues with brain imaging)

**Notes**: Standard statistical references. One good textbook citation may suffice for most of these. The permutation test + FDR combination is common in neuroscience/genomics — citing precedent from those fields strengthens our methodology.

---

## 8. Prompt Sensitivity and Robustness

**What we claim**: We use varied language registers in evaluation questions to test robustness.

**What to look for**:
- Studies on LLM sensitivity to input phrasing or style
- Work on robustness of model behavior across paraphrased inputs
- Evaluation diversity and question difficulty gradients
- Any evidence that models perform differently with formal vs. casual inputs

**Search terms**: `LLM prompt sensitivity robustness`, `language model paraphrase sensitivity`, `input phrasing LLM performance`, `evaluation robustness language models`, `linguistic register LLM`

---

## 9. Synthetic Data Generation for Evaluation

**What we claim**: We use LLM-generated universes and questions for evaluation, validated through a structured pipeline.

**What to look for**:
- Papers on using LLMs to generate evaluation data
- Synthetic benchmark creation methodology
- Validation of LLM-generated test sets
- Concerns and mitigations around using synthetic data for evaluation

**Search terms**: `synthetic evaluation data LLM`, `LLM generated benchmarks`, `synthetic test set validation`, `automated evaluation dataset creation`

---

## Priority Order

If time is limited, prioritize references in this order:

1. **Sparse Autoencoders / Gemma Scope** — core methodology, most novel aspect
2. **LLM Safety and Refusal** — frames the problem
3. **Prompt Formatting Effects** — direct precedent for our hypothesis (may be sparse)
4. **LLM-as-Judge** — validates our evaluation approach
5. **RAG Architecture** — standard methodology, well-cited
6. **Small Language Models** — contextualizes our model choices
7. **Statistical Methods** — standard references, easy to find
8. **Prompt Sensitivity** — supporting argument
9. **Synthetic Data** — supporting methodology

---

## Format

For each reference found, please record:
- Full citation (authors, title, year, venue/arxiv)
- Which section(s) above it supports (by number)
- One sentence on why it's relevant
- URL (arxiv link, DOI, or paper page)

This will make it easy to slot references into the paper draft later.
