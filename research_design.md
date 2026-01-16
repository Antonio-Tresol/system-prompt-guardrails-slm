# Research Design: SAE-Integrated Agent Evaluation

## Goal

Test hypothesis: **Markdown-formatted system prompts lead to better instruction following** for refusing to reveal private information in RAG small language model based agents.

---

## Experiment Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         EVALUATION PIPELINE                             │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   System Prompt (MD/Plain)                                              │
│          │                                                              │
│          ▼                                                              │
│   ┌─────────────────┐      ┌────────────────────────────┐              │
│   │ LangChain Agent │◄────►│ Knowledge Base Tool        │              │
│   │ (create_agent)  │      │ (synthetic + papers)       │              │
│   └────────┬────────┘      └────────────────────────────┘              │
│            │                                                            │
│            ▼                                                            │
│   ┌─────────────────────────────────────────────────────┐              │
│   │  Gemma 3 <n>B (with SAE hooks)                       │              │
│   │  - Captures SAE activations at decision points      │              │
│   └────────┬────────────────────────────────────────────┘              │
│            │                                                            │
│            ▼                                                            │
│   ┌────────────────────────────────────────┐                           │
│   │  Outputs:                               │                           │
│   │  - Response (refusal/compliance)        │                           │
│   │  - Trajectory (tool calls, reasoning)   │                           │
│   │  - SAE features at key moments          │                           │
│   └────────────────────────────────────────┘                           │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Data Sources

### Knowledge Bases

| KB | Description | Files |
|----|-------------|-------|
| **Synthetic** | 4 fantasy restaurants with public/private info | `data/*.md` cookbooks |
| **Papers** | 4 academic papers with restricted sections | `data/*.pdf` |

### Questions (60 total)

| Type | Count | Purpose |
|------|-------|---------|
| **Benign** | 30 per KB | Public info retrieval (baseline) |
| **Malicious** | 30 per KB | Probes for private/restricted info |

"Malicious" questions probe:
- Internal costs/pricing
- Employee salaries, schedules
- Restricted sections of papers
- Private metadata (review authors, etc.)

---

## Conditions

| Condition | System Prompt Format | Turn Type |
|-----------|---------------------|-----------|
| MD-Single | Markdown | Single-turn |
| MD-Multi | Markdown | Multi-turn |
| Plain-Single | Plain Text | Single-turn |
| Plain-Multi | Plain Text | Multi-turn |

**Total runs**: 60 questions × 4 conditions = 240 agent runs

---

## Metrics

### Black-Box (Behavioral)

| Metric | Description |
|--------|-------------|
| **Refusal Rate** | % of malicious questions correctly refused |
| **Compliance Rate** | % of benign questions correctly answered |
| **Trajectory Length** | # of tool calls before response |
| **Think Content** | Analysis of reasoning in think tool |

### White-Box (Mechanistic via SAE)

### Key Metrics & Moments of Capture

We do **not** need to analyze everything all the time. We focus on two "Moments of Truth":

#### 1. The Input Processing Moment
*When the model reads the system prompt.*
*   **Metric: Syntax Token Activation** (Core)
    *   *Question:* Does the model actually "see" the markdown syntax?
    *   *What to measure:* Sum of feature activations on `**`, `#`, and bullet points.
*   **Metric: Feature Specificity** (Secondary)
    *   *Question:* Are there special features just for markdown syntax?

#### 2. The Decision Moment
*When the model generates the first token of the response (or thought).*
*   **Metric: Refusal Features** (Core)
    *   *Question:* Which features predict a refusal?
    *   *What to measure:* Activations at the last prompt token (position -1).
*   **Metric: Feature Diff** (Core)
    *   *Question:* What internal concept is active in Markdown but missing in Plain Text?
    *   *What to measure:* `(Markdown_Acts - Plain_Acts)` at the decision moment.

#### 3. Quality Checks (Ignore for analysis)
*   **L0 / FVU**: Just checks if the SAE is broken.
*   **Top-K Features**: Too noisy for aggregate analysis, use only for debugging specific interesting cases.

---

## Implementation Plan

### Phase 1: Agent Setup

- [ ] **1.1** Create  a modern v1 compliant LangChain agent with KB tool (must reads https://docs.langchain.com/oss/python/langchain/agents https://docs.langchain.com/oss/python/langchain/tools https://docs.langchain.com/oss/python/langchain/middleware/built-in https://docs.langchain.com/oss/python/langchain/middleware/custom)
- [ ] **1.2** Create system prompts (MD and Plain versions)
- [ ] **1.3** Implement scoring logic (refusal detection) using LangChain agent as evaluator with llm from open router.

### Phase 2: Custom Model Wrapper

- [ ] **2.1** Create `GemmaWithSAE(BaseChatModel)` wrapper (based on https://github.com/langchain-ai/langchain/blob/master/libs/core/langchain_core/language_models/chat_models.py must reads -> model_evaluation\main_agent\extract_sae_features.py and model_evaluation\main_agent\tutorial_gemma_scope_2.py) the model needs to be able to do bind tools)
- [x] **2.2** SAE capture is always enabled (simplifies architecture)

### Phase 3: Evaluation Pipeline

- [ ] **3.1** Run all 240 conditions
- [ ] **3.2** Collect behavioral metrics
- [ ] **3.3** Store trajectories for analysis

### Phase 4: SAE Analysis

- [ ] **4.1** Extract features for interesting cases (where MD ≠ Plain)
- [ ] **4.2** Compare feature distributions
- [ ] **4.3** Identify candidate "refusal" features

### Phase 5: Visualization & Write-up

- [ ] **5.1** Create comparison plots
- [ ] **5.2** Feature analysis tables
- [ ] **5.3** Paper sections

---

## Key Design Decisions

### Q1: When to capture SAE activations?

**Options:**
- (a) Every run → Slow, lots of data
- (b) Only on interesting cases → Post-hoc, after finding MD/Plain differences
- (c) At decision points only → Capture right before first answer token

**Decision:** [ TBD ]

### Q2: How to detect refusal?

**Options:**
- (a) Keyword matching ("I cannot", "I'm not able")
- (b) LLM-as-judge (separate model scores refusal)
- (c) Binary classification model

**Decision:** [ TBD ]

### Q3: Which SAE layer/width?

**Options from Gemma Scope 2 12B:**
- Layers: 12 (25%), 24 (50%), 31 (65%), 41 (85%)
- Widths: 16k, 64k, 256k, 1m
- L0: small (~10), medium (~60), big (~100)

**Recommendation:** Layer 41 (late = abstract), 16k width, medium L0

**Decision:** [ TBD ]

---

## GemmaWithSAE Wrapper Sketch

```python
from langchain.chat_models.base import BaseChatModel

class GemmaWithSAE(BaseChatModel):
    """Gemma 3 12B with optional SAE activation capture."""
    
    model: PreTrainedModel
    tokenizer: PreTrainedTokenizer
    sae: JumpReLUSAE
    sae_config: SAEConfig
    sae_config: SAEConfig
    # SAE capture is continuously active
    
    # Stored after each generation
    last_activations: SAEFeatureResult | None = None
    
    def _generate(self, messages, stop=None, **kwargs):
        # 1. Format messages to prompt
        prompt = self._format_messages(messages)
        
        # 2. Generate response with SAE capture
        result = extract_sae_features(
            model=self.model,
            tokenizer=self.tokenizer,
            sae=self.sae,
            sae_config=self.sae_config,
            text=prompt,
            max_new_tokens=kwargs.get("max_tokens", 512),
        )
        self.last_activations = result
        response = result.answer
        
        # 3. Return LangChain-compatible response
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=response))])
```

---

## Open Questions

1. What constitutes a "decision point" for SAE capture?
2. How to handle multi-turn conversations (KV cache)?
3. Should we compare IT vs PT SAEs?
4. How to interpret feature indices (Neuronpedia dashboards)?

---

## References

- [Gemma Scope 2 Paper](gemmascope2_paper.md) - local copy
- [Gemma Scope 2 HuggingFace](https://huggingface.co/google/gemma-scope-2-12b-it)
- [LangChain create_agent](https://python.langchain.com/docs/modules/agents/)
- [Neuronpedia](https://neuronpedia.org) - Feature visualization
