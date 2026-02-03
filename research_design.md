# Research Design: SAE-Integrated Agent Evaluation

## Goal

Test hypothesis: **Markdown-formatted system prompts lead to better instruction following in gemma 3 family of small language models when used as agents** for refusing to reveal private information in RAG.

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
│   │ (create_agent)  │      │ (Universe Context Aware)   │              │
│   └────────┬────────┘      └────────────────────────────┘              │
│            │                             ▲                              │
│            ▼                             │ (Context Injection)          │
│   ┌─────────────────┐           ┌────────┴─────────┐                   │
│   │ Gemma With SAE  │           │ Universe Context │                   │
│   │ (Wrapper)       │           │ (YAML Definition)│                   │
│   └────────┬────────┘           └──────────────────┘                   │
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

### Knowledge & Question Generation

| Component            | Description                                                                                                               |
| -------------------- | ------------------------------------------------------------------------------------------------------------------------- |
| **Universe Context** | YAML-defined worlds (e.g., *Zenith-9 Space Station*, *Eldoria Magic Shop*) containing specific facts, rules, and secrets. |
| **Deep Agent**       | A prioritized sub-agent (`question_generator`) that creates complex, multi-hop questions based on the Universe Context.   |
| **Ground Truth**     | Each generated question includes an `is_refusal` flag (True/False) and the `universe_context` needed to answer it.        |

### Dataset Structure

Dynamic generation ensures infinite scaling, but for standard evaluation we freeze a set:

| Type            | Definition                                                                                            | Ground Truth       |
| --------------- | ----------------------------------------------------------------------------------------------------- | ------------------ |
| **Non-Refusal** | Questions about public menu items, lore, or general facts defined in the Universe.                    | `is_refusal=False` |
| **Refusal**     | Questions probing explicitly protected info (e.g., "secret ingredients", "staff schedules", "codes"). | `is_refusal=True`  |

---

## Conditions

| Condition    | System Prompt Format | Turn Type   |
| ------------ | -------------------- | ----------- |
| MD-Single    | Markdown             | Single-turn |
| MD-Multi     | Markdown             | Multi-turn  |
| Plain-Single | Plain Text           | Single-turn |
| Plain-Multi  | Plain Text           | Multi-turn  |

**Total runs**: 60 questions × 4 conditions = 240 agent runs

---

## Metrics

### Black-Box (Behavioral)

| Metric                | Description                                |
| --------------------- | ------------------------------------------ |
| **Refusal Rate**      | % of malicious questions correctly refused |
| **Compliance Rate**   | % of benign questions correctly answered   |
| **Trajectory Length** | # of tool calls before response            |
| **Think Content**     | Analysis of reasoning in think tool        |

### White-Box (Mechanistic via SAE)

### Key Metrics & Moments of Capture

We do **not** need to analyze everything all the time. We focus on two "Moments of Truth":

#### 1. The Input Processing Moment

*When the model reads the system prompt.*

* **Metric: Syntax Token Activation** (Core)
  * *Question:* Does the model actually "see" the markdown syntax?
  * *What to measure:* Sum of feature activations on `**`, `#`, and bullet points.
* **Metric: Feature Specificity** (Secondary)
  * *Question:* Are there special features just for markdown syntax?

#### 2. The Decision Moment

*When the model generates the first token of the response (or thought).*

* **Metric: Refusal Features** (Core)
  * *Question:* Which features predict a refusal?
  * *What to measure:* Activations at the last prompt token (position -1).
* **Metric: Feature Diff** (Core)
  * *Question:* What internal concept is active in Markdown but missing in Plain Text?
  * *What to measure:* `(Markdown_Acts - Plain_Acts)` at the decision moment.

#### 3. Quality Checks (Ignore for analysis)

* **L0 / FVU**: Just checks if the SAE is broken.
* **Top-K Features**: Too noisy for aggregate analysis, use only for debugging specific interesting cases.

---

## Implementation Plan

### Phase 1: Agent & Data Infrastructure (Completed)

* [x] **1.1** Modern LangChain/LangGraph agent with KB tool
* [x] **1.2** System prompts (MD and Plain versions)
* [x] **1.3** **Deep Agent Data Generator** (Synthetic questions + Universe Contexts)
* [x] **1.4** **Context-Aware Retrieval** (`search_knowledge_base` uses Universe Context)

### Phase 2: Custom Model Wrapper (Completed)

* [x] **2.1** `GemmaWithSAE(BaseChatModel)` wrapper with SAE capture
* [x] **2.2** Token usage tracking (Input/Output/Context)
* [x] **2.3** Integrated Gemma Scope 2 SAEs (JumpReLU)

### Phase 3: Evaluation Pipeline (In Progress)

* [ ] **3.1** Run comparison: MD vs Plain System Prompts
* [ ] **3.2** Metrics aggregation (Refusal Rate vs Ground Truth)
* [ ] **3.3** Latency and Tokens/sec analysis

### Phase 4: SAE Analysis (Planned)

* [ ] **4.1** Feature activation analysis at "Refusal Decision Point"
* [ ] **4.2** Compare feature diffs: `(Markdown - Plain)`
* [ ] **4.3** Identify "Compliance Features" vs "Safety Features"

### Phase 5: Visualization & Write-up

* [ ] **5.1** Create comparison plots
* [ ] **5.2** Feature analysis tables
* [ ] **5.3** Paper sections

---

## Key Design Decisions

### Q1: When to capture SAE activations?

**Options:**
* (a) Every run → Slow, lots of data
* (b) Only on interesting cases → Post-hoc, after finding MD/Plain differences
* (c) At decision points only → Capture right before first answer token

**Decision:** [ TBD ]

### Q2: How to detect refusal?

**Options:**
* (a) Keyword matching ("I cannot", "I'm not able")
* (b) LLM-as-judge (separate model scores refusal)
* (c) Binary classification model

**Decision:** [ TBD ]

### Q3: Which SAE layer/width?

**Options from Gemma Scope 2 12B:**
* Layers: 12 (25%), 24 (50%), 31 (65%), 41 (85%)
* Widths: 16k, 64k, 256k, 1m
* L0: small (~10), medium (~60), big (~100)

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

* [Gemma Scope 2 Paper](gemmascope2_paper.md) - local copy
* [Gemma Scope 2 HuggingFace](https://huggingface.co/google/gemma-scope-2-12b-it)
* [LangChain create_agent](https://python.langchain.com/docs/modules/agents/)
* [Neuronpedia](https://neuronpedia.org) - Feature visualization
