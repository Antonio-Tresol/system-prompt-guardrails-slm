# Research Design: SAE-Integrated Agent Evaluation

## Goal

Test hypothesis: **Markdown-formatted system prompts lead to better instruction following in Gemma 3 family of small language models when used as agents** for refusing to reveal private information in RAG.

---

## Experiment Overview

```
┌──────────────────────────────────────────────────────────────────────────┐
│                         EVALUATION PIPELINE                              │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  System Prompt (MD / Plain)                                              │
│         │                                                                │
│         ▼                                                                │
│  ┌──────────────────┐      ┌────────────────────────────────┐           │
│  │ Safety Agent      │◄────►│ search_knowledge_base tool     │           │
│  │ (create_agent)    │      │ (returns cached KB chunks)     │           │
│  └───────┬──────────┘      └─────────────┬──────────────────┘           │
│          │                               │                               │
│          │  ┌────────────────┐           │ (Pre-generated & cached)      │
│          │  │ think tool     │           │                               │
│          │  │ (reasoning)    │           ▼                               │
│          │  └────────────────┘  ┌────────────────────────────┐          │
│          │                      │ KB Generator Agent          │          │
│          │                      │ (dynamic privacy prompt)    │          │
│          │                      │ → 1-3 DocumentChunks        │          │
│          │                      │ → privacy_level labels      │          │
│          │                      └─────────────┬──────────────┘          │
│          │                                    │                          │
│          │                                    ▼                          │
│          │                      ┌────────────────────────────┐          │
│          ▼                      │ Universe Context (YAML)     │          │
│  ┌─────────────────────────┐   │ (public + private info)     │          │
│  │ GemmaWithSAE            │   └────────────────────────────┘          │
│  │ (BaseChatModel wrapper) │                                            │
│  │ - Gemma 3 <n>B          │                                            │
│  │ - Multi-layer SAE hooks │                                            │
│  │ - XML tool call parsing │                                            │
│  │ - Token tracking        │                                            │
│  └───────┬─────────────────┘                                            │
│          │                                                               │
│          ▼                                                               │
│  ┌───────────────────────────────────────────────────┐                  │
│  │ LLM-as-Judge (GLM-4.7-Flash via OpenRouter)       │                  │
│  │ 1. Refusal Judge → REFUSAL / COMPLIANCE            │                  │
│  │ 2. Groundedness Judge → GROUNDED / HALLUCINATED    │                  │
│  │    (compliance only; retries agent on hallucination)│                  │
│  └───────┬───────────────────────────────────────────┘                  │
│          │                                                               │
│          ▼                                                               │
│  ┌───────────────────────────────────────────────────┐                  │
│  │ Outputs:                                           │                  │
│  │ - Response (refusal / compliance)                  │                  │
│  │ - Judge classifications + reasoning                │                  │
│  │ - KB sources received by the agent                 │                  │
│  │ - Trajectory (steps, tool calls, token counts)     │                  │
│  │ - SAE features at 2 layers (middle + upper)        │                  │
│  │ - Retry count (if hallucination retries occurred)  │                  │
│  └───────────────────────────────────────────────────┘                  │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## Data Sources

### Data Generation Process

All synthetic data (universes and questions) is generated via **structured
prompt-driven workflows** using coding agents (e.g., Claude Code). The prompts
are stored in `data_generation/prompts/` and designed to be executed by an
agent that reads the prompt, generates the data, and validates the output.

| Step | Prompt File | Output | Description |
|------|------------|--------|-------------|
| 1. Universe generation | `universe_generation.md` | `universes/*.yaml` | Generate 4 YAML universe files with public/private info |
| 2. Question generation | `question_generation.md` | `questions/*.csv` | Generate 150 questions per universe (75 public + 75 private) |
| 3. Question validation | `question_validation.md` | `questions/*_validation.md` | Validate grounding, coverage, variety; auto-fix issues |

This approach replaces the earlier code-based generation pipeline (deleted
`question_generator/` module) with a more flexible prompt-based workflow that
produces higher-quality, validated data with full coverage checks.

### Universe Contexts

Four fictional universes across diverse domains, defined as YAML files in
`data_generation/universes/`, each with a clear separation of public and
private information:

| Universe | Domain | Description |
|----------|--------|-------------|
| **The Carnelian Table** | Restaurant | Meat-focused fine dining in Vermillion Harbor |
| **Hartwell & Grey** | Law Firm | Mid-size litigation and corporate law firm in Ashford Crossing |
| **Linden Grove Clinic** | Medical Clinic | Multi-specialty outpatient clinic in Cedarhill |
| **Nova Circuit Labs** | Tech Startup | AI-powered logistics optimization startup in Neon Flats |

Each YAML follows a standardized schema with domain-appropriate content:
- **Public info**: services/offerings, public staff bios, general info (hours, location, contact), policies, highlights
- **Private info**: financials (revenue, costs, margins), staff private details (salaries, nicknames, personal info), trade secrets, internal operations, confidential incidents

The schema is consistent across domains — `services` entries adapt to be
menu items (restaurant), practice areas (law firm), specialties (clinic), or
products (tech startup), while keeping all key names identical.

### Evaluation Questions

600 evaluation questions (150 per universe) are pre-generated and stored as
static CSV files in `data_generation/questions/`. Each CSV has columns:
`Number`, `Question`, `Universe`, `Malicious`.

Each 150-question set contains 75 public + 75 private questions with
controlled distribution across question types:

| Question Type | Count per Universe | Distribution | Ground Truth |
|---------------|-------------------|--------------|-------------|
| **Public** | 75 | 30 direct factual, 20 descriptive, 15 listing, 10 specific detail | `expects_refusal=False` |
| **Private** | 75 | 20 direct extraction, 20 indirect/innocent, 15 probing, 10 staff private, 10 incident fishing | `expects_refusal=True` |

**Quality controls** enforced during generation:
- All questions are **grounded** in actual YAML content (no hallucinated entities)
- Private questions contain **no giveaway words** (secret, confidential, private, etc.)
- Questions use **varied language registers** (formal, neutral, casual, terse, verbose)
- Full **coverage** across all YAML sections (every staff member, service, secret, incident)
- **No answer leakage** (questions don't embed the answer or private values)

### Dynamic Knowledge Base

Instead of a static vector store, a **KB Generator Agent** creates 1-3
`DocumentChunk` objects per query at runtime. Each chunk carries a
`privacy_level` label (`public`, `private`, `mixed`).

- Uses `@dynamic_prompt` middleware to switch between public/private generation
- Uses `ToolStrategy(GeneratorOutput)` for structured output
- Maintains conversation history via `GeneratorSession` with LangGraph checkpointer
- Backed by an OpenRouter model (configurable via `KB_GENERATOR_MODEL`)
- Receives the full universe YAML for grounded, coherent generation

For evaluation, KB content is **pre-generated and cached** (`kb_cache.json`)
so both Markdown and Plain prompt runs see identical retrieval results.

---

## Conditions

| Condition | System Prompt Format |
|-----------|---------------------|
| Markdown | Formatted with `#`, `**`, numbered lists, bullet points |
| Plain | Same content, no formatting markup |

**Total runs**: 600 questions x 2 conditions = 1200 agent runs per model size

**Model sizes evaluated**: Gemma 3 4B and 12B (both instruction-tuned, int4
quantized). Cross-model comparison tests whether the formatting effect scales
with model capacity.

---

## Safety Agent

The agent (`create_safety_agent`) is built with LangChain's `create_agent`
and has access to two tools:

1. **`think(thought)`**: Reasoning scratchpad. The agent records its plan and
   reasoning before and after tool calls.
2. **`search_knowledge_base(query, num_results=5)`**: Returns cached KB chunks
   with privacy labels (`[Public]`, `[Private]`, `[Mixed]`). Uses
   `EvaluationContext` (dataclass) injected at invocation time.

### System Prompt Comparison

Both prompts contain identical content and instructions:
- Organization context (dynamically loaded from universe YAMLs at module import)
- Tool descriptions and usage instructions
- Step-by-step privacy decision workflow (search -> check label -> answer or refuse)
- Privacy rules (public = share, private = never share)

The only difference is **formatting**: Markdown uses `#` headings, `**bold**`
emphasis, and structured lists; Plain uses flat text.

---

## Model Wrapper: GemmaWithSAE

Custom `BaseChatModel` wrapping Gemma 3 + Gemma Scope 2 SAEs:

```python
class GemmaWithSAE(BaseChatModel):
    """Gemma 3 with always-on multi-layer SAE feature extraction."""

    # Private attrs (Pydantic PrivateAttr)
    _model: PreTrainedModel          # HuggingFace Gemma 3 (1B/4B/12B/27B)
    _tokenizer: PreTrainedTokenizer
    _sae: JumpReLUSAE                # Primary layer SAE
    _sae_config: SAEConfig
    _all_saes: Dict[int, Tuple[JumpReLUSAE, SAEConfig]]  # Multi-layer
    _bound_tools: List[Dict]         # OpenAI-format tool definitions
    _last_activations: SAEFeatureResult | None
    _last_multi_layer_activations: MultiLayerSAEFeatureResult | None
    _total_input_tokens: int
    _total_output_tokens: int

    def _generate(self, messages, ...):
        # 1. Inject tool definitions into system prompt
        # 2. Format messages for Gemma's strict user/model alternation
        #    - System messages merged into user with <system_prompt> tags
        #    - Tool results formatted as <tool_result> user messages
        #    - Consecutive same-role messages merged
        # 3. Apply chat template
        # 4. Generate + extract SAE features (multi-layer if configured)
        # 5. Parse tool calls from <tool_call>func(arg="val")</tool_call> XML
        # 6. Return ChatResult with AIMessage (content + tool_calls)
```

### Tool Calling Format

Since Gemma 3 has limited native tool-calling support, the wrapper uses a custom
XML + Python function call format:

```
<tool_call>search_knowledge_base(query="lamb recipe")</tool_call>
```

Tool calls are parsed using Python's `ast` module for safe, reliable extraction.

### Message Formatting

Gemma's chat template requires strict user/model alternation. The wrapper handles:
- System messages merged into the first user message with `<system_prompt>` XML tags
- Tool results wrapped in `<tool_result>` tags and sent as user messages
- Consecutive same-role messages merged to maintain alternation
- User content wrapped in `<user_message>` tags for clarity when combined

---

## Trajectory Capture

A custom `TrajectoryCapture` middleware (LangChain `AgentMiddleware`) records the
full agent execution as structured `AgentTrace` objects:

- **`before_agent`**: Initializes a new trace
- **`wrap_model_call`**: Records messages snapshot, model response, tool calls, token counts
- **`wrap_tool_call`**: Records tool name, arguments, result, and duration
- **`after_agent`**: Finalizes and stores the completed trace

Each trace contains:
- Full message history at each step
- Model response content and tool call requests
- Tool execution results and timing
- Per-step and aggregate token counts
- Total wall-clock duration

Traces are saved as JSON files in `traces/trace_{uuid}.json`.

---

## Metrics

### Black-Box (Behavioral)

| Metric | Description |
|--------|-------------|
| **Refusal Rate** | % of private questions correctly refused (LLM judge) |
| **Compliance Rate** | % of public questions correctly answered (LLM judge) |
| **Groundedness Rate** | % of compliance responses grounded in KB sources |
| **Hallucination Retry Rate** | % of runs requiring re-tries due to hallucination |
| **Trajectory Length** | # of agent steps (model invocations) before final response |
| **Tool Call Count** | Total tool executions per run |
| **Tool Usage Pattern** | Sequence of think/search calls |
| **Token Efficiency** | Input/output tokens per run |
| **Latency** | Wall-clock duration per run |

### Statistical Tests (Behavioral)

This is a **paired design**: the same 600 questions run through both conditions
with identical cached KB content. All tests account for the pairing.

| Comparison | Test | Details |
|------------|------|---------|
| Refusal rate: Markdown vs. Plain (pooled) | McNemar's test | Primary result. 300 paired private questions. Uses exact binomial when discordant pairs < 25, χ² otherwise. |
| Compliance rate: Markdown vs. Plain (pooled) | McNemar's test | Secondary result. 300 paired public questions. |
| Refusal rate: per-universe breakdown | McNemar's test | Exploratory. 75 paired questions per universe. Underpowered for small effects. |
| Refusal rate: by question subcategory | McNemar's exact | Exploratory. Small discordant counts expected. |
| Trajectory length: Markdown vs. Plain | Wilcoxon signed-rank | Paired non-parametric; count data, likely skewed. |
| Token efficiency: Markdown vs. Plain | Wilcoxon signed-rank | Paired non-parametric; same rationale. |
| Groundedness rate: Markdown vs. Plain | McNemar's test | Paired comparison on questions where both formats complied. |

All primary/secondary tests use α=0.05. Exploratory tests are reported without
multiple comparison correction and flagged as exploratory.

#### Cross-Model Comparison

The same McNemar's test is run **separately per model size** (4B and 12B) to
examine whether the formatting effect scales with model capacity. This is a
key analysis: if Markdown improves refusal only in the larger model, it
suggests that sufficient model capacity is needed to leverage structural
formatting for instruction following.

| Comparison | Test | Details |
|------------|------|---------|
| Refusal rate: MD vs. Plain per model size | McNemar's test | Same paired test, run independently for 4B and 12B. |
| Compliance rate: MD vs. Plain per model size | McNemar's test | Secondary. Checks if formatting affects public question answering. |
| Per-universe refusal diff × model size | Descriptive heatmap | Exploratory. Shows whether the direction/magnitude of the format effect is consistent across domains within each model. |

### Refusal Detection (LLM-as-Judge)

Refusal is classified via an **LLM-as-judge pipeline** (`judge.py`) using
GLM-4.7-Flash via OpenRouter. The judge receives the original question and the
agent's response, then classifies it as **REFUSAL** or **COMPLIANCE** using
structured output (`JudgeClassification` Pydantic model with reasoning-first
chain-of-thought).

- Runs **inline** during evaluation (every run is classified immediately)
- Supports **standalone backfill** on existing CSVs via `uv run judge_results`
- Uses up to 4 attempts (3 retries) for resilience against unreliable providers
- Partial refusals (answer some parts but refuse specific info) are classified
  as refusals

### Groundedness Detection (Hallucination Judge)

A second LLM-as-judge checks whether compliance responses are **grounded** in
the KB sources the agent received (`classify_groundedness`). This ensures
hallucinated data points are detected and retried.

- **KB source tracking**: The formatted KB content the agent received is stored
  in `RunResult.kb_sources` for every run
- **Inline retry loop**: When enabled (`--check-groundedness`), hallucinated
  responses trigger automatic agent re-runs (same cached KB, fresh agent) up
  to `max_groundedness_retries` times (default: 2)
- **Refusals auto-grounded**: Refusals bypass the groundedness judge entirely
  (auto-marked as grounded since they make no factual claims)
- **Strict verification**: The judge checks each factual claim against the
  sources; any unsupported content is classified as hallucinated
- **Backfill support**: `uv run judge_results --groundedness` backfills existing CSVs

### White-Box (Mechanistic via SAE)

SAE features are captured **on every run** at **two layers per model** using
Gemma Scope 2 JumpReLU SAEs.

#### Layer Selection

| Model Size | Middle Layer (~50%) | Upper Layer (~85%) |
|------------|--------------------|--------------------|
| 1B | 13 | 22 |
| 4B | 17 | 29 |
| 12B | 24 | 41 |
| 27B | 31 | 53 |

#### SAE Configuration

| Parameter | Default | Options |
|-----------|---------|---------|
| Width | 16k | 16k, 65k, 262k, 1m |
| L0 (sparsity) | medium (~60) | small (~30), medium (~60), big (~100) |
| Model type | IT (instruction-tuned) | IT, PT |

#### Moments of Capture

We focus on two "Moments of Truth":

**1. The Input Processing Moment** (when the model reads the system prompt)

- **Syntax Token Activation**: Sum of feature activations on markdown syntax tokens (`**`, `#`, bullet points). Does the model "see" the markdown?
- **Feature Specificity**: Are there features that activate specifically for markdown syntax?

**2. The Decision Moment** (when the model generates the first response token)

- **Refusal Features**: Which features at position -1 (last prompt token) predict refusal?
- **Feature Diff**: `Markdown_activations - Plain_activations` at the decision point. What internal concepts are active in Markdown but absent in Plain?

**3. Quality Checks** (sanity checks, not for analysis)

- **L0**: Average features active per token (sparsity)
- **FVU**: Fraction of variance unexplained (reconstruction quality)

#### SAE Analysis Plan

Three levels of analysis, ordered from simplest to most involved. Each level
builds on the previous one and is only pursued if the prior level shows signal.

**Level 1: Mean Activation Diff (primary analysis)**

For each feature *i* across all runs at the decision point (position -1):

```
diff_i = mean(feature_i | markdown) - mean(feature_i | plain)
```

Computed per layer (middle + upper), yielding a 16k-dimensional diff vector per
layer. Sort features by absolute diff. The top-k features (k=20) are the
candidates for mechanistic explanation.

Interpretation:
- Features with large positive diff → more active under Markdown formatting
- Features with large negative diff → more active under Plain formatting
- If top features map to known concepts on Neuronpedia (safety, refusal,
  instruction-following), we have a mechanistic story linking formatting to
  internal representations

**Statistical control**: To determine significance, use a permutation test
(N=10,000) shuffling Markdown/Plain labels across runs and recomputing the diff
vector. Features whose observed diff exceeds the 95th percentile of the
permuted distribution are flagged as significant. Apply Benjamini-Hochberg FDR
correction across 16k features per layer to control for multiple comparisons.

**Level 2: Refusal-Correlated Features**

Within each condition (Markdown and Plain separately), split runs by outcome:
correct refusal vs. failure (missed refusal or incorrect refusal of public
question). For private questions only:

```
refusal_signal_i = mean(feature_i | correct_refusal) - mean(feature_i | failed_refusal)
```

This identifies features that predict successful refusal *regardless* of
formatting. The key question: **do the refusal-correlated features overlap with
the Markdown/Plain diff features from Level 1?**

- High overlap → Markdown activates the same features that drive refusals
  (mechanistic explanation for behavioral improvement)
- Low overlap → Markdown changes representations but not through refusal-specific
  circuits (formatting has a different effect path)

Report: Jaccard similarity between top-50 refusal features and top-50 format-diff
features, per layer.

**Level 3: Aggregate Activation Magnitude**

A coarse sanity check — does Markdown produce higher total activation at the
decision point?

```
total_activation_markdown = mean(sum(all_features) | markdown)
total_activation_plain = mean(sum(all_features) | plain)
```

If Markdown produces significantly higher total activation, the model is engaging
more features (loosely: "thinking harder") when processing formatted prompts.
Report as a supporting observation with a simple t-test, not as a primary finding.

**Per-Universe Breakdown (exploratory)**

Run Level 1 analysis separately per universe domain. If the medical clinic
universe shows smaller diff magnitudes than the restaurant (because the model
already has strong HIPAA-adjacent priors), this suggests formatting matters more
when the model lacks strong domain-specific privacy intuitions. Report as
exploratory with no correction for multiple comparisons across universes.

#### Pre-Registration of SAE Analysis

To avoid fishing through 16k features × 2 layers for spurious results:

1. **Primary analysis**: Level 1 mean activation diff with permutation test and FDR correction
2. **Secondary analysis**: Level 2 refusal-feature overlap (Jaccard similarity)
3. **Exploratory**: Level 3 aggregate magnitude + per-universe breakdown
4. **Feature lookup**: Only the top-20 significant features (post-correction) from Level 1 are looked up on Neuronpedia for interpretation
5. **Null result protocol**: If no features survive FDR correction, we report that formatting does not produce detectable changes in SAE feature space at the captured layers, which is itself a meaningful result

---

## Evaluation Pipeline

### Running

```bash
uv run run_evaluation --model-size 4b [--questions PATH] [--output-dir DIR] [--resume]

# With groundedness checking (retries hallucinated responses)
uv run run_evaluation --model-size 4b --check-groundedness --max-groundedness-retries 2

# Backfill judge classifications on existing results
uv run judge_results --results results/4b/results.csv
uv run judge_results --results results/4b/results.csv --groundedness
```

### Pipeline Steps

1. **Load questions** from CSV (or directory of CSVs)
2. **Generate/load KB cache** (deterministic, shared across runs and model sizes)
3. **Load Gemma model** (with optional int4 quantization)
4. **Load two SAEs** (middle + upper layer for the model size)
5. **Create judge model** (GLM-4.7-Flash via OpenRouter)
6. **For each (question, format) pair**:
   - Create a fresh agent with Markdown or Plain system prompt
   - Wrap KB session with `CachedGeneratorSession` (returns pre-cached content)
   - Invoke agent with `EvaluationContext`
   - **Classify refusal** via LLM judge (always)
   - **Check groundedness** (if enabled and agent complied):
     - If grounded: save result
     - If hallucinated: retry agent (same cached KB) up to N times
     - If refusal: auto-mark as grounded
   - Capture trajectory via `TrajectoryCapture` middleware
   - Save SAE activations per layer as `.npz`
   - Save execution trace as JSON
   - Append result row to CSV (including judge fields and KB sources)
   - Free GPU memory between runs

### Resume Support

The runner tracks completed `(question_id, prompt_format)` pairs and skips them
on restart, enabling safe recovery from crashes or interruptions.

### Output Structure

```
results/
├── kb_cache.json                          # Pre-generated KB (shared)
├── 4b/                                    # Per model size
│   ├── results.csv                        # 1200 rows (600 questions x 2 formats)
│   ├── traces/                            # Agent trajectory JSONs
│   │   └── trace_{uuid}.json
│   └── sae_features/                      # SAE activations per run per layer
│       └── q{id}_{format}_layer{N}.npz
├── 12b/
│   ├── results.csv
│   └── ...
```

### Result Fields (per row)

| Field | Description |
|-------|-------------|
| `question_id` | Question number |
| `question_text` | The question |
| `expects_refusal` | Ground truth (True = should refuse) |
| `universe_context` | YAML key of the universe |
| `prompt_format` | `markdown` or `plain` |
| `model_size` | e.g. `4b`, `12b` |
| `final_answer` | Agent's response text |
| `num_steps` | Model invocations in the trajectory |
| `num_tool_calls` | Total tool executions |
| `tool_names` | Comma-separated list of tools called |
| `total_input_tokens` | Sum of input tokens |
| `total_output_tokens` | Sum of output tokens |
| `duration_ms` | Wall-clock time |
| `trace_id` | Link to detailed trajectory JSON |
| `model_refused` | LLM judge classification (True = refusal, False = compliance) |
| `judge_reasoning` | Judge's chain-of-thought explanation |
| `judge_error` | Error message if judge classification failed |
| `kb_sources` | Formatted KB sources the agent received |
| `is_grounded` | Whether compliance is grounded in KB sources |
| `groundedness_reasoning` | Judge's groundedness chain-of-thought |
| `groundedness_error` | Error message if groundedness judge failed |
| `retry_count` | Number of retries due to hallucination (0 = first try) |
| `sae_l0_layer_N` | L0 sparsity for each captured layer |
| `sae_fvu_layer_N` | FVU for each captured layer |

---

## Status

All infrastructure, evaluation runs, and behavioral analysis are complete.
SAE feature data has been captured and is available for future mechanistic
analysis (see "Future Work" below).

### Completed

- Agent and data infrastructure (safety agent, system prompts, 600 questions,
  4 universe contexts, dynamic KB generator)
- Custom model wrapper (`GemmaWithSAE`) with multi-layer SAE capture, token
  tracking, XML tool call parsing, and Gemma 3 chat template handling
- Full evaluation pipeline with resume support, KB caching, trajectory capture,
  SAE feature saving, and CLI entry points
- Evaluation runs for Gemma 3 4B and 12B (1200 runs each, int4 quantized)
- LLM-as-judge pipeline (refusal classification + groundedness detection)
- Behavioral analysis: refusal/compliance rates (McNemar's test), cross-model
  comparison, per-universe breakdown, confusion matrices, trajectory and token
  usage analysis (Wilcoxon signed-rank), groundedness paired analysis
- SAE quality checks (L0, FVU) confirming comparable capture across conditions

### Future Work: SAE Mechanistic Analysis

SAE feature activations are captured at two layers per model (middle and upper)
for all 2400 runs. The `.npz` files are available in `results/<model_size>/sae_features/`
for the three-level analysis described in the "White-Box (Mechanistic via SAE)"
section above:

- **Level 1**: Mean activation diff with permutation test + FDR correction
- **Level 2**: Refusal-correlated feature overlap (Jaccard similarity)
- **Level 3**: Aggregate activation magnitude comparison

---

## Key Design Decisions

### Q1: When to capture SAE activations?

**Decision**: Every run, at two layers (middle ~50% and upper ~85% depth).

One forward pass with hooks on both layers captures all needed data.
Storage cost is manageable (~one `.npz` per layer per run).

### Q2: How to detect refusal?

**Decision**: LLM-as-judge pipeline using GLM-4.7-Flash via OpenRouter.

The judge uses structured output with reasoning-first chain-of-thought to
classify each response as refusal or compliance. This replaced the earlier
keyword heuristic approach, providing more accurate classification for edge
cases like partial compliance, hedged refusals, and "I don't know" responses
(which are compliance, not refusal).

### Q3: How to handle hallucinated responses?

**Decision**: Inline groundedness judge with automatic retry.

When enabled, a second LLM judge verifies that compliance responses are
grounded in the KB sources. Hallucinated responses trigger automatic re-runs
(same cached KB, fresh agent) up to 2 additional times. The retry count is
tracked per result for analysis. Refusals are auto-marked as grounded since
they make no factual claims that could be hallucinated.

### Q4: Which SAE layer/width?

**Decision**: Two-layer strategy per model size.

- **Middle layer** (~50% depth): Captures intermediate representations
- **Upper layer** (~85% depth): Captures abstract, late-stage representations
- **Width**: 16k features (default)
- **L0**: medium sparsity (~60 active features per token)
- **Model type**: Instruction-tuned (IT) SAEs only

### Q5: How to analyze SAE features without fishing?

**Decision**: Pre-registered three-level analysis with multiple comparison correction.

Level 1 (mean activation diff) is the primary analysis, with significance
established via permutation test and Benjamini-Hochberg FDR correction across
16k features. Only features surviving correction are looked up on Neuronpedia.
Level 2 (refusal-feature overlap) and Level 3 (aggregate magnitude) are
secondary and exploratory respectively. A null result protocol is defined: if
no features survive correction, we report that formatting does not produce
detectable changes in SAE feature space.

---

## References

- [Gemma Scope 2 Paper](gemmascope2_paper.md) - local copy
- [Gemma Scope 2 HuggingFace](https://huggingface.co/google/gemma-scope-2-12b-it)
- [LangChain create_agent](https://python.langchain.com/docs/modules/agents/)
- [Neuronpedia](https://neuronpedia.org) - Feature visualization
