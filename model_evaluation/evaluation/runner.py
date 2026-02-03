"""Core evaluation runner for safety prompt comparison."""

import csv
import time
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from langchain_core.messages import HumanMessage

from model_evaluation.evaluation.kb_cache import (
    CachedGeneratorSession,
    generate_kb_cache,
    load_kb_cache,
)
from model_evaluation.evaluation.schemas import (
    DISPLAY_NAME_TO_YAML_KEY,
    QuestionRow,
    RunResult,
)
from model_evaluation.main_agent.gemma_scope_sae import (
    MultiLayerSAEFeatureResult,
    get_evaluation_layers,
    load_gemma_scope_sae,
)
from model_evaluation.main_agent.gemma_wrapper import GemmaWithSAE
from model_evaluation.main_agent.kb_generator.schemas import GeneratorOutput
from model_evaluation.main_agent.rag_agent import create_safety_agent
from model_evaluation.main_agent.tools import EvaluationContext
from model_evaluation.tracing import AgentTrace, TrajectoryCapture
from model_evaluation.tracing.storage import save_trace


def load_questions(*, questions_path: Path) -> list[QuestionRow]:
    """Load and parse questions from CSV.

    Args:
        questions_path: Path to the questions CSV file.

    Returns:
        List of parsed QuestionRow objects.
    """
    rows: list[QuestionRow] = []

    with questions_path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for csv_row in reader:
            doc_origin = csv_row["Document of origin"].strip()
            rows.append(
                QuestionRow(
                    number=int(csv_row["Number"]),
                    question=csv_row["Question"].strip(),
                    document_of_origin=doc_origin,
                    is_malicious=csv_row["Malicious question"].strip() == "Yes",
                    universe_context_key=DISPLAY_NAME_TO_YAML_KEY.get(doc_origin),
                ),
            )

    return rows


def save_sae_features(
    *,
    activations: MultiLayerSAEFeatureResult,
    question_id: int,
    prompt_format: str,
    output_dir: Path,
) -> None:
    """Save SAE features as .npz files, one per layer.

    Args:
        activations: Multi-layer SAE feature results.
        question_id: Question number for filename.
        prompt_format: 'markdown' or 'plain' for filename.
        output_dir: Directory to write .npz files.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    for layer_idx, result in activations.layer_results.items():
        filename = f"q{question_id}_{prompt_format}_layer{layer_idx}.npz"
        np.savez_compressed(
            output_dir / filename,
            top_features=result.top_features.cpu().numpy(),
            top_activations=result.top_activations.cpu().numpy(),
            tokens=np.array(result.tokens, dtype=object),
            prompt_len=np.array(result.prompt_len),
            l0=np.array(result.l0),
            fvu=np.array(result.fvu),
        )


def flatten_run_result(
    *,
    result: RunResult,
    layers: tuple[int, int],
) -> dict:
    """Flatten a RunResult into a dict suitable for CSV export.

    Expands sae_l0_by_layer and sae_fvu_by_layer into individual columns.

    Args:
        result: The run result to flatten.
        layers: (middle_layer, upper_layer) for column naming.

    Returns:
        Flat dictionary with all fields.
    """
    data = result.model_dump(exclude={"sae_l0_by_layer", "sae_fvu_by_layer"})
    data["tool_names"] = ",".join(result.tool_names)

    for layer in layers:
        data[f"sae_l0_layer_{layer}"] = result.sae_l0_by_layer.get(layer, 0.0)
        data[f"sae_fvu_layer_{layer}"] = result.sae_fvu_by_layer.get(layer, 0.0)

    return data


def load_completed_runs(*, results_path: Path) -> set[tuple[int, str]]:
    """Load already-completed (question_id, prompt_format) pairs for resume.

    Args:
        results_path: Path to the results CSV.

    Returns:
        Set of (question_id, prompt_format) tuples already completed.
    """
    if not results_path.exists():
        return set()

    df = pd.read_csv(results_path)
    return {(int(row["question_id"]), row["prompt_format"]) for _, row in df.iterrows()}


def run_single_question(
    *,
    model: GemmaWithSAE,
    question: QuestionRow,
    prompt_format: Literal["markdown", "plain"],
    kb_output: GeneratorOutput,
    model_size: str,
) -> tuple[RunResult, AgentTrace | None]:
    """Run a single (question, prompt_format) evaluation.

    Creates a fresh agent, invokes it with cached KB, and collects results.

    Args:
        model: The GemmaWithSAE model instance.
        question: The question to evaluate.
        prompt_format: 'markdown' or 'plain'.
        kb_output: Pre-generated KB content for this question.
        model_size: Model size string for metadata.

    Returns:
        Tuple of (RunResult, AgentTrace or None).
    """
    tracer = TrajectoryCapture()
    agent = create_safety_agent(
        model,
        use_markdown_rules=(prompt_format == "markdown"),
        middleware=[tracer],
    )

    cached_session = CachedGeneratorSession(cached_output=kb_output)
    eval_context = EvaluationContext(
        include_private_info=True,
        generator_session=cached_session,
        universe_context=question.universe_context_key,
    )

    start_ms = time.perf_counter() * 1000

    result = agent.invoke(
        {"messages": [HumanMessage(content=question.question)]},
        context=eval_context,
    )

    duration_ms = time.perf_counter() * 1000 - start_ms

    trace = tracer.last_trace
    final_answer = ""
    if result and "messages" in result:
        messages = result["messages"]
        if messages:
            final_answer = str(messages[-1].content)

    sae_l0: dict[int, float] = {}
    sae_fvu: dict[int, float] = {}
    multi_acts = model.last_multi_layer_activations
    if multi_acts:
        for layer_idx, layer_result in multi_acts.layer_results.items():
            sae_l0[layer_idx] = layer_result.l0
            sae_fvu[layer_idx] = layer_result.fvu

    tool_names: list[str] = []
    total_tool_calls = 0
    if trace:
        for step in trace.steps:
            for tool_exec in step.tool_executions:
                tool_names.append(tool_exec.tool_name)
                total_tool_calls += 1

    run_result = RunResult(
        question_id=question.number,
        question_text=question.question,
        expects_refusal=question.is_malicious,
        universe_context=question.universe_context_key,
        prompt_format=prompt_format,
        model_size=model_size,
        final_answer=final_answer,
        num_steps=trace.total_steps if trace else 0,
        num_tool_calls=total_tool_calls,
        tool_names=tool_names,
        total_input_tokens=trace.total_input_tokens if trace else 0,
        total_output_tokens=trace.total_output_tokens if trace else 0,
        duration_ms=duration_ms,
        trace_id=trace.trace_id if trace else "",
        sae_l0_by_layer=sae_l0,
        sae_fvu_by_layer=sae_fvu,
    )
    return run_result, trace


def run_evaluation(
    *,
    model_size: str,
    questions_path: Path,
    output_dir: Path,
    resume: bool = False,
) -> None:
    """Run the full evaluation pipeline.

    Args:
        model_size: Gemma model size (1b, 4b, 12b, 27b).
        questions_path: Path to questions CSV.
        output_dir: Directory for all output files.
        resume: If True, skip already-completed runs.
    """
    from model_evaluation.config import Settings
    from model_evaluation.main_agent.gemma_model_loader import (
        GemmaModelConfig,
        load_gemma_model,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    traces_dir = output_dir / "traces"
    traces_dir.mkdir(exist_ok=True)
    sae_dir = output_dir / "sae_features"
    sae_dir.mkdir(exist_ok=True)
    results_path = output_dir / "results.csv"
    kb_cache_path = output_dir / "kb_cache.json"

    # Load questions
    print("Loading questions...")
    questions = load_questions(questions_path=questions_path)
    print(f"  Loaded {len(questions)} questions")

    # Load or generate KB cache
    if resume and kb_cache_path.exists():
        print("Loading KB cache from disk...")
        kb_cache = load_kb_cache(cache_path=kb_cache_path)
        print(f"  Loaded {len(kb_cache)} cached entries")
    else:
        print("Generating KB cache...")
        kb_cache = generate_kb_cache(
            questions=questions,
            output_path=kb_cache_path,
        )
        print(f"  Generated {len(kb_cache)} entries")

    # Load completed runs for resume
    completed = load_completed_runs(results_path=results_path) if resume else set()
    if completed:
        print(f"  Resuming: {len(completed)} runs already completed")

    # Load model
    settings = Settings()  # type: ignore[call-arg]
    print(f"Loading Gemma {model_size} model...")
    config = GemmaModelConfig(
        model_id=f"google/gemma-3-{model_size}-it",
        max_context_length=settings.gemma_max_context_length,
    )
    model_hf, tokenizer = load_gemma_model(config=config, token=settings.hf_token)
    device = str(next(model_hf.parameters()).device)

    # Load SAEs for two layers
    middle_layer, upper_layer = get_evaluation_layers(model_size=model_size)
    layers = (middle_layer, upper_layer)
    print(f"Loading SAEs for layers {middle_layer} and {upper_layer}...")

    sae_primary, config_primary = load_gemma_scope_sae(
        model_size=model_size,
        model_type="it",
        layer=upper_layer,
        width=settings.sae_width,
        l0_size=settings.sae_l0_size,
        device=device,
    )
    sae_secondary, config_secondary = load_gemma_scope_sae(
        model_size=model_size,
        model_type="it",
        layer=middle_layer,
        width=settings.sae_width,
        l0_size=settings.sae_l0_size,
        device=device,
    )

    gemma = GemmaWithSAE(
        model=model_hf,
        tokenizer=tokenizer,
        sae=sae_primary,
        sae_config=config_primary,
        additional_saes={middle_layer: (sae_secondary, config_secondary)},
        max_tokens=settings.max_new_tokens,
    )

    # Run evaluation
    total_runs = len(questions) * 2
    run_count = 0
    formats: list[Literal["markdown", "plain"]] = ["markdown", "plain"]

    print(f"\nStarting evaluation: {len(questions)} questions x 2 formats = {total_runs} runs")
    print("=" * 60)

    for question in questions:
        for prompt_format in formats:
            run_key = (question.number, prompt_format)
            if run_key in completed:
                run_count += 1
                continue

            run_count += 1
            print(
                f"\n[{run_count}/{total_runs}] "
                f"Q{question.number} ({prompt_format}) "
                f"{'MALICIOUS' if question.is_malicious else 'benign'}"
            )

            kb_output = kb_cache.get(question.number)
            if kb_output is None:
                print(f"  WARNING: No KB cache for question {question.number}, skipping")
                continue

            result, trace = run_single_question(
                model=gemma,
                question=question,
                prompt_format=prompt_format,
                kb_output=kb_output,
                model_size=model_size,
            )

            # Save SAE features
            multi_acts = gemma.last_multi_layer_activations
            if multi_acts:
                save_sae_features(
                    activations=multi_acts,
                    question_id=question.number,
                    prompt_format=prompt_format,
                    output_dir=sae_dir,
                )

            # Save trace
            if trace:
                save_trace(trace=trace, output_dir=traces_dir)

            # Append result to CSV
            flat = flatten_run_result(result=result, layers=layers)
            write_header = not results_path.exists()
            with results_path.open("a", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=flat.keys())
                if write_header:
                    writer.writeheader()
                writer.writerow(flat)

            print(
                f"  Steps: {result.num_steps} | "
                f"Tools: {result.num_tool_calls} | "
                f"Tokens: {result.total_input_tokens}in/{result.total_output_tokens}out | "
                f"Duration: {result.duration_ms:.0f}ms"
            )
            print(f"  Answer: {result.final_answer[:100]}...")

    print("\n" + "=" * 60)
    print("Evaluation complete!")
    print(f"  Results: {results_path}")
    print(f"  SAE features: {sae_dir}")
    print(f"  KB cache: {kb_cache_path}")
