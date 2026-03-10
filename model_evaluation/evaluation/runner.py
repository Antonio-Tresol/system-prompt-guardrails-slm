"""Core evaluation runner for safety prompt comparison."""

import csv
import gc
import time
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import torch
from langchain_core.messages import HumanMessage
from langgraph.errors import GraphRecursionError

from model_evaluation.evaluation.judge import (
    classify_groundedness,
    classify_refusal,
    create_judge_model,
)
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
from utils.logging import logger


def _load_single_csv(*, csv_path: Path, start_number: int) -> list[QuestionRow]:
    """Load questions from a single CSV file with sequential numbering.

    Args:
        csv_path: Path to the CSV file.
        start_number: Starting number for sequential question IDs.

    Returns:
        List of parsed QuestionRow objects.
    """
    rows: list[QuestionRow] = []

    with csv_path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row_idx, csv_row in enumerate(reader, start=start_number):
            universe_name = csv_row["Universe"].strip()
            universe_key = DISPLAY_NAME_TO_YAML_KEY[universe_name]
            rows.append(
                QuestionRow(
                    number=row_idx,
                    question=csv_row["Question"].strip(),
                    universe=universe_name,
                    is_malicious=csv_row["Malicious"].strip() == "Yes",
                    universe_context_key=universe_key,
                ),
            )

    return rows


def load_questions(*, questions_path: Path) -> list[QuestionRow]:
    """Load and parse questions from a CSV file or a directory of CSVs.

    When given a directory, all CSV files are concatenated with globally
    sequential numbering (1..N) to ensure unique KB cache keys.

    Args:
        questions_path: Path to a single CSV file or a directory containing CSVs.

    Returns:
        List of parsed QuestionRow objects with unique sequential numbers.
    """
    if questions_path.is_dir():
        csv_files = sorted(questions_path.glob("*.csv"))
        if not csv_files:
            msg = f"No CSV files found in {questions_path}"
            raise FileNotFoundError(msg)

        all_rows: list[QuestionRow] = []
        next_number = 1
        for csv_file in csv_files:
            rows = _load_single_csv(csv_path=csv_file, start_number=next_number)
            all_rows.extend(rows)
            next_number += len(rows)

        return all_rows

    return _load_single_csv(csv_path=questions_path, start_number=1)


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


def remove_hallucinated_rows(*, results_path: Path) -> int:
    """Remove rows with is_grounded=False from the results CSV.

    This allows re-running hallucinated questions with ``--resume``.

    Args:
        results_path: Path to the results CSV file.

    Returns:
        Number of rows removed.
    """
    if not results_path.exists():
        logger.info("No results file at {}, nothing to clean", results_path)
        return 0

    df = pd.read_csv(results_path)
    before = len(df)
    hallucinated = df[df["is_grounded"] == False]  # noqa: E712

    if hallucinated.empty:
        logger.info("No hallucinated rows found in {}", results_path)
        return 0

    for _, row in hallucinated.iterrows():
        logger.info(
            "Removing hallucinated row: Q{} ({})",
            int(row["question_id"]),
            row["prompt_format"],
        )

    df = df[df["is_grounded"] != False]  # noqa: E712
    df.to_csv(results_path, index=False)
    removed = before - len(df)
    logger.info("Removed {} hallucinated rows, {} remaining", removed, len(df))
    return removed


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
    max_recursion_retries = 2

    for recursion_attempt in range(max_recursion_retries + 1):
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

        try:
            result = agent.invoke(
                {"messages": [HumanMessage(content=question.question)]},
                context=eval_context,
                config={"recursion_limit": 10},
            )
            break
        except GraphRecursionError:
            if recursion_attempt < max_recursion_retries:
                logger.warning(
                    "Agent hit recursion limit on Q{} ({}), retrying ({}/{})...",
                    question.number,
                    prompt_format,
                    recursion_attempt + 1,
                    max_recursion_retries,
                )
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue

            logger.error(
                "Agent hit recursion limit {} times for Q{} ({}), skipping",
                max_recursion_retries + 1,
                question.number,
                prompt_format,
            )
            raise

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
    enable_groundedness: bool = False,
    max_groundedness_retries: int = 2,
) -> None:
    """Run the full evaluation pipeline.

    Args:
        model_size: Gemma model size (1b, 4b, 12b, 27b).
        questions_path: Path to questions CSV.
        output_dir: Directory for all output files.
        resume: If True, skip already-completed runs.
        enable_groundedness: If True, check groundedness and retry on hallucination.
        max_groundedness_retries: Maximum retry attempts for hallucinated responses.
    """
    from model_evaluation.config import Settings
    from model_evaluation.main_agent.gemma_model_loader import (
        GemmaModelConfig,
        load_gemma_model,
    )

    # KB cache lives at the base output_dir (shared across model sizes)
    output_dir.mkdir(parents=True, exist_ok=True)
    kb_cache_path = output_dir / "kb_cache.json"

    # Model-specific outputs go under output_dir/<model_size>/
    model_output_dir = output_dir / model_size
    model_output_dir.mkdir(parents=True, exist_ok=True)
    traces_dir = model_output_dir / "traces"
    traces_dir.mkdir(exist_ok=True)
    sae_dir = model_output_dir / "sae_features"
    sae_dir.mkdir(exist_ok=True)
    results_path = model_output_dir / "results.csv"

    # Load questions
    logger.info("Loading questions from {}", questions_path)
    questions = load_questions(questions_path=questions_path)
    logger.info("Loaded {} questions", len(questions))

    # Load existing KB cache and generate any missing entries
    existing_cache: dict[int, GeneratorOutput] | None = None
    if kb_cache_path.exists():
        logger.info("Loading existing KB cache from {}", kb_cache_path)
        existing_cache = load_kb_cache(cache_path=kb_cache_path)
        logger.info("Loaded {} cached KB entries", len(existing_cache))

    missing = len(questions) - len(existing_cache or {})
    if missing > 0:
        logger.info("Generating {} missing KB entries...", missing)
        kb_cache = generate_kb_cache(
            questions=questions,
            output_path=kb_cache_path,
            existing_cache=existing_cache,
        )
    else:
        logger.info("KB cache is complete, no generation needed")
        kb_cache = existing_cache or {}

    # Load completed runs for resume
    completed = load_completed_runs(results_path=results_path) if resume else set()
    if completed:
        logger.info("Resuming: {} runs already completed", len(completed))

    # Create judge model for refusal classification
    settings = Settings()  # type: ignore[call-arg]
    logger.info("Creating judge model ({})", settings.judge_model)
    judge_model = create_judge_model(settings=settings)

    # Load model
    logger.info("Loading Gemma {} model...", model_size)
    config = GemmaModelConfig(
        model_id=f"google/gemma-3-{model_size}-it",
        size=model_size,
        max_context_length=settings.gemma_max_context_length,
    )
    model_hf, tokenizer = load_gemma_model(config=config, token=settings.hf_token)
    device = str(next(model_hf.parameters()).device)

    # Load SAEs for two layers
    middle_layer, upper_layer = get_evaluation_layers(model_size=model_size)
    layers = (middle_layer, upper_layer)
    logger.info("Loading SAEs for layers {} and {}...", middle_layer, upper_layer)

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

    logger.info(
        "Starting evaluation: {} questions x 2 formats = {} runs",
        len(questions),
        total_runs,
    )
    if enable_groundedness:
        logger.info(
            "Groundedness checking enabled (max retries: {})",
            max_groundedness_retries,
        )

    for question in questions:
        for prompt_format in formats:
            run_key = (question.number, prompt_format)
            if run_key in completed:
                run_count += 1
                continue

            run_count += 1
            label = "MALICIOUS" if question.is_malicious else "benign"
            logger.info(
                "[{}/{}] Q{} ({}) {}",
                run_count,
                total_runs,
                question.number,
                prompt_format,
                label,
            )

            kb_output = kb_cache.get(question.number)
            if kb_output is None:
                logger.warning("No KB cache for question {}, skipping", question.number)
                continue

            kb_sources = kb_output.format_sources()
            max_attempts = (max_groundedness_retries + 1) if enable_groundedness else 1
            result: RunResult | None = None
            trace: AgentTrace | None = None
            retry_count = 0

            try:
                for attempt in range(1, max_attempts + 1):
                    result, trace = run_single_question(
                        model=gemma,
                        question=question,
                        prompt_format=prompt_format,
                        kb_output=kb_output,
                        model_size=model_size,
                    )
                    result.kb_sources = kb_sources

                    # Classify refusal using LLM judge
                    judge_result = classify_refusal(
                        judge_model=judge_model,
                        question=question.question,
                        answer=result.final_answer,
                    )
                    result.model_refused = judge_result.model_refused
                    result.judge_reasoning = judge_result.judge_reasoning
                    result.judge_error = judge_result.judge_error

                    # Groundedness check (only if enabled and agent complied)
                    if enable_groundedness and result.model_refused is False:
                        groundedness = classify_groundedness(
                            judge_model=judge_model,
                            question=question.question,
                            kb_sources=kb_sources,
                            answer=result.final_answer,
                        )
                        result.is_grounded = groundedness.is_grounded
                        result.groundedness_reasoning = groundedness.groundedness_reasoning
                        result.groundedness_error = groundedness.groundedness_error

                        if groundedness.is_grounded is True:
                            break

                        if groundedness.is_grounded is None:
                            logger.warning(
                                "Groundedness judge error, saving result as-is: {}",
                                groundedness.groundedness_error,
                            )
                            break

                        if attempt < max_attempts:
                            retry_count = attempt
                            logger.warning(
                                "Hallucination detected (attempt {}/{}), retrying...",
                                attempt,
                                max_attempts,
                            )
                            gc.collect()
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                        else:
                            retry_count = attempt - 1
                            logger.error(
                                "Hallucination persists after {} retries, saving anyway",
                                max_groundedness_retries,
                            )
                    elif enable_groundedness and result.model_refused is True:
                        result.is_grounded = True
                        result.groundedness_reasoning = "Refusals are always grounded."
                        break
                    else:
                        break
            except GraphRecursionError:
                logger.error(
                    "Skipping Q{} ({}) — agent hit recursion limit on all retries",
                    question.number,
                    prompt_format,
                )
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue

            if result is None:
                continue

            result.retry_count = retry_count

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

            judge_label = (
                "REFUSED"
                if result.model_refused is True
                else "COMPLIED"
                if result.model_refused is False
                else "JUDGE_ERROR"
            )
            groundedness_label = ""
            if enable_groundedness and result.model_refused is False:
                groundedness_label = (
                    " | Grounded: YES"
                    if result.is_grounded is True
                    else " | Grounded: NO"
                    if result.is_grounded is False
                    else " | Grounded: ERROR"
                )
                if retry_count > 0:
                    groundedness_label += f" (retries: {retry_count})"
            logger.info(
                "Steps: {} | Tools: {} | Judge: {}{} | Tokens: {}in/{}out | {:.0f}ms",
                result.num_steps,
                result.num_tool_calls,
                judge_label,
                groundedness_label,
                result.total_input_tokens,
                result.total_output_tokens,
                result.duration_ms,
            )
            logger.debug("Answer: {}...", result.final_answer[:100])

            # Free GPU memory between runs
            del result, trace, multi_acts
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    logger.info("Evaluation complete!")
    logger.info("Results: {}", results_path)
    logger.info("SAE features: {}", sae_dir)
    logger.info("KB cache: {}", kb_cache_path)
