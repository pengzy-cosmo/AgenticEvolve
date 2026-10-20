"""Island: one persistent LLM session evolving one program.

The six tools are closures over the island's state. All of them are async, so
they run as tasks on the shared event loop and ``agent.cancel()`` can cancel an
in-flight evaluation. The commit tool keeps everything after the evaluation
await synchronous: archive append, MAP-Elites update, migration query, and
feedback construction happen atomically with respect to other islands.
"""

from __future__ import annotations

import asyncio
import logging
import math
import os
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import cast

from mycode import Agent, ToolExecutionResult, ToolSpec, tool
from mycode.tools import AsyncToolRunner

from agentic_evolve.archive import Archive, Behavior, CommitRecord
from agentic_evolve.config import Config
from agentic_evolve.executor import EvaluationError, Executor

logger = logging.getLogger(__name__)

DRIVER_MESSAGE = "Continue."


def resource_files(cfg: Config) -> list[tuple[str, int]]:
    """List resource paths as exposed to the model, with sizes in bytes."""

    if cfg.task.resources is None:
        return []
    root = Path(cfg.task.resources)
    files = [
        path
        for path in root.rglob("*")
        if path.is_file() and "__pycache__" not in path.parts and path.name != ".DS_Store"
    ]
    return sorted(
        (f"{root.name}/{path.relative_to(root).as_posix()}", path.stat().st_size) for path in files
    )


def _system_prompt(cfg: Config) -> str:
    """The running contract: task, feedback semantics, capabilities. It varies
    only with the task and with the analysis capability; runtime state (program,
    budget, compaction) is communicated in messages, never here."""

    max_analyses = cfg.evolution.max_analyses_per_commit
    analysis = max_analyses > 0
    me = cfg.map_elites
    files = "program.py and analysis.py" if analysis else "program.py"

    parts = [
        "You are the engine of an evolutionary program search: you improve a Python\n"
        "program against a formal evaluator.\n",
        f"Task:\n{cfg.task.description.strip()}\n",
        "Feedback:\n"
        "- The evaluator returns the metrics described in the task; the fitness is\n"
        f"  {me.fitness}, higher is better.\n"
        "- An archive keeps every commit and, per island, a MAP-Elites grid over the\n"
        f"  feature fields {', '.join(me.features)} ({me.feature_bins} bins per axis,\n"
        "  each field normalized to [0, 1]) holding the fittest commit of each cell.\n"
        "- You run one of several islands searching in parallel. Each commit returns\n"
        "  the commit's metrics, the island's best commit, its MAP-Elites, and\n"
        "  periodically migration candidates: top elites of the neighboring islands.\n",
    ]
    file_lines = ["- program.py: the evolving candidate; only this file is formally evaluated."]
    if analysis:
        file_lines.append(
            "- analysis.py: your workspace for investigating hypotheses; it may `import program`."
        )
    resources = resource_files(cfg)
    held = files
    if resources:
        name = Path(cast(str, cfg.task.resources)).name
        held = f"{files.replace(' and ', ', ')} and {name}/"
        file_lines.append(
            f"- {name}/: read-only task resources, importable from {files}:\n"
            + "\n".join(f"    {path} ({size / 1000:.1f} kB)" for path, size in resources)
        )
    parts.append(
        "Files, accessible only through your tools:\n"
        + "\n".join(file_lines)
        + f"\nEvery execution runs in a fresh temporary directory holding {held}:\n"
        "only stdout comes back, do not write outside it or use the network, and\n"
        f"state persists across executions only through {files}.\n"
    )
    tools = [
        f"- read / write / edit: operate on {files}; read also serves the task resources."
        if resources
        else f"- read / write / edit: operate on {files}."
    ]
    if analysis:
        tools.append(
            "- run_analysis: execute the current analysis.py and return its output, up to\n"
            f"  {max_analyses} runs per commit cycle."
        )
    tools.append(
        "- commit: run the formal evaluator on program.py, append the program and result\n"
        "  to the archive, and return the feedback above. Formal metrics are available\n"
        "  only through commit. A failed evaluation is recorded and consumes budget like\n"
        "  any other commit; its feedback is the exception type and line, plus the\n"
        "  message when the evaluator itself rejected the program."
    )
    tools.append(
        "- read_commit: return the full program.py of a recorded commit; write it to\n"
        "  program.py to continue from it."
    )
    parts.append("Tools:\n" + "\n".join(tools) + "\n")
    parts.append(
        "Experiment loop, repeated until the framework tells you to stop:\n"
        "1. Form a hypothesis from the current program and the feedback so far.\n"
        "2. Modify program.py accordingly.\n"
        "3. Commit the experiment and inspect its formal feedback.\n"
        "4. Use the result, the MAP-Elites, and the migration candidates to choose the\n"
        "   next idea.\n"
    )
    if analysis:
        parts.append(
            "At any point, use analysis.py and run_analysis to explore the data, check an\n"
            "idea, or diagnose a result.\n"
        )
        parts.append(
            "You decide what to investigate, how much analysis to perform, and what to\n"
            "change. When a message tells you to end the turn, comply immediately."
        )
    else:
        parts.append(
            "You decide what to investigate and what to change. When a message tells you\n"
            "to end the turn, comply immediately."
        )
    return "\n".join(parts)


STOP_INSTRUCTION = (
    "[context limit] Your context is nearly full. State your current hypothesis "
    "in one or two sentences, then end this turn immediately without calling "
    "any more tools."
)

END_INSTRUCTION = (
    "[budget exhausted] The run budget is exhausted. End this turn immediately "
    "without calling any more tools."
)

SUMMARY_INSTRUCTION = """\
Ignore the running contract; do not call tools. Your context is being reset. A
successor will continue from the current program.py and the archive state, but
without this conversation. Write a concise handoff covering exactly:
1. Approaches already tried that failed, and why each failed.
2. Facts about the problem and data established by your experiments.
3. Your current hypothesis.
4. The next step you would take.
"""


async def _stamp_timestamp(message: dict) -> None:
    """on_persist callback: stamp every session message before it is stored."""
    message.setdefault("meta", {})["timestamp"] = datetime.now(UTC).isoformat()


class Island:
    def __init__(
        self,
        island_id: int,
        config: Config,
        archive: Archive,
        executor: Executor,
        session_dir: Path,
        initial_program: str,
        initial_metrics: dict,
    ):
        self.id = island_id
        self.cfg = config
        self.archive = archive
        self.executor = executor
        self.session_dir = session_dir

        self.program = initial_program
        self.initial_metrics = initial_metrics
        self.analysis = ""
        self.resource_names = {path for path, _ in resource_files(config)}
        self.analysis_count = 0
        self.commit_count = 0
        self.request_count = 0
        self.requests_exhausted = False
        self.idle_turns = 0  # consecutive turns without a tool call
        self.behavior = Behavior()
        self.context_tokens = 0
        self.total_tokens = 0
        self.completed_context_tokens = 0
        self.compact_requested = False
        self.compact_count = 0
        self.last_migration: list[CommitRecord] = []
        self.shown_migration_ids: set[str] = set()
        self._cancel_requested = False

        self.system_prompt = _system_prompt(config)
        self.agent = self._make_agent(str(self.id))

    # ------------------------------------------------------------------
    # Session loop
    # ------------------------------------------------------------------

    async def run(self) -> None:
        logger.info("island %d: start", self.id)
        evolution = self.cfg.evolution
        prompt = self._opening_message()
        while True:
            await self._run_turn(prompt)
            prompt = DRIVER_MESSAGE

            if self.commit_count >= evolution.max_commits:
                reason = f"max_commits ({evolution.max_commits}) reached"
                break
            if (
                evolution.max_total_tokens is not None
                and self.total_tokens >= evolution.max_total_tokens
            ):
                reason = f"max_total_tokens ({evolution.max_total_tokens}) reached"
                break
            if self.requests_exhausted:
                reason = f"max_requests ({evolution.max_requests}) reached"
                break
            if self.idle_turns >= evolution.max_idle_turns:
                reason = f"agent stopped ({self.idle_turns} turns without tool calls)"
                break
            if self.compact_requested:
                summary = await self._compact()
                if (
                    evolution.max_total_tokens is not None
                    and self.total_tokens >= evolution.max_total_tokens
                ):
                    reason = f"max_total_tokens ({evolution.max_total_tokens}) reached"
                    break
                if self.requests_exhausted:
                    reason = f"max_requests ({evolution.max_requests}) reached"
                    break
                prompt = self._compact_message(summary)

        logger.info(
            "island %d: terminated (%s), commits=%d, requests=%d, total_tokens=%d",
            self.id,
            reason,
            self.commit_count,
            self.request_count,
            self.total_tokens,
        )

    async def _run_turn(self, prompt: str) -> None:
        self._cancel_requested = False
        self.idle_turns += 1
        async for event in self.agent.achat(prompt, on_persist=_stamp_timestamp):
            if event.type == "usage":
                self._on_usage(event.data.get("context_tokens"))
            elif event.type == "retry":
                self._log_retry(event.data)
            elif event.type == "cancelled":
                if not self._cancel_requested:
                    raise asyncio.CancelledError
                logger.info("island %d: turn cancelled by framework", self.id)
            elif event.type == "error":
                message = str(event.data.get("message") or "")
                raise RuntimeError(f"island {self.id}: agent error: {message}")

    def _on_usage(self, context_tokens: int | None) -> None:
        self.behavior.context_tokens = context_tokens
        self.behavior.model_calls += 1
        self.request_count += 1
        max_requests = self.cfg.evolution.max_requests
        if max_requests is not None and self.request_count >= max_requests:
            # The response that hit the limit runs its tool calls; every tool
            # result then tells the model to end the turn. A further request is
            # cancelled before its tool calls run.
            self.requests_exhausted = True
            if self.request_count > max_requests:
                self._cancel_turn(f"max_requests ({max_requests}) exceeded")
        if context_tokens is None:
            return

        self.context_tokens = context_tokens
        self.total_tokens = self.completed_context_tokens + context_tokens

        window = self.agent.context_window
        if context_tokens >= window * self.cfg.compact.compact_threshold:
            self.compact_requested = True

        evolution = self.cfg.evolution
        if self.commit_count >= evolution.max_commits:
            self._cancel_turn(f"max_commits ({evolution.max_commits}) reached")
        elif (
            evolution.max_total_tokens is not None
            and self.total_tokens >= evolution.max_total_tokens
        ):
            self._cancel_turn(f"max_total_tokens ({evolution.max_total_tokens}) reached")
        elif context_tokens >= window * self.cfg.compact.safety_threshold:
            self._cancel_turn(f"context safety threshold reached at {context_tokens} tokens")

    def _log_retry(self, data: dict) -> None:
        logger.warning(
            "island %d: model request retry %s/%s in %ss | %s | %s",
            self.id,
            data.get("attempt", "?"),
            data.get("max_attempts", "?"),
            data.get("delay_seconds", "?"),
            data.get("reason", "unknown"),
            data.get("message", ""),
        )

    def _cancel_turn(self, reason: str, level: int = logging.WARNING) -> None:
        if self._cancel_requested:
            return
        logger.log(level, "island %d: cancelling turn: %s", self.id, reason)
        self._cancel_requested = True
        self.agent.cancel()

    # ------------------------------------------------------------------
    # Compaction: our own context compression. The old session is summarized
    # by a tool-less call, then a fresh session continues from the summary and
    # the current state.
    # ------------------------------------------------------------------

    async def _compact(self) -> str:
        logger.info(
            "island %d: compaction %d (context at %d tokens)",
            self.id,
            self.compact_count + 1,
            self.context_tokens,
        )
        summary = await self._summarize()
        self.completed_context_tokens += self.context_tokens
        self.total_tokens = self.completed_context_tokens
        self.compact_count += 1
        self.compact_requested = False
        self.context_tokens = 0
        limit = self.cfg.evolution.max_total_tokens
        if limit is None or self.total_tokens < limit:
            self.agent = self._make_agent(f"{self.id}-c{self.compact_count}")
        return summary

    async def _summarize(self) -> str:
        """Summary call: a temp agent resumes the old session with no tools."""

        summarizer = self._make_agent(self.agent.session_id, summary_mode=True)
        parts: list[str] = []
        async for event in summarizer.achat(SUMMARY_INSTRUCTION, on_persist=_stamp_timestamp):
            if event.type == "text":
                parts.append(str(event.data.get("delta") or ""))
            elif event.type == "usage":
                self.request_count += 1  # the summary is a model request too
                max_requests = self.cfg.evolution.max_requests
                if max_requests is not None and self.request_count >= max_requests:
                    self.requests_exhausted = True
                context_tokens = event.data.get("context_tokens")
                if context_tokens is not None:
                    self.context_tokens = context_tokens
                    self.total_tokens = self.completed_context_tokens + context_tokens
            elif event.type == "retry":
                self._log_retry(event.data)
            elif event.type == "error":
                raise RuntimeError(
                    f"island {self.id}: summary call failed: {event.data.get('message')}"
                )
        return "".join(parts)

    # ------------------------------------------------------------------
    # Agent and messages
    # ------------------------------------------------------------------

    def _make_agent(self, session_id: str, summary_mode: bool = False) -> Agent:
        llm = self.cfg.llm
        api_key = os.environ[llm.api_key_env] if llm.api_key_env else None
        return Agent(
            model=llm.model,
            provider=llm.provider,
            api_key=api_key,
            api_base=llm.base_url,
            session_dir=self.session_dir,
            session_id=session_id,
            system=self.system_prompt,
            tools=[] if summary_mode else self._build_tools(),
            context_window=llm.context_window,
            max_tokens=llm.max_tokens,
            compact_threshold=0,
            reasoning_effort=llm.reasoning_effort,
            max_retries=8,  # Allow more retries for transient connection failures.
        )

    def _opening_message(self) -> str:
        parts = [
            f"Current program.py:\n```python\n{self.program}\n```",
            f"Initial evaluation:\n{self._format_metrics(self.initial_metrics)}",
            self._budget_line(),
            "Begin.",
        ]
        return "\n\n".join(parts)

    def _compact_message(self, summary: str) -> str:
        parts = [
            "Your context was compacted; you are continuing this run. Current state follows.",
            f"Current program.py:\n```python\n{self.program}\n```",
        ]
        last = self.archive.records.get(f"{self.id}-{self.commit_count}")
        if last is not None:
            if last.metrics is None:
                parts.append(f"Last commit:\n{last.id} | evaluation failed | {last.description}")
                parts.append(f"Evaluation error:\n{last.error}")
            else:
                parts.append(f"Last commit:\n{self._format_record(last)}")
        best = self.archive.island_best[self.id]
        if best is not None:
            parts.append("Island best:\n" + self._format_record(self.archive.records[best]))
        elites = self.archive.island_elites(self.id)
        if elites:
            parts.append("Island MAP-Elites:\n" + "\n".join(self._format_record(r) for r in elites))
        if self.last_migration:
            parts.append(
                "Migration candidates:\n"
                + "\n".join(self._format_record(r, with_island=True) for r in self.last_migration)
            )
        parts.append(self._budget_line())
        parts.append(f"Summary from your previous session:\n{summary}")
        parts.append("Continue.")
        return "\n\n".join(parts)

    @staticmethod
    def _format_metrics(metrics: dict) -> str:
        # Verbatim values: rounding here would lose information the model acts on.
        return " ".join(f"{key}={value}" for key, value in metrics.items())

    @staticmethod
    def _format_record(record: CommitRecord, with_island: bool = False) -> str:
        assert record.metrics is not None  # only successful commits are displayed
        island = f" | island {record.island_id}" if with_island else ""
        return (
            f"{record.id}{island} | {Island._format_metrics(record.metrics)} | {record.description}"
        )

    # ------------------------------------------------------------------
    # Tools
    # ------------------------------------------------------------------

    def _file_error(self, path: str, *, writing: bool = False) -> str | None:
        analysis = self.cfg.evolution.max_analyses_per_commit > 0
        if path == "program.py" or (path == "analysis.py" and analysis):
            return None
        if path in self.resource_names:
            return f"error: {path} is a read-only task resource" if writing else None
        usable = '"program.py" or "analysis.py"' if analysis else '"program.py"'
        if self.resource_names:
            usable += " or a task resource path"
        return f"error: unknown path {path!r}; use {usable}"

    def _get_file(self, path: str) -> str:
        if path == "program.py":
            return self.program
        if path == "analysis.py":
            return self.analysis
        # A listed resource path; its first component is the resources directory name.
        try:
            return (Path(cast(str, self.cfg.task.resources)).parent / path).read_text()
        except UnicodeDecodeError:
            return f"error: {path} is not a text file; use it from analysis.py"

    def _set_file(self, path: str, content: str) -> None:
        if path == "program.py":
            self.program = content
        else:
            self.analysis = content

    def _build_tools(self) -> list[ToolSpec]:
        island = self
        max_analyses = self.cfg.evolution.max_analyses_per_commit
        target = "program.py or analysis.py" if max_analyses else "program.py"
        path_doc = '"program.py" or "analysis.py".' if max_analyses else '"program.py".'
        read_target, read_doc = target, path_doc
        if self.resource_names:
            read_target = f"{target} or a task resource"
            read_doc = (
                path_doc[:-1] + f', or a resource path such as "{sorted(self.resource_names)[0]}".'
            )

        @tool(
            description=f"Read the current content of {read_target}.",
            parameters={"path": read_doc},
        )
        async def read(path: str) -> str:
            if error := island._file_error(path):
                return error
            return island._get_file(path) or "(empty)"

        @tool(
            description=f"Overwrite {target} with new content.",
            parameters={"path": path_doc, "content": "Full new file content."},
        )
        async def write(path: str, content: str) -> str:
            if error := island._file_error(path, writing=True):
                return error
            island._set_file(path, content)
            return f"wrote {len(content)} chars to {path}"

        @tool(
            description=f"Replace one exact occurrence of old_text in {target}.",
            parameters={
                "path": path_doc,
                "old_text": "Exact text to replace; must occur exactly once.",
                "new_text": "Replacement text.",
            },
        )
        async def edit(path: str, old_text: str, new_text: str) -> str:
            if error := island._file_error(path, writing=True):
                return error
            content = island._get_file(path)
            count = content.count(old_text)
            if count == 0:
                return "error: old_text not found"
            if count > 1:
                return f"error: old_text occurs {count} times; provide a unique match"
            island._set_file(path, content.replace(old_text, new_text, 1))
            return f"edited {path}"

        @tool
        async def run_analysis() -> str | ToolExecutionResult:
            """Execute the current analysis.py (it may `import program`) and return its output."""
            if island.behavior.run_analysis_count >= max_analyses:
                return "analysis budget for this commit cycle is exhausted"
            island.behavior.run_analysis_count += 1
            cycle = island.behavior.run_analysis_count
            island.analysis_count += 1
            number = island.analysis_count
            logger.info("island %d: analysis %d started", island.id, number)
            started = perf_counter()
            status, output = await island.executor.run_analysis(island.program, island.analysis)
            elapsed = perf_counter() - started
            level = logging.INFO if status == "ok" else logging.WARNING
            logger.log(
                level, "island %d: analysis %d %s in %.2fs", island.id, number, status, elapsed
            )
            return ToolExecutionResult(
                output=f"{output}\n\nanalyses {cycle}/{max_analyses} this cycle",
                metadata={"kind": "analysis", "status": status, "duration": round(elapsed, 3)},
            )

        @tool
        async def commit(description: str) -> str:
            """Formally evaluate the current program.py, record it, and return evolution feedback.

            Args:
                description: One or two sentences on what this version does and what changed.
            """
            if island.commit_count >= island.cfg.evolution.max_commits:
                return f"{island._budget_line()}\n\n{END_INSTRUCTION}"
            island.commit_count += 1
            commit_id = f"{island.id}-{island.commit_count}"
            program = island.program
            logger.info("island %d: evaluating commit %s", island.id, commit_id)
            started = perf_counter()
            output = None
            try:
                metrics = await island.executor.evaluate(program)
            except EvaluationError as exc:
                metrics, status, error, output = None, exc.status, str(exc), exc.output
            else:
                status, error = "ok", island._validate_metrics(metrics)
                if error is not None:
                    metrics, status = None, "error"
            # No await below: record, archive update and feedback construction
            # are atomic with respect to other islands.
            record = CommitRecord(
                id=commit_id,
                island_id=island.id,
                code=program,
                description=description,
                behavior=island.behavior,
                timestamp=datetime.now(UTC).isoformat(),
                duration=round(perf_counter() - started, 3),
                status=status,
                metrics=metrics,
                error=error,
                output=output,
            )
            return island._record_commit(record)

        @tool
        async def read_commit(commit_id: str) -> str:
            """Return the full program.py source of a recorded commit.

            Args:
                commit_id: A commit id from feedback, e.g. "2-14".
            """
            record = island.archive.records.get(commit_id)
            if record is None:
                return f"error: unknown commit id {commit_id!r}"
            island.behavior.read_commit_ids.append(commit_id)
            logger.info(
                "island %d: read commit %s from island %d",
                island.id,
                commit_id,
                record.island_id,
            )
            return record.code

        tools = [read, write, edit, run_analysis, commit, read_commit]
        if not max_analyses:
            tools.remove(run_analysis)
        return [self._counted(spec) for spec in tools]

    def _counted(self, spec: ToolSpec) -> ToolSpec:
        """Every tool call marks the turn as active; once the request budget is
        spent, every tool result ends the turn."""
        runner = cast(AsyncToolRunner, spec.runner)  # every tool above is an async def

        async def run(ctx, args) -> ToolExecutionResult:
            self.idle_turns = 0
            result = await runner(ctx, args)
            if self.requests_exhausted and END_INSTRUCTION not in result.output:
                result = replace(result, output=f"{result.output}\n\n{END_INSTRUCTION}")
            return result

        return replace(spec, runner=run)

    def _validate_metrics(self, metrics: dict) -> str | None:
        me = self.cfg.map_elites
        for key in [me.fitness, *me.features]:
            value = metrics.get(key)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
            ):
                return f"metrics field {key!r} must be a finite number, got {value!r}"
        return None

    def _record_commit(self, record: CommitRecord) -> str:
        best = self.archive.add(record)
        if record.metrics is None or best is None:  # failed evaluation
            parts = [
                f"Current commit:\n{record.id} | evaluation failed | {record.description}",
                f"Evaluation error:\n{record.error}",
            ]
        else:
            parts = [f"Current commit:\n{self._format_record(record)}"]
            parts.append(f"Current best:\n{self._format_record(best)}")
            elites = self.archive.island_elites(self.id)
            parts.append("Island MAP-Elites:\n" + "\n".join(self._format_record(r) for r in elites))

        evolution = self.cfg.evolution
        if (
            evolution.migration_interval > 0
            and self.commit_count % evolution.migration_interval == 0
        ):
            self.last_migration = self.archive.migration_candidates(
                self.id, evolution.migration_rate
            )
            fresh = [r for r in self.last_migration if r.id not in self.shown_migration_ids]
            self.shown_migration_ids.update(r.id for r in fresh)
            logger.info(
                "island %d: migration candidates: %s",
                self.id,
                ", ".join(record.id for record in fresh) or "no new candidates",
            )
            if fresh:
                parts.append(
                    "Migration candidates:\n"
                    + "\n".join(self._format_record(r, with_island=True) for r in fresh)
                )

        parts.append(self._budget_line())

        feedback = "\n\n".join(parts)
        if self.commit_count >= evolution.max_commits:
            feedback += f"\n\n{END_INSTRUCTION}"
        elif self.compact_requested:
            feedback += f"\n\n{STOP_INSTRUCTION}"
        self.behavior = Behavior()
        return feedback

    def _budget_line(self) -> str:
        evolution = self.cfg.evolution
        line = f"commits {self.commit_count}/{evolution.max_commits}"
        if evolution.max_total_tokens is not None:
            line += f" | total_tokens {self.total_tokens}/{evolution.max_total_tokens}"
        if evolution.max_requests is not None:
            line += f" | requests {self.request_count}/{evolution.max_requests}"
        return f"Budget:\n{line}"
