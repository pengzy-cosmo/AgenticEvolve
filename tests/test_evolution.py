"""Core evolution behavior through the controller and real executor."""

from __future__ import annotations

import asyncio
import json

from conftest import usage
from mycode import TextEvent


def archive_records(run_dir):
    return [json.loads(line) for line in (run_dir / "archive.jsonl").read_text().splitlines()]


def feedback_section(feedback: str, heading: str) -> str:
    return feedback.split(f"{heading}:\n", 1)[1].split("\n\n", 1)[0]


async def test_task_resources_support_analysis_guided_evolution(run_task):
    async def scenario(agent, prompt):
        assert "data/cases.txt (0.0 kB)" in agent.system
        await agent.call(
            "write",
            path="analysis.py",
            content=(
                "cases = [int(x) for x in open('data/cases.txt').read().split()]\n"
                "print(max(value * value for value in cases))\n"
            ),
        )
        analysis = await agent.call("run_analysis")
        score = analysis.removeprefix("stdout:\n").split("\n")[0]
        await agent.call(
            "write",
            path="program.py",
            content=f"SCORE = {score}\nFEATURE = 0.5\n",
        )
        yield usage(100)
        await agent.call("commit", description="used the observed cases to find the worst case")

    run_dir = await run_task(scenario, resources={"cases.txt": "2 4 7\n"})
    record = archive_records(run_dir)[0]

    assert record["metrics"]["score"] == 49.0
    assert record["behavior"]["run_analysis_count"] == 1
    assert record["behavior"]["context_tokens"] == 100


async def test_initial_program_is_evaluated_before_the_run(run_task):
    """The framework evaluates the initial program once at run start: the
    result is recorded in initial.json as the run's first formal evaluation,
    its metrics appear in the opening message, and no commit budget is
    consumed."""

    prompts = []

    async def scenario(agent, prompt):
        prompts.append(prompt)
        yield usage(100)
        await agent.call("write", path="program.py", content="SCORE = 3\nFEATURE = 0.5\n")
        await agent.call("commit", description="improved")

    run_dir = await run_task(scenario)
    initial = json.loads((run_dir / "initial.json").read_text())

    assert initial["metrics"] == {"score": 0.0, "feature": 0.0}
    assert initial["duration"] > 0
    assert "Initial evaluation:\nscore=0.0 feature=0.0" in prompts[0]
    assert "commits 0/1" in prompts[0]
    assert archive_records(run_dir)[0]["id"] == "0-1"


async def test_multi_island_evolution_produces_archive_and_best(run_task):
    first_commit = asyncio.Event()

    async def scenario(agent, prompt):
        if agent.session_id == "0":
            source = await agent.call("read", path="program.py")
            await agent.call(
                "write",
                path="analysis.py",
                content="import program\nprint(program.SCORE + 1)\n",
            )
            analysis = await agent.call("run_analysis")
            score = analysis.removeprefix("stdout:\n").split("\n")[0]
            improved = source.replace("SCORE = 0", f"SCORE = {score}")
            await agent.call("write", path="program.py", content=improved)
            yield usage(100)
            await agent.call("commit", description="analysis-guided improvement")
            first_commit.set()
            return

        await first_commit.wait()
        source = await agent.call("read_commit", commit_id="0-1")
        await agent.call("write", path="program.py", content=source)
        await agent.call(
            "edit",
            path="program.py",
            old_text="SCORE = 1\nFEATURE = 0",
            new_text="SCORE = 2\nFEATURE = 0.8",
        )
        yield usage(100)
        await agent.call("commit", description="improved neighboring program")

    run_dir = await run_task(scenario, islands=2)
    records = archive_records(run_dir)

    assert {record["id"] for record in records} == {"0-1", "1-1"}
    assert (
        next(record for record in records if record["id"] == "0-1")["behavior"][
            "run_analysis_count"
        ]
        == 1
    )
    assert next(record for record in records if record["id"] == "1-1")["behavior"][
        "read_commit_ids"
    ] == ["0-1"]
    assert "SCORE = 2" in (run_dir / "best.py").read_text()
    assert json.loads((run_dir / "best.json").read_text())["id"] == "1-1"


async def test_map_elites_feedback_replaces_only_within_a_cell(run_task):
    final_feedback = None

    async def scenario(agent, prompt):
        nonlocal final_feedback
        candidates = [(1, 0.1), (2, 0.1), (1.5, 0.9)]
        for score, feature in candidates:
            await agent.call(
                "write",
                path="program.py",
                content=f"SCORE = {score}\nFEATURE = {feature}\n",
            )
            yield usage(100)
            final_feedback = await agent.call("commit", description=f"score {score}")

    run_dir = await run_task(scenario, max_commits=3)
    assert final_feedback is not None
    elites = feedback_section(final_feedback, "Island MAP-Elites")

    assert "0-2" in elites
    assert "0-3" in elites
    assert "0-1" not in elites
    assert len(archive_records(run_dir)) == 3
    assert json.loads((run_dir / "best.json").read_text())["id"] == "0-2"


async def test_migration_feedback_uses_ring_neighbors(run_task):
    neighbors_ready = asyncio.Event()
    completed_neighbors = 0
    target_feedback = None

    async def scenario(agent, prompt):
        nonlocal completed_neighbors, target_feedback
        island = int(agent.session_id)
        if island == 0:
            await neighbors_ready.wait()
        await agent.call(
            "write",
            path="program.py",
            content=f"SCORE = {island + 1}\nFEATURE = {island / 4}\n",
        )
        yield usage(100)
        feedback = await agent.call("commit", description=f"island {island}")
        if island == 0:
            target_feedback = feedback
        else:
            completed_neighbors += 1
            if completed_neighbors == 3:
                neighbors_ready.set()

    await run_task(scenario, islands=4, migration_interval=1)
    assert target_feedback is not None
    migration = feedback_section(target_feedback, "Migration candidates")

    assert "1-1" in migration
    assert "3-1" in migration
    assert "2-1" not in migration


async def test_migration_candidates_are_shown_once(run_task):
    neighbor_done = asyncio.Event()
    feedbacks = []

    async def scenario(agent, prompt):
        island = int(agent.session_id)
        if island == 1:
            await agent.call("write", path="program.py", content="SCORE = 5\nFEATURE = 0.5\n")
            yield usage(100)
            await agent.call("commit", description="neighbor")
            await agent.call("commit", description="neighbor unchanged")
            neighbor_done.set()
            return
        await neighbor_done.wait()
        yield usage(100)
        feedbacks.append(await agent.call("commit", description="first"))
        feedbacks.append(await agent.call("commit", description="second"))

    await run_task(scenario, islands=2, max_commits=2, migration_interval=1)

    assert "1-1" in feedback_section(feedbacks[0], "Migration candidates")
    assert "Migration candidates" not in feedbacks[1]


async def test_failed_evaluations_are_recorded_and_consume_budget(run_task):
    """Every commit call becomes an archive row with a consecutive id. Failures
    carry the error and program snapshot, count against max_commits, and take
    the behavior stats accumulated before them; analysis stats stay in session
    tool metadata."""

    feedbacks = []
    analysis_metadata = None
    failed_source = None

    async def scenario(agent, prompt):
        nonlocal analysis_metadata, failed_source
        yield usage(100)
        await agent.call("write", path="analysis.py", content="print('looking around')\n")
        await agent.call("run_analysis")
        analysis_metadata = agent.last_metadata
        await agent.call("write", path="program.py", content="raise RuntimeError('boom')\n")
        feedbacks.append(await agent.call("commit", description="broken candidate"))
        failed_source = await agent.call("read_commit", commit_id="0-1")
        await agent.call("write", path="program.py", content="SCORE = 1\nFEATURE = 0.5\n")
        feedbacks.append(await agent.call("commit", description="fixed candidate"))
        feedbacks.append(await agent.call("commit", description="over budget"))

    run_dir = await run_task(scenario, max_commits=2)
    failed, succeeded = archive_records(run_dir)

    assert failed["id"] == "0-1"
    assert failed["status"] == "error"
    assert failed["error"] == "RuntimeError at <string> line 1"
    assert "RuntimeError: boom" in failed["output"]
    assert "metrics" not in failed
    assert failed["behavior"]["run_analysis_count"] == 1
    assert failed_source == "raise RuntimeError('boom')\n"
    assert succeeded["id"] == "0-2"
    assert succeeded["status"] == "ok"
    assert "error" not in succeeded
    assert succeeded["metrics"]["score"] == 1.0
    assert succeeded["behavior"]["run_analysis_count"] == 0
    assert all(record["duration"] > 0 for record in (failed, succeeded))

    assert "evaluation failed" in feedbacks[0]
    assert "commits 1/2" in feedbacks[0]
    assert "budget exhausted" in feedbacks[2]
    assert analysis_metadata is not None
    assert analysis_metadata["kind"] == "analysis"
    assert analysis_metadata["status"] == "ok"
    assert analysis_metadata["duration"] > 0


async def test_commit_budget_stops_a_continuous_tool_loop(run_task):
    async def scenario(agent, prompt):
        for score in range(1, 6):
            yield usage(100)
            if agent.cancelled:
                return
            await agent.call(
                "write",
                path="program.py",
                content=f"SCORE = {score}\nFEATURE = 0.5\n",
            )
            await agent.call("commit", description=f"score {score}")

    run_dir = await run_task(scenario, max_commits=2)

    assert [record["metrics"]["score"] for record in archive_records(run_dir)] == [1.0, 2.0]


async def test_request_budget_lets_the_last_response_finish_its_tools(run_task):
    """Every model request counts. The response that reaches the limit still
    runs its tool calls and each result carries the end instruction; one more
    request is cancelled before its tools run, and the run terminates."""
    feedbacks = []

    async def scenario(agent, prompt):
        for score in range(1, 5):
            yield usage(100)
            if agent.cancelled:
                return
            await agent.call(
                "write", path="program.py", content=f"SCORE = {score}\nFEATURE = 0.5\n"
            )
            feedbacks.append(await agent.call("commit", description=f"score {score}"))

    run_dir = await run_task(scenario, max_commits=10, max_requests=2)

    assert [record["metrics"]["score"] for record in archive_records(run_dir)] == [1.0, 2.0]
    assert "requests 1/2" in feedbacks[0] and "budget exhausted" not in feedbacks[0]
    assert "requests 2/2" in feedbacks[1] and "budget exhausted" in feedbacks[1]


async def test_island_ends_after_consecutive_turns_without_tool_calls(run_task):
    """A turn with a tool call resets the count; max_idle_turns idle turns end the island."""
    prompts = []

    async def scenario(agent, prompt):
        prompts.append(prompt)
        yield usage(100)
        if len(prompts) == 3:
            await agent.call("write", path="program.py", content="SCORE = 1\nFEATURE = 0.5\n")
            await agent.call("commit", description="the only commit")

    run_dir = await run_task(scenario, max_commits=10, max_idle_turns=5)

    assert len(prompts) == 2 + 1 + 5
    assert [record["metrics"]["score"] for record in archive_records(run_dir)] == [1.0]


async def test_token_budget_uses_current_context_before_compaction(run_task):
    async def scenario(agent, prompt):
        yield usage(100)
        await agent.call("commit", description="within budget")
        yield usage(200)
        if agent.cancelled:
            return
        await agent.call("commit", description="still within budget")
        yield usage(300)
        if agent.cancelled:
            return
        await agent.call("commit", description="over budget")

    run_dir = await run_task(scenario, max_commits=10, max_total_tokens=250)
    records = archive_records(run_dir)

    assert len(records) == 2
    assert records[-1]["behavior"]["context_tokens"] == 200


async def test_compaction_handoff_continues_the_evolution(run_task):
    async def scenario(agent, prompt):
        if not agent.tools:
            yield TextEvent(delta="The next step is to set SCORE to 5.")
            yield usage(650)
            return
        if agent.session_id == "0":
            yield usage(600)
            return

        score = 5 if "set SCORE to 5" in prompt else -1
        await agent.call(
            "write",
            path="program.py",
            content=f"SCORE = {score}\nFEATURE = 0.5\n",
        )
        yield usage(100)
        await agent.call("commit", description="continued after context reset")

    run_dir = await run_task(
        scenario,
        context_window=1_000,
        compact_threshold=0.5,
        safety_threshold=0.9,
    )

    assert archive_records(run_dir)[0]["metrics"]["score"] == 5.0
    assert archive_records(run_dir)[0]["behavior"]["context_tokens"] == 100


async def test_model_metadata_overrides_reach_the_agent(run_task):
    """context_window and max_tokens from the llm config replace the catalog
    values, so a model missing from the catalog runs with its true limits."""

    async def scenario(agent, prompt):
        assert agent.context_window == 262_144
        assert agent.max_tokens == 32_768
        yield usage(100)
        await agent.call("commit", description="metadata check")

    run_dir = await run_task(scenario, context_window=262_144, max_tokens=32_768)
    assert archive_records(run_dir)[0]["status"] == "ok"


async def test_token_budget_accumulates_across_compaction(run_task):
    async def scenario(agent, prompt):
        if not agent.tools:
            yield usage(650)
            return
        if agent.session_id == "0":
            yield usage(600)
            return

        yield usage(100)
        if agent.cancelled:
            return
        await agent.call("commit", description="over budget")

    run_dir = await run_task(
        scenario,
        max_commits=10,
        max_total_tokens=700,
        context_window=1_000,
        compact_threshold=0.5,
        safety_threshold=0.9,
    )

    assert not (run_dir / "archive.jsonl").exists()


async def test_execution_timeouts_are_visible_and_recoverable(run_task):
    evaluator = """\
import time

def evaluate(program_path):
    source = open(program_path).read()
    if "SLOW = True" in source:
        time.sleep(5)
    namespace = {}
    exec(source, namespace)
    return {"score": float(namespace["SCORE"]), "feature": float(namespace["FEATURE"])}
"""

    analysis_status = None

    async def scenario(agent, prompt):
        nonlocal analysis_status
        yield usage(100)
        await agent.call("write", path="analysis.py", content="import time\ntime.sleep(5)\n")
        analysis = await agent.call("run_analysis")
        analysis_status = agent.last_metadata["status"]
        if "timed out" not in analysis:
            return

        await agent.call(
            "write",
            path="program.py",
            content="SLOW = True\nSCORE = 1\nFEATURE = 0.5\n",
        )
        failed = await agent.call("commit", description="slow candidate")
        if "timed out" not in failed:
            return

        await agent.call(
            "write",
            path="program.py",
            content="SLOW = False\nSCORE = 3\nFEATURE = 0.5\n",
        )
        yield usage(200)
        await agent.call("commit", description="removed slow path")

    run_dir = await run_task(scenario, max_commits=2, timeout=0.5, evaluator=evaluator)
    slow, recovered = archive_records(run_dir)

    assert analysis_status == "timeout"
    assert slow["status"] == "timeout"
    assert slow["behavior"]["run_analysis_count"] == 1
    assert recovered["id"] == "0-2"
    assert recovered["status"] == "ok"
    assert recovered["metrics"]["score"] == 3.0


async def test_no_analysis_arm_removes_the_analysis_capability(run_task):
    """max_analyses_per_commit=0 removes run_analysis, analysis.py, and every
    mention of the analysis environment from the model's world; evolution
    proceeds on program.py alone. Task resources stay listed and readable."""

    prompts = []

    async def scenario(agent, prompt):
        prompts.append(prompt)
        yield usage(100)
        assert set(agent.tools) == {"read", "write", "edit", "commit", "read_commit"}
        assert "analysis" not in agent.system
        assert "data/cases.txt" in agent.system
        error = await agent.call("write", path="analysis.py", content="print(1)\n")
        assert error == (
            "error: unknown path 'analysis.py'; use \"program.py\" or a task resource path"
        )
        assert await agent.call("read", path="data/cases.txt") == "1\n"
        await agent.call("write", path="program.py", content="SCORE = 1\nFEATURE = 0.5\n")
        await agent.call("commit", description="no analysis capability")

    run_dir = await run_task(scenario, max_analyses_per_commit=0, resources={"cases.txt": "1\n"})
    record = archive_records(run_dir)[0]

    assert "commits 0/1" in prompts[0]  # the opening message states the budget
    assert record["status"] == "ok"
    assert record["behavior"]["run_analysis_count"] == 0


async def test_analysis_budget_is_per_commit_cycle(run_task):
    """Each commit cycle allows max_analyses_per_commit run_analysis calls;
    calls over the limit are refused without executing, and a commit, failed
    or not, renews the budget."""

    async def scenario(agent, prompt):
        yield usage(100)
        await agent.call("write", path="analysis.py", content="print('probe')\n")
        first = await agent.call("run_analysis")
        assert "analyses 1/2 this cycle" in first
        await agent.call("run_analysis")
        refused = await agent.call("run_analysis")
        assert refused == "analysis budget for this commit cycle is exhausted"
        await agent.call("write", path="program.py", content="broken =\n")
        await agent.call("commit", description="invalid program")
        renewed = await agent.call("run_analysis")
        assert "analyses 1/2 this cycle" in renewed
        await agent.call("write", path="program.py", content="SCORE = 1\nFEATURE = 0.5\n")
        await agent.call("commit", description="after renewal")

    run_dir = await run_task(scenario, max_analyses_per_commit=2, max_commits=2)
    failed, succeeded = archive_records(run_dir)

    assert failed["status"] == "error"
    assert failed["behavior"]["run_analysis_count"] == 2  # the refused call is not counted
    assert succeeded["behavior"]["run_analysis_count"] == 1


async def test_analysis_environment_holds_resources_and_nothing_else(run_task):
    """The resources directory is copied into every analysis execution; no
    other task file (the evaluator included) is present."""

    outputs = []

    async def scenario(agent, prompt):
        yield usage(100)
        await agent.call(
            "write",
            path="analysis.py",
            content=(
                "from pathlib import Path\n"
                "print(Path('data/cases.txt').read_text().strip())\n"
                "print(sorted(entry.name for entry in Path('.').iterdir()))\n"
            ),
        )
        outputs.append(await agent.call("run_analysis"))
        await agent.call("write", path="program.py", content="SCORE = 1\nFEATURE = 0.5\n")
        await agent.call("commit", description="data probe")

    await run_task(scenario, resources={"cases.txt": "public cases\n"})

    assert "public cases\n['analysis.py', 'data', 'program.py']" in outputs[0]


async def test_executions_run_in_ephemeral_redirected_environments(run_task):
    """HOME and the temp-dir variables point into the per-execution directory,
    so well-behaved temp files vanish with it; BLAS threads are pinned."""

    outputs = []

    async def scenario(agent, prompt):
        yield usage(100)
        await agent.call(
            "write",
            path="analysis.py",
            content=(
                "import os\n"
                "cwd = os.getcwd()\n"
                "keys = ('HOME', 'TMPDIR', 'TMP', 'TEMP')\n"
                "print(all(os.path.realpath(os.environ[key]) == cwd for key in keys))\n"
                "print(os.environ['OMP_NUM_THREADS'])\n"
            ),
        )
        outputs.append(await agent.call("run_analysis"))
        await agent.call("commit", description="environment probe")

    await run_task(scenario)

    assert "True\n1" in outputs[0]


async def test_failed_evaluation_feedback_carries_no_program_output(run_task):
    """Candidate diagnostics stay in the archive; evaluator rejection messages reach the model."""
    evaluator = """\
def evaluate(program_path):
    namespace = {}
    exec(open(program_path).read(), namespace)
    if namespace["SCORE"] < 0:
        raise ValueError("score must be non-negative")
    return {"score": float(namespace["SCORE"]), "feature": 0.5}
"""
    programs = [
        "class Leak(Exception):\n    pass\n\nraise Leak('frequency 0.776')\n",
        "SCORE = -1\n",
        "import sys\nprint('frequency 0.776')\nsys.exit(3)\n",
    ]
    feedbacks = []

    async def scenario(agent, prompt):
        yield usage(100)
        for program in programs:
            await agent.call("write", path="program.py", content=program)
            feedbacks.append(await agent.call("commit", description="probe"))

    run_dir = await run_task(scenario, max_commits=3, evaluator=evaluator)
    errors = [record["error"] for record in archive_records(run_dir)]

    assert errors == [
        "Exception at <string> line 4",
        "ValueError: score must be non-negative",
        "evaluator exited with code 3",
    ]
    assert all("0.776" not in feedback for feedback in feedbacks)
    assert all("0.776" in record["output"] for record in archive_records(run_dir)[::2])


async def test_non_finite_metrics_are_rejected(run_task):
    evaluator = """\
def evaluate(program_path):
    namespace = {}
    exec(open(program_path).read(), namespace)
    score = float(namespace["SCORE"])
    return {"score": score if score else float("nan"), "feature": 0.5}
"""
    first_feedback = None

    async def scenario(agent, prompt):
        nonlocal first_feedback
        yield usage(100)
        if first_feedback is None:
            await agent.call("write", path="program.py", content="SCORE = 0\nFEATURE = 0\n")
            first_feedback = await agent.call("commit", description="nan score")
            return
        await agent.call("write", path="program.py", content="SCORE = 1\nFEATURE = 0.5\n")
        await agent.call("commit", description="finite score")

    run_dir = await run_task(
        scenario, max_commits=2, evaluator=evaluator, initial_program="SCORE = 2\nFEATURE = 0.5\n"
    )
    records = archive_records(run_dir)

    assert first_feedback is not None
    assert "must be a finite number" in first_feedback
    assert [record["status"] for record in records] == ["error", "ok"]
    assert records[1]["metrics"]["score"] == 1.0
