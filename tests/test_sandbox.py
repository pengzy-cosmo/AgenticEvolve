"""The execution boundary: task resources, the whitelisted environment, and the OS sandbox."""

from __future__ import annotations

import os
import subprocess
import uuid

import pytest
from conftest import usage
from test_evolution import archive_records

from agentic_evolve.config import load_config
from agentic_evolve.executor import _bwrap_args, _venv_home

PROBE = """\
import os, socket

def attempt(action):
    try:
        action()
        return "allowed"
    except OSError as exc:
        return type(exc).__name__

print(attempt(lambda: open({evaluator!r}).read()))
print(attempt(lambda: open({neighbor!r}).read()))
print(attempt(lambda: open({outside!r}, "w").write("x")))
print(attempt(lambda: os.kill({host_pid}, 0)))
print(attempt(lambda: socket.socket(socket.AF_INET, socket.SOCK_DGRAM).connect(("1.1.1.1", 53))))
print("EVOLVE_TEST_SECRET" in os.environ)
print(attempt(lambda: open("local.txt", "w").write("x")))
"""


async def probe_analysis(run_task, tmp_path, **kwargs) -> list[str]:
    """Run PROBE as analysis.py and return its output lines."""

    outputs = []
    (tmp_path / "neighbor.txt").write_text("x")

    async def scenario(agent, prompt):
        yield usage(100)
        probe = PROBE.format(
            evaluator=str(tmp_path / "task" / "evaluator.py"),
            neighbor=str(tmp_path / "neighbor.txt"),
            outside=str(tmp_path / "outside.txt"),
            host_pid=os.getpid(),
        )
        await agent.call("write", path="analysis.py", content=probe)
        outputs.append(await agent.call("run_analysis"))
        await agent.call("write", path="program.py", content="SCORE = 1\nFEATURE = 0.5\n")
        await agent.call("commit", description="boundary probe")

    await run_task(scenario, **kwargs)
    return outputs[0].removeprefix("stdout:\n").split("\n")[:7]


async def test_analysis_executions_are_sandboxed(run_task, tmp_path, monkeypatch):
    """Block private reads, outside writes, host signals, network and inherited secrets.

    The neighbor file is under the temporary root, like other executions.
    """

    monkeypatch.setenv("EVOLVE_TEST_SECRET", "hidden")
    evaluator, neighbor, outside, signal, network, secret, local = await probe_analysis(
        run_task, tmp_path
    )

    assert evaluator != "allowed"
    assert neighbor != "allowed"
    assert outside != "allowed" and not (tmp_path / "outside.txt").exists()
    assert signal != "allowed"
    assert network != "allowed"
    assert secret == "False"
    assert local == "allowed"


async def test_environment_is_whitelisted_without_the_sandbox(run_task, tmp_path, monkeypatch):
    monkeypatch.setenv("EVOLVE_TEST_SECRET", "hidden")
    lines = await probe_analysis(run_task, tmp_path, sandbox="disabled")

    assert lines == ["allowed"] * 5 + ["False", "allowed"]
    (tmp_path / "outside.txt").unlink()


async def test_resources_are_readable_importable_and_read_only(run_task):
    """Resources are served by read, refused by write and edit, and importable
    from program.py in both analysis and evaluation, where the copy in the
    execution directory is the one imported. Errors raised in resource code
    keep their messages, like evaluator errors."""

    outputs = []
    program = (
        "import os\n"
        "from data import helper\n"
        "SCORE = helper.VALUE\n"
        "copy = os.path.realpath(os.path.dirname(os.path.dirname(helper.__file__)))\n"
        "FEATURE = float(copy == os.path.realpath(os.getcwd()))\n"
    )

    async def scenario(agent, prompt):
        yield usage(100)
        assert (await agent.call("read", path="data/helper.py")).startswith("VALUE = 7\n")
        refused = await agent.call("write", path="data/helper.py", content="VALUE = 8\n")
        assert refused == "error: data/helper.py is a read-only task resource"
        refused = await agent.call("edit", path="data/helper.py", old_text="7", new_text="8")
        assert refused == "error: data/helper.py is a read-only task resource"
        unknown = await agent.call("read", path="data/../evaluator.py")
        assert unknown.startswith("error: unknown path")
        await agent.call("write", path="program.py", content=program)
        analysis = "import program\nprint(program.SCORE, program.FEATURE)\n"
        await agent.call("write", path="analysis.py", content=analysis)
        outputs.append(await agent.call("run_analysis"))
        await agent.call("commit", description="uses the resource module")
        await agent.call(
            "write", path="program.py", content="from data import helper\nhelper.check(-1)\n"
        )
        await agent.call("commit", description="invalid input")

    helper = "VALUE = 7\n\ndef check(x):\n    if x < 0:\n        raise ValueError(f'x={x} is negative')\n"
    run_dir = await run_task(scenario, max_commits=2, resources={"helper.py": helper})

    assert outputs[0].startswith("stdout:\n7 1.0\n")
    assert archive_records(run_dir)[0]["metrics"] == {"score": 7.0, "feature": 1.0}
    assert archive_records(run_dir)[1]["error"] == "ValueError: x=-1 is negative"


def test_bwrap_mounts_the_interpreter_as_the_venv_refers_to_it(tmp_path):
    """A venv whose python link and pyvenv.cfg go through an alias symlink of the
    base interpreter directory (uv's layout) needs the alias mounted too."""

    base = tmp_path / "cpython-3.14.7"
    (base / "bin").mkdir(parents=True)
    alias = tmp_path / "cpython-3.14"
    alias.symlink_to(base)
    venv = tmp_path / "venv"
    venv.mkdir()
    (venv / "pyvenv.cfg").write_text(f"home = {alias}/bin\nversion_info = 3.14.7\n")

    assert _venv_home(venv) == alias
    assert _venv_home(tmp_path) is None
    mounts = " ".join(_bwrap_args([venv, base, alias], tmp_path / "work"))
    assert f"--ro-bind {alias} {alias}" in mounts
    assert f"--ro-bind {base} {base}" in mounts


def test_resources_must_be_self_contained(tmp_path):
    """A symlink leaving the resources directory is rejected when the config
    loads, before any snapshot or copy could follow it."""

    task = tmp_path / "task"
    resources = task / "resources"
    resources.mkdir(parents=True)
    (task / "evaluator.py").write_text("def evaluate(p): return {}\n")
    (task / "initial_program.py").write_text("")
    (task / "config.yaml").write_text(
        "task: {initial_program: initial_program.py, evaluator: evaluator.py,"
        " resources: resources, description: x}\n"
        "llm: {model: m}\nmap_elites: {features: [f]}\n"
    )
    (resources / "note.txt").write_text("ok")
    (resources / "alias.txt").symlink_to(resources / "note.txt")
    load_config(task / "config.yaml")

    (resources / "leak.py").symlink_to(task / "evaluator.py")
    with pytest.raises(ValueError, match="points outside"):
        load_config(task / "config.yaml")


async def test_executions_kill_their_process_group(run_task):
    """Kill the process group after timeout and normal exit; setsid can escape it."""

    marker = f"evolve-sandbox-test-{uuid.uuid4().hex}"
    spawn = (
        "import subprocess, sys, time\n"
        f"subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)  # {marker}'],"
        " stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)\n"
    )

    async def scenario(agent, prompt):
        yield usage(100)
        await agent.call("write", path="analysis.py", content=spawn + "time.sleep(30)\n")
        assert "timed out" in await agent.call("run_analysis")
        await agent.call("write", path="analysis.py", content=spawn)
        assert (await agent.call("run_analysis")).startswith("(no output)")
        await agent.call("write", path="program.py", content="SCORE = 1\nFEATURE = 0.5\n")
        await agent.call("commit", description="after timeout")

    await run_task(scenario, timeout=1.0)

    survivors = subprocess.run(["pgrep", "-f", marker], capture_output=True, text=True)
    assert survivors.stdout == ""


async def test_output_and_result_files_are_read_defensively(run_task, tmp_path):
    """Truncate output, reject linked metrics and tolerate malformed error records."""

    elsewhere = tmp_path / "elsewhere.json"
    elsewhere.write_text('{"score": 9.0, "feature": 0.5}')
    outputs = []

    async def scenario(agent, prompt):
        yield usage(100)
        flood = "import sys\nfor _ in range(1000):\n    sys.stdout.write('x' * 1000)\n"
        await agent.call("write", path="analysis.py", content=flood)
        outputs.append(await agent.call("run_analysis"))
        forge = f"import os\nos.symlink({str(elsewhere)!r}, 'metrics.json')\nos._exit(0)\n"
        await agent.call("write", path="program.py", content=forge)
        await agent.call("commit", description="linked metrics")
        forge = (
            "import os\nopen('error.json', 'w').write('{\"frames\": [null, []]}')\nos._exit(1)\n"
        )
        await agent.call("write", path="program.py", content=forge)
        await agent.call("commit", description="forged error")

    run_dir = await run_task(scenario, max_commits=2)

    truncated = "stdout:\n" + "x" * 10_000 + "\n... (truncated, 1000000 bytes total)\n"
    assert outputs[0].startswith(truncated)
    linked, forged = archive_records(run_dir)
    assert linked["error"] == "invalid metrics file: metrics.json is not a regular file"
    assert forged["error"] == "evaluator exited with code 1"
