"""Subprocess execution of analysis runs and formal evaluations.

Executions use fresh temporary directories, a restricted environment, and
optional OS sandboxing. The evaluator and candidate share a process and read
permissions, including access to private evaluator inputs.

Analysis output is returned to the model. Evaluation failures return a summary
of the runner's exception record, keeping raw stdout and stderr in the archive.
Each output stream is truncated to output_limit before storage or feedback.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import signal
import subprocess
import sys
import tempfile
from pathlib import Path

from agentic_evolve.config import ExecutorConfig, TaskConfig

logger = logging.getLogger(__name__)

_RUNNER_PATH = Path(__file__).parent / "_eval_runner.py"
METRICS_FILENAME = "metrics.json"
ERROR_FILENAME = "error.json"
# -E ignores PYTHON* variables, -s skips the user site-packages directory.
_INTERPRETER = [sys.executable, "-E", "-s"]


class EvaluationError(Exception):
    """Failed evaluation with model-facing feedback and separate diagnostic output."""

    def __init__(self, message: str, status: str = "error", output: str = ""):
        super().__init__(message)
        self.status = status
        self.output = output


class Executor:
    def __init__(self, cfg: ExecutorConfig, task: TaskConfig, run_dir: Path):
        self.cfg = cfg
        self.evaluator_path = Path(task.evaluator).resolve()
        self.resources = Path(task.resources).resolve() if task.resources else None
        self.run_dir = Path(run_dir).resolve()
        self.sandbox = _select_sandbox(cfg.sandbox)
        logger.info("execution sandbox: %s", self.sandbox or "none")

    async def run_analysis(self, program: str, analysis: str) -> tuple[str, str]:
        """Run analysis.py next to program.py; return (status, truncated output text)."""

        tmp = Path(tempfile.mkdtemp(prefix="evolve-analysis-"))
        try:
            (tmp / "program.py").write_text(program)
            (tmp / "analysis.py").write_text(analysis)
            self._copy_resources(tmp)
            code, out, err, timed_out = await self._run([*_INTERPRETER, "analysis.py"], tmp)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

        parts = []
        if timed_out:
            status = "timeout"
            parts.append(f"error: analysis timed out after {self.cfg.timeout}s")
        elif code != 0:
            status = "error"
            parts.append(f"exit code: {code}")
        else:
            status = "ok"
        if out:
            parts.append(f"stdout:\n{out}")
        if err:
            parts.append(f"stderr:\n{err}")
        return status, "\n".join(parts) or "(no output)"

    async def evaluate(self, program: str) -> dict:
        """Run the formal evaluator against program.py and return its metrics dict."""

        tmp = Path(tempfile.mkdtemp(prefix="evolve-eval-"))
        try:
            (tmp / "program.py").write_text(program)
            self._copy_resources(tmp)
            metrics_path, error_path = tmp / METRICS_FILENAME, tmp / ERROR_FILENAME
            cmd = [
                *_INTERPRETER,
                str(_RUNNER_PATH),
                str(self.evaluator_path),
                str(tmp / "program.py"),
                str(metrics_path),
                str(error_path),
            ]
            code, out, err, timed_out = await self._run(cmd, tmp, evaluation=True)
            if timed_out:
                raise EvaluationError(
                    f"evaluation timed out after {self.cfg.timeout}s", status="timeout"
                )
            output = f"stdout:\n{out}\nstderr:\n{err}"
            if code != 0:
                if error_path.exists():
                    raise EvaluationError(
                        self._summary(json.loads(error_path.read_text())), output=output
                    )
                raise EvaluationError(f"evaluator exited with code {code}", output=output)
            if not metrics_path.exists():
                raise EvaluationError("evaluator wrote no metrics file", output=output)
            try:
                metrics = json.loads(metrics_path.read_text())
            except json.JSONDecodeError as exc:
                raise EvaluationError(f"metrics file is not valid JSON: {exc}") from exc
            if not isinstance(metrics, dict):
                raise EvaluationError(
                    f"metrics must be a JSON object, got {type(metrics).__name__}"
                )
            return metrics
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def _copy_resources(self, tmp: Path) -> None:
        if self.resources is not None:
            shutil.copytree(self.resources, tmp / self.resources.name)

    async def _run(
        self, cmd: list[str], cwd: Path, *, evaluation: bool = False
    ) -> tuple[int | None, str, str, bool]:
        """Run with a restricted environment and clean up on timeout or cancellation."""

        env = {"PATH": os.environ.get("PATH", os.defpath)}
        env |= {key: str(cwd) for key in ("HOME", "TMPDIR", "TMP", "TEMP")}
        env |= {
            key: "1"
            for key in (
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS",
            )
        }
        if self.sandbox is not None:
            cmd = self._sandboxed(cmd, cwd.resolve(), evaluation)
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=cwd,
            env=env,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
        )
        try:
            out, err = await asyncio.wait_for(proc.communicate(), self.cfg.timeout)
        except TimeoutError:
            await self._kill_group(proc)
            return None, "", "", True
        except asyncio.CancelledError:
            await self._kill_group(proc)
            raise
        return (
            proc.returncode,
            self._truncate(out.decode(errors="replace")),
            self._truncate(err.decode(errors="replace")),
            False,
        )

    def _sandboxed(self, cmd: list[str], cwd: Path, evaluation: bool) -> list[str]:
        """Wrap cmd in the platform sandbox; cwd is the only writable location."""

        readable = [Path(sys.prefix), Path(sys.base_prefix), _RUNNER_PATH.parent, cwd]
        if (home := _venv_home(Path(sys.prefix))) is not None:
            readable.append(home)
        if evaluation:
            readable.append(self.evaluator_path.parent)
        if self.sandbox == "sandbox-exec":
            import pwd

            home = Path(pwd.getpwuid(os.getuid()).pw_dir).resolve()
            hidden = [home, self.run_dir, self.evaluator_path.parent]
            return ["sandbox-exec", "-p", _seatbelt_profile(readable, cwd, hidden), *cmd]
        return [*_bwrap_args(readable, cwd), *cmd]

    def _summary(self, error: dict) -> str:
        """Keep messages raised in evaluator code; summarize candidate and library errors
        by builtin exception type and source location. Custom types become Exception.
        """
        evaluator_dir = self.evaluator_path.parent
        frames = error["frames"][1:]  # the first frame is the runner's own call
        if frames and Path(frames[-1][0]).is_relative_to(evaluator_dir):
            return f"{error['type']}: {error['message']}".partition("\n")[0]
        name = error["type"] if error["builtin"] else "Exception"
        library = (sys.prefix, sys.base_prefix)
        program = [
            (path, line)
            for path, line in frames
            if not Path(path).is_relative_to(evaluator_dir) and not path.startswith(library)
        ]
        if not program:
            return name
        path, line = program[-1]
        return f"{name} at {Path(path).name} line {line}"

    def _truncate(self, text: str) -> str:
        if len(text) <= self.cfg.output_limit:
            return text
        return text[: self.cfg.output_limit] + f"\n... (truncated, {len(text)} chars total)"

    @staticmethod
    async def _kill_group(proc: asyncio.subprocess.Process) -> None:
        # On POSIX, start_new_session=True makes the child's PID its process group ID.
        # On Windows, this terminates only the direct child.
        try:
            if sys.platform == "win32":
                proc.kill()
            else:
                os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        await proc.wait()


def _select_sandbox(mode: str) -> str | None:
    """Select the platform sandbox; raise or warn if unavailable, according to mode."""

    if mode == "disabled":
        return None
    if sys.platform == "darwin":
        tool, problem = "sandbox-exec", None
    elif sys.platform == "linux":
        tool, problem = "bwrap", _bwrap_problem()
    else:
        tool, problem = "", f"no sandbox is available on {sys.platform}"
    if problem is None:
        return tool
    if mode == "required":
        raise RuntimeError(
            f"{problem}; set executor.sandbox to auto or disabled to run without one"
        )
    logger.warning("running without a sandbox: %s", problem)
    return None


def _bwrap_problem() -> str | None:
    """Probe namespace creation, which host policy may restrict even with bwrap installed."""

    if shutil.which("bwrap") is None:
        return "bwrap not found (install bubblewrap)"
    probe = subprocess.run(
        ["bwrap", "--unshare-all", "--die-with-parent", "--ro-bind", "/", "/", "/bin/true"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    if probe.returncode != 0:
        return f"bwrap cannot create a sandbox: {probe.stderr.strip()}"
    return None


def _venv_home(prefix: Path) -> Path | None:
    """The base interpreter directory as the virtual environment at prefix refers
    to it (pyvenv.cfg's home is its bin directory), or None outside a venv. uv
    installs interpreters in version directories reached through an alias
    symlink; venvs refer to the alias while sys.base_prefix is the resolved
    directory, and the venv's python symlink goes through the alias."""

    cfg = prefix / "pyvenv.cfg"
    if not cfg.exists():
        return None
    for line in cfg.read_text().splitlines():
        key, sep, value = line.partition("=")
        if sep and key.strip() == "home":
            return Path(value.strip()).parent
    return None


def _seatbelt_profile(readable: list[Path], writable: Path, hidden: list[Path]) -> str:
    """macOS Seatbelt profile. A later rule wins, so the readable paths inside the
    hidden directories are allowed after the deny. Metadata stays readable
    everywhere; the interpreter cannot start without it. Seatbelt matches
    resolved paths."""

    def subpaths(paths: list[Path]) -> str:
        return " ".join(f'(subpath "{_quote(path.resolve())}")' for path in paths)

    return "\n".join(
        [
            "(version 1)",
            "(allow default)",
            "(deny network*)",
            f"(deny file-read* {subpaths(hidden)})",
            "(allow file-read-metadata)",
            f"(allow file-read* {subpaths(readable)})",
            "(deny file-write*)",
            f"(allow file-write* {subpaths([writable])})",
        ]
    )


def _quote(path: Path) -> str:
    return str(path).replace("\\", "\\\\").replace('"', '\\"')


def _bwrap_args(readable: list[Path], writable: Path) -> list[str]:
    """bubblewrap invocation: only the system directories and the readable paths
    are mounted; unsharing every namespace removes the network."""

    args = ["bwrap", "--unshare-all", "--die-with-parent", "--proc", "/proc", "--dev", "/dev"]
    args += ["--tmpfs", "/tmp"]
    for path in ("/usr", "/etc", "/bin", "/sbin", "/lib", "/lib64"):
        if os.path.islink(path):  # merged-usr distributions
            args += ["--symlink", os.readlink(path), path]
        elif os.path.isdir(path):
            args += ["--ro-bind", path, path]
    # Mount each path as spelled and as resolved: programs use the spelled path,
    # symlinks inside the mounts may use the resolved one.
    mounts = dict.fromkeys(p for path in readable for p in (path, path.resolve()))
    for path in mounts:
        args += ["--ro-bind", str(path), str(path)]
    args += ["--bind", str(writable), str(writable), "--chdir", str(writable)]
    return args
