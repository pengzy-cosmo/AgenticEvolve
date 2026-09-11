"""Temp-dir subprocess execution for analysis runs and formal evaluation.

Each execution gets a fresh temp directory and its own process group; timeout
and task cancellation kill the whole group. This module is the isolation
boundary: replace it to add system-level sandboxing.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import signal
import sys
import tempfile
from pathlib import Path

from agentic_evolve.config import ExecutorConfig

_RUNNER_PATH = Path(__file__).parent / "_eval_runner.py"
METRICS_FILENAME = "metrics.json"


class EvaluationError(Exception):
    """Formal evaluation failed: crash, timeout, or invalid metrics output."""

    def __init__(self, message: str, status: str = "error"):
        super().__init__(message)
        self.status = status


class Executor:
    def __init__(
        self,
        cfg: ExecutorConfig,
        evaluator_path: Path,
        data: Path | None = None,
    ):
        self.cfg = cfg
        self.evaluator_path = Path(evaluator_path).resolve()
        self.data = Path(data).resolve() if data else None

    async def run_analysis(self, program: str, analysis: str) -> tuple[str, str]:
        """Run analysis.py next to program.py; return (status, truncated output text)."""

        tmp = Path(tempfile.mkdtemp(prefix="evolve-analysis-"))
        try:
            (tmp / "program.py").write_text(program)
            (tmp / "analysis.py").write_text(analysis)
            if self.data is not None:
                shutil.copytree(self.data, tmp / self.data.name)
            code, out, err, timed_out = await self._run([sys.executable, "analysis.py"], tmp)
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
            metrics_path = tmp / METRICS_FILENAME
            cmd = [
                sys.executable,
                str(_RUNNER_PATH),
                str(self.evaluator_path),
                str(tmp / "program.py"),
                str(metrics_path),
            ]
            code, out, err, timed_out = await self._run(cmd, tmp)
            if timed_out:
                raise EvaluationError(
                    f"evaluation timed out after {self.cfg.timeout}s", status="timeout"
                )
            if code != 0:
                raise EvaluationError(
                    f"evaluator exited with code {code}\nstdout:\n{out}\nstderr:\n{err}"
                )
            if not metrics_path.exists():
                raise EvaluationError(
                    f"evaluator wrote no metrics file\nstdout:\n{out}\nstderr:\n{err}"
                )
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

    async def _run(self, cmd: list[str], cwd: Path) -> tuple[int | None, str, str, bool]:
        """Run one subprocess in its own process group; kill the group on timeout or cancel.

        HOME and the temp-dir variables point into the ephemeral working
        directory; BLAS/OpenMP threads are pinned to avoid oversubscription.
        """

        env = os.environ | {key: str(cwd) for key in ("HOME", "TMPDIR", "TMP", "TEMP")}
        env |= {
            key: "1"
            for key in (
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS",
            )
        }
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=cwd,
            env=env,
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

    def _truncate(self, text: str) -> str:
        if len(text) <= self.cfg.output_limit:
            return text
        return text[: self.cfg.output_limit] + f"\n... (truncated, {len(text)} chars total)"

    @staticmethod
    async def _kill_group(proc: asyncio.subprocess.Process) -> None:
        # start_new_session=True makes the child a session leader, so its pid is the pgid.
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        await proc.wait()
