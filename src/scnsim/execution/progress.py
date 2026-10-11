"""Transient user-facing display for Optimization progress.

Progress display is deliberately independent of durable progress callbacks.
It owns no numerical state and never changes a callback or storage outcome.
"""

from __future__ import annotations

import sys
import warnings
from dataclasses import dataclass
from html import escape
from typing import TextIO


@dataclass(frozen=True, slots=True)
class _ProgressState:
    phase: str
    completed_generations: int
    total_generations: int
    evaluated_count: int
    best_cost: float | None
    committed_generations: int | None
    reused: bool


class OptimizationProgressDisplay:
    """Render transient Optimization progress in notebooks or terminals.

    The caller supplies actual execution observations through :meth:`update`.
    Completed and committed generations are separate because a generation can
    finish in memory before its checkpoint group is durably acknowledged.
    """

    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled
        self._started = False
        self._closed = False
        self._disabled = not enabled
        self._warned = False
        self._state = _ProgressState(
            phase="preparing",
            completed_generations=0,
            total_generations=0,
            evaluated_count=0,
            best_cost=None,
            committed_generations=0,
            reused=False,
        )
        self._notebook_handle = None
        self._stream: TextIO | None = None
        self._terminal_tty = False

    def start(self) -> None:
        """Begin the display before Optimization preparation starts."""

        if self._started or self._closed or self._disabled:
            return
        self._started = True
        try:
            notebook = self._notebook_display()
            if notebook is not None:
                html_type, display = notebook
                self._notebook_handle = display(html_type(self._html_text()))
                return
            self._stream = sys.stderr
            self._terminal_tty = bool(self._stream.isatty())
            self._write_terminal()
        except Exception as error:
            self._disable(error)

    def update(
        self,
        *,
        phase: str,
        completed_generations: int,
        total_generations: int,
        evaluated_count: int,
        best_cost: float | None,
        committed_generations: int | None,
        reused: bool = False,
    ) -> None:
        """Display one real lifecycle update from the execution owner."""

        if self._closed or self._disabled:
            return
        self._state = _ProgressState(
            phase=phase,
            completed_generations=completed_generations,
            total_generations=total_generations,
            evaluated_count=evaluated_count,
            best_cost=best_cost,
            committed_generations=committed_generations,
            reused=reused,
        )
        if not self._started:
            self.start()
            return
        if self._disabled:
            return
        try:
            if self._notebook_handle is not None:
                html_type, _ = self._notebook_display()
                self._notebook_handle.update(html_type(self._html_text()))
            else:
                self._write_terminal()
        except Exception as error:
            self._disable(error)

    def close(self) -> None:
        """Finish the transient display without changing operation outcome."""

        if self._closed:
            return
        self._closed = True
        if self._disabled or not self._started:
            return
        try:
            if self._notebook_handle is not None:
                html_type, _ = self._notebook_display()
                self._notebook_handle.update(html_type(self._html_text()))
            elif self._stream is not None:
                if self._terminal_tty:
                    self._stream.write("\n")
                self._stream.flush()
        except Exception as error:
            self._disable(error)

    def _plain_text(self) -> str:
        state = self._state
        if state.phase == "preparing":
            return "Preparing optimization…"
        if state.phase == "baseline":
            return self._with_best(
                f"Baseline evaluated; 0/{state.total_generations} generations completed; "
                f"{state.evaluated_count} evaluations; {self._saved_text()}"
            )
        if state.phase == "failed":
            return "Optimization failed; no completion was reported"
        if state.phase == "interrupted":
            return "Optimization interrupted; no completion was reported"
        if state.phase == "resume":
            status = "Resuming from saved checkpoint"
        elif state.phase == "population_evaluating":
            status = "Evaluating population"
        elif state.phase == "complete":
            status = "Optimization finished"
        elif state.phase == "result_reuse" or state.reused:
            status = "Reused completed result"
        else:
            status = "Optimization progress"
        return self._with_best(
            f"{status}: {state.completed_generations}/{state.total_generations} "
            f"generations completed; {state.evaluated_count} evaluations; "
            f"{self._saved_text()}"
        )

    def _saved_text(self) -> str:
        count = self._state.committed_generations
        return "durable generation count unknown" if count is None else f"{count} generations saved"

    def _with_best(self, text: str) -> str:
        cost = self._state.best_cost
        if cost is None:
            return text
        return f"{text}; best cost {cost:.12g}"

    def _html_text(self) -> str:
        return (
            '<div class="scnsim-optimization-progress" role="status" '
            'aria-live="polite" style="font-family: sans-serif">'
            f"{escape(self._plain_text())}</div>"
        )

    def _write_terminal(self) -> None:
        if self._stream is None:
            self._stream = sys.stderr
            self._terminal_tty = bool(self._stream.isatty())
        prefix = "\r" if self._terminal_tty else ""
        suffix = "" if self._terminal_tty else "\n"
        self._stream.write(prefix + self._plain_text() + suffix)
        self._stream.flush()

    @staticmethod
    def _notebook_display():
        try:
            from IPython import get_ipython
            from IPython.display import HTML, display
        except ImportError:
            return None
        shell = get_ipython()
        if shell is None or not (
            hasattr(shell, "kernel") or type(shell).__name__ == "ZMQInteractiveShell"
        ):
            return None
        return HTML, lambda value: display(value, display_id=True)

    def _disable(self, error: Exception) -> None:
        self._disabled = True
        if self._stream is not None and self._terminal_tty:
            try:
                self._stream.write("\n")
                self._stream.flush()
            except Exception:
                pass
        if self._warned:
            return
        self._warned = True
        try:
            warnings.warn(
                f"Optimization progress display disabled after {type(error).__name__}",
                RuntimeWarning,
                stacklevel=2,
            )
        except Exception:
            # A warning filter must not replace the numerical or user-callback
            # outcome that the transient display is observing.
            pass

