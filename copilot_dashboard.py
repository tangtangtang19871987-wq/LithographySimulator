"""A small, dependency-free dashboard for supervising multiple CLI processes.

The user interface is deliberately built with tkinter (part of CPython).  The
process controller is independent from tkinter so it can also be embedded in a
different Python UI later.
"""

from __future__ import annotations

import os
import queue
import re
import shlex
import signal
import subprocess
import threading
import tkinter as tk
from dataclasses import dataclass, field
from tkinter import messagebox, ttk
from typing import Callable


ANSI_ESCAPE = re.compile(r"\x1b(?:[@-_]|\[[0-?]*[ -/]*[@-~])")


def parse_environment(text: str) -> dict[str, str]:
    """Parse a ``KEY=VALUE`` editor, ignoring blank and comment lines."""
    result: dict[str, str] = {}
    for line_number, raw_line in enumerate(text.splitlines(), 1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            raise ValueError(f"line {line_number}: expected KEY=VALUE")
        key, value = line.split("=", 1)
        key = key.strip()
        if not key or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
            raise ValueError(f"line {line_number}: invalid environment name")
        result[key] = value
    return result


@dataclass
class SessionConfig:
    name: str
    command: str
    cwd: str = field(default_factory=os.getcwd)
    environment: dict[str, str] = field(default_factory=dict)


class CliSession:
    """Own one child process and stream its events without blocking the GUI."""

    def __init__(self, config: SessionConfig, emit: Callable[[str, str], None]):
        self.config = config
        self.emit = emit
        self.process: subprocess.Popen[bytes] | None = None
        self._master_fd: int | None = None

    @property
    def running(self) -> bool:
        return self.process is not None and self.process.poll() is None

    def start(self) -> None:
        if self.running:
            raise RuntimeError("session is already running")
        args = shlex.split(self.config.command, posix=os.name != "nt")
        if not args:
            raise ValueError("command cannot be empty")
        environment = os.environ.copy()
        environment.update(self.config.environment)
        options = dict(cwd=self.config.cwd, env=environment)

        if os.name == "posix":
            master, slave = os.openpty()
            self._master_fd = master
            self.process = subprocess.Popen(
                args, stdin=slave, stdout=slave, stderr=slave,
                start_new_session=True, **options
            )
            os.close(slave)
            target = self._read_pty
        else:  # Windows pipe fallback; ConPTY can be added behind this API.
            self.process = subprocess.Popen(
                args, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
                **options
            )
            target = self._read_pipe
        self.emit("status", f"running (pid {self.process.pid})")
        threading.Thread(target=target, daemon=True, name=f"cli-{self.config.name}").start()

    def _read_pty(self) -> None:
        assert self._master_fd is not None
        try:
            while True:
                try:
                    chunk = os.read(self._master_fd, 4096)
                except OSError:
                    break
                if not chunk:
                    break
                self.emit("output", chunk.decode(errors="replace"))
        finally:
            os.close(self._master_fd)
            self._master_fd = None
            self._finish()

    def _read_pipe(self) -> None:
        assert self.process and self.process.stdout
        while chunk := self.process.stdout.read(4096):
            self.emit("output", chunk.decode(errors="replace"))
        self._finish()

    def _finish(self) -> None:
        assert self.process
        code = self.process.wait()
        self.emit("status", f"exited ({code})")

    def send(self, text: str) -> None:
        if not self.running:
            raise RuntimeError("session is not running")
        data = text.encode()
        if self._master_fd is not None:
            os.write(self._master_fd, data)
        else:
            assert self.process and self.process.stdin
            self.process.stdin.write(data)
            self.process.stdin.flush()

    def stop(self) -> None:
        if not self.running:
            return
        assert self.process
        if os.name == "posix":
            os.killpg(self.process.pid, signal.SIGTERM)
        else:
            self.process.terminate()


class Dashboard(tk.Tk):
    """Tk dashboard that marshals worker-thread events onto the UI thread."""

    def __init__(self) -> None:
        super().__init__()
        self.title("Copilot CLI Dashboard")
        self.geometry("1100x720")
        self.minsize(800, 520)
        self.events: queue.Queue[tuple[CliSession, str, str]] = queue.Queue()
        self.sessions: dict[str, tuple[CliSession, tk.Text, ttk.Label]] = {}
        self._build()
        self.after(50, self._drain_events)
        self.protocol("WM_DELETE_WINDOW", self._close)

    def _build(self) -> None:
        form = ttk.LabelFrame(self, text="New CLI session", padding=10)
        form.pack(fill="x", padx=10, pady=10)
        self.name = tk.StringVar(value="Copilot 1")
        self.command = tk.StringVar(value="copilot")
        self.cwd = tk.StringVar(value=os.getcwd())
        for row, (label, variable) in enumerate((("Name", self.name), ("Command", self.command), ("Working directory", self.cwd))):
            ttk.Label(form, text=label).grid(row=row, column=0, sticky="w", padx=(0, 8), pady=3)
            ttk.Entry(form, textvariable=variable).grid(row=row, column=1, sticky="ew", pady=3)
        ttk.Label(form, text="Environment\n(KEY=VALUE)").grid(row=0, column=2, rowspan=3, sticky="nw", padx=(18, 8))
        self.environment = tk.Text(form, height=4, width=34)
        self.environment.grid(row=0, column=3, rowspan=3, sticky="nsew")
        ttk.Button(form, text="Start session", command=self._add_session).grid(row=0, column=4, rowspan=3, padx=(12, 0))
        form.columnconfigure(1, weight=1)
        form.columnconfigure(3, weight=1)
        self.tabs = ttk.Notebook(self)
        self.tabs.pack(fill="both", expand=True, padx=10, pady=(0, 10))

    def _add_session(self) -> None:
        name = self.name.get().strip()
        if not name or name in self.sessions:
            messagebox.showerror("Invalid session", "Choose a unique, non-empty name.")
            return
        try:
            env = parse_environment(self.environment.get("1.0", "end"))
            config = SessionConfig(name, self.command.get(), self.cwd.get(), env)
            page = ttk.Frame(self.tabs, padding=8)
            status = ttk.Label(page, text="starting…")
            status.pack(anchor="w")
            output = tk.Text(page, wrap="word", state="disabled", bg="#111827", fg="#e5e7eb", insertbackground="white")
            output.pack(fill="both", expand=True, pady=6)
            input_row = ttk.Frame(page)
            input_row.pack(fill="x")
            entry = ttk.Entry(input_row)
            entry.pack(side="left", fill="x", expand=True)
            session = CliSession(config, lambda kind, data: self.events.put((session, kind, data)))
            entry.bind("<Return>", lambda _event: self._send(session, entry))
            ttk.Button(input_row, text="Send", command=lambda: self._send(session, entry)).pack(side="left", padx=6)
            ttk.Button(input_row, text="Stop", command=session.stop).pack(side="left")
            self.sessions[name] = session, output, status
            self.tabs.add(page, text=name)
            self.tabs.select(page)
            session.start()
        except (OSError, ValueError, RuntimeError) as error:
            if name in self.sessions:
                del self.sessions[name]
                self.tabs.forget(page)
            messagebox.showerror("Could not start CLI", str(error))

    def _send(self, session: CliSession, entry: ttk.Entry) -> None:
        value = entry.get()
        if value:
            try:
                session.send(value + "\n")
                entry.delete(0, "end")
            except RuntimeError as error:
                messagebox.showerror("Could not send input", str(error))

    def _drain_events(self) -> None:
        while True:
            try:
                session, kind, data = self.events.get_nowait()
            except queue.Empty:
                break
            _, output, status = self.sessions[session.config.name]
            if kind == "status":
                status.configure(text=data)
            else:
                output.configure(state="normal")
                output.insert("end", ANSI_ESCAPE.sub("", data))
                output.see("end")
                output.configure(state="disabled")
        self.after(50, self._drain_events)

    def _close(self) -> None:
        for session, _, _ in self.sessions.values():
            session.stop()
        self.destroy()


def main() -> None:
    Dashboard().mainloop()


if __name__ == "__main__":
    main()
