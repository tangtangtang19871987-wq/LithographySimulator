"""Dependency-free desktop dashboard for supervising interactive CLI processes.

The process layer deliberately has no dependency on tkinter.  It can therefore
be tested headlessly or reused by a web/TUI front end.
"""

from __future__ import annotations

import codecs
import json
import os
import queue
import re
import shlex
import signal
import subprocess
import tempfile
import threading
import tkinter as tk
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Callable, TextIO


ANSI_ESCAPE = re.compile(r"\x1b(?:[@-_]|\[[0-?]*[ -/]*[@-~])")
PROGRESS = re.compile(r"(?<!\d)(100|[1-9]?\d)(?:\.\d+)?\s*%")
DEFAULT_CONFIG = Path.home() / ".config" / "copilot-dashboard" / "profiles.json"


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
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
            raise ValueError(f"line {line_number}: invalid environment name")
        result[key] = value
    return result


class SessionState(str, Enum):
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    EXITED = "exited"
    FAILED = "failed"


@dataclass
class SessionConfig:
    name: str
    command: str
    cwd: str = field(default_factory=os.getcwd)
    environment: dict[str, str] = field(default_factory=dict)

    def validate(self) -> None:
        if not self.name.strip():
            raise ValueError("session name cannot be empty")
        if not self.command.strip():
            raise ValueError("command cannot be empty")
        if not Path(self.cwd).is_dir():
            raise ValueError(f"working directory does not exist: {self.cwd}")


class ProfileStore:
    """Atomically persist reusable session configurations as JSON."""

    def __init__(self, path: Path = DEFAULT_CONFIG):
        self.path = path

    def load(self) -> list[SessionConfig]:
        if not self.path.exists():
            return []
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
            if not isinstance(payload, list):
                raise ValueError("profile file must contain a list")
            profiles = [SessionConfig(**item) for item in payload]
            for profile in profiles:
                profile.validate()
            return profiles
        except (OSError, TypeError, json.JSONDecodeError, ValueError) as error:
            raise ValueError(f"cannot load {self.path}: {error}") from error

    def save(self, profiles: list[SessionConfig]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text(
            json.dumps([asdict(profile) for profile in profiles], indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        if os.name == "posix":
            temporary.chmod(0o600)
        os.replace(temporary, self.path)


class CliSession:
    """Own one child process and safely stream its events from a worker thread."""

    def __init__(self, config: SessionConfig, emit: Callable[[str, object], None]):
        self.config = config
        self.emit = emit
        self.process: subprocess.Popen[bytes] | None = None
        self.state = SessionState.STOPPED
        self._master_fd: int | None = None
        self._lock = threading.RLock()

    @property
    def running(self) -> bool:
        return self.process is not None and self.process.poll() is None

    def _set_state(self, state: SessionState, detail: str = "") -> None:
        self.state = state
        self.emit("state", (state, detail))

    def start(self) -> None:
        with self._lock:
            if self.running:
                raise RuntimeError("session is already running")
            self.config.validate()
            args = shlex.split(self.config.command, posix=os.name != "nt")
            environment = os.environ.copy()
            environment.update(self.config.environment)
            options = {"cwd": self.config.cwd, "env": environment}
            self._set_state(SessionState.STARTING)
            try:
                if os.name == "posix":
                    master, slave = os.openpty()
                    self._master_fd = master
                    try:
                        self.process = subprocess.Popen(
                            args, stdin=slave, stdout=slave, stderr=slave,
                            start_new_session=True, **options,
                        )
                    finally:
                        os.close(slave)
                    target = self._read_pty
                else:
                    self.process = subprocess.Popen(
                        args, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP, **options,
                    )
                    target = self._read_pipe
            except OSError as error:
                if self._master_fd is not None:
                    os.close(self._master_fd)
                    self._master_fd = None
                self._set_state(SessionState.FAILED, str(error))
                raise
            assert self.process
            self._set_state(SessionState.RUNNING, f"pid {self.process.pid}")
            threading.Thread(target=target, daemon=True, name=f"cli-{self.config.name}").start()

    def _read_pty(self) -> None:
        assert self._master_fd is not None
        decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        try:
            while True:
                try:
                    chunk = os.read(self._master_fd, 4096)
                except OSError:
                    break
                if not chunk:
                    break
                text = decoder.decode(chunk)
                if text:
                    self.emit("output", text)
            tail = decoder.decode(b"", final=True)
            if tail:
                self.emit("output", tail)
        finally:
            with self._lock:
                if self._master_fd is not None:
                    os.close(self._master_fd)
                    self._master_fd = None
            self._finish()

    def _read_pipe(self) -> None:
        assert self.process and self.process.stdout
        decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        while chunk := self.process.stdout.read(4096):
            self.emit("output", decoder.decode(chunk))
        tail = decoder.decode(b"", final=True)
        if tail:
            self.emit("output", tail)
        self._finish()

    def _finish(self) -> None:
        assert self.process
        code = self.process.wait()
        self.emit("exit", code)
        self._set_state(SessionState.EXITED, f"exit code {code}")

    def send(self, text: str) -> None:
        with self._lock:
            if not self.running:
                raise RuntimeError("session is not running")
            data = text.encode()
            if self._master_fd is not None:
                os.write(self._master_fd, data)
            else:
                assert self.process and self.process.stdin
                self.process.stdin.write(data)
                self.process.stdin.flush()

    def stop(self, grace_seconds: float = 2.0) -> None:
        """Request graceful termination, then kill after a bounded grace period."""
        with self._lock:
            if not self.running:
                return
            assert self.process
            self._set_state(SessionState.STOPPING)
            if os.name == "posix":
                os.killpg(self.process.pid, signal.SIGTERM)
            else:
                self.process.terminate()
            process = self.process

        def kill_later() -> None:
            try:
                process.wait(timeout=grace_seconds)
            except subprocess.TimeoutExpired:
                if os.name == "posix":
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                else:
                    process.kill()

        threading.Thread(target=kill_later, daemon=True).start()


@dataclass
class SessionView:
    session: CliSession
    page: ttk.Frame
    output: tk.Text
    status: ttk.Label
    progress: ttk.Progressbar
    entry: ttk.Entry
    log: TextIO = field(
        default_factory=lambda: tempfile.SpooledTemporaryFile(
            mode="w+", encoding="utf-8", max_size=1024 * 1024
        )
    )


class Dashboard(tk.Tk):
    """Tk dashboard that marshals worker-thread events onto the UI thread."""

    MAX_OUTPUT_LINES = 5000

    def __init__(self, store: ProfileStore | None = None) -> None:
        super().__init__()
        self.title("Copilot CLI Dashboard")
        self.geometry("1180x760")
        self.minsize(840, 560)
        self.store = store or ProfileStore()
        self.events: queue.Queue[tuple[str, str, object]] = queue.Queue()
        self.sessions: dict[str, SessionView] = {}
        self.profiles: dict[str, SessionConfig] = {}
        self._closing = False
        self._build()
        self._load_profiles()
        self.after(50, self._drain_events)
        self.protocol("WM_DELETE_WINDOW", self._close)

    def _build(self) -> None:
        form = ttk.LabelFrame(self, text="Session configuration", padding=10)
        form.pack(fill="x", padx=10, pady=10)
        self.name = tk.StringVar(value="Copilot 1")
        self.command = tk.StringVar(value="copilot")
        self.cwd = tk.StringVar(value=os.getcwd())
        self.profile_name = tk.StringVar()
        ttk.Label(form, text="Saved profile").grid(row=0, column=0, sticky="w")
        self.profile_box = ttk.Combobox(form, textvariable=self.profile_name, state="readonly")
        self.profile_box.grid(row=0, column=1, sticky="ew", pady=3)
        self.profile_box.bind("<<ComboboxSelected>>", self._select_profile)
        for row, (label, variable) in enumerate(
            (("Name", self.name), ("Command", self.command), ("Working directory", self.cwd)), 1
        ):
            ttk.Label(form, text=label).grid(row=row, column=0, sticky="w", padx=(0, 8), pady=3)
            ttk.Entry(form, textvariable=variable).grid(row=row, column=1, sticky="ew", pady=3)
        ttk.Button(form, text="Browse…", command=self._choose_cwd).grid(row=3, column=2, padx=5)
        ttk.Label(form, text="Environment\n(KEY=VALUE)").grid(row=0, column=3, rowspan=4, sticky="nw", padx=(18, 8))
        self.environment = tk.Text(form, height=6, width=34, undo=True)
        self.environment.grid(row=0, column=4, rowspan=4, sticky="nsew")
        buttons = ttk.Frame(form)
        buttons.grid(row=0, column=5, rowspan=4, padx=(12, 0))
        ttk.Button(buttons, text="Start session", command=self._add_session).pack(fill="x", pady=2)
        ttk.Button(buttons, text="Save profile", command=self._save_profile).pack(fill="x", pady=2)
        ttk.Button(buttons, text="Delete profile", command=self._delete_profile).pack(fill="x", pady=2)
        form.columnconfigure(1, weight=1)
        form.columnconfigure(4, weight=1)
        self.tabs = ttk.Notebook(self)
        self.tabs.pack(fill="both", expand=True, padx=10, pady=(0, 10))

    def _configuration(self) -> SessionConfig:
        return SessionConfig(
            self.name.get().strip(), self.command.get().strip(), self.cwd.get().strip(),
            parse_environment(self.environment.get("1.0", "end")),
        )

    def _load_profiles(self) -> None:
        try:
            self.profiles = {profile.name: profile for profile in self.store.load()}
            self._refresh_profiles()
        except ValueError as error:
            messagebox.showwarning("Profiles not loaded", str(error))

    def _refresh_profiles(self) -> None:
        self.profile_box.configure(values=sorted(self.profiles))

    def _select_profile(self, _event: object = None) -> None:
        profile = self.profiles.get(self.profile_name.get())
        if not profile:
            return
        self.name.set(profile.name)
        self.command.set(profile.command)
        self.cwd.set(profile.cwd)
        self.environment.delete("1.0", "end")
        self.environment.insert("1.0", "\n".join(f"{key}={value}" for key, value in profile.environment.items()))

    def _save_profile(self) -> None:
        try:
            config = self._configuration()
            config.validate()
            self.profiles[config.name] = config
            self.store.save(list(self.profiles.values()))
            self._refresh_profiles()
            self.profile_name.set(config.name)
        except (OSError, ValueError) as error:
            messagebox.showerror("Could not save profile", str(error))

    def _delete_profile(self) -> None:
        name = self.profile_name.get()
        if name and name in self.profiles:
            del self.profiles[name]
            try:
                self.store.save(list(self.profiles.values()))
            except OSError as error:
                messagebox.showerror("Could not delete profile", str(error))
                return
            self.profile_name.set("")
            self._refresh_profiles()

    def _choose_cwd(self) -> None:
        selected = filedialog.askdirectory(initialdir=self.cwd.get())
        if selected:
            self.cwd.set(selected)

    def _add_session(self) -> None:
        try:
            config = self._configuration()
            config.validate()
            if config.name in self.sessions:
                raise ValueError("choose a unique session name")
            page = ttk.Frame(self.tabs, padding=8)
            header = ttk.Frame(page)
            header.pack(fill="x")
            status = ttk.Label(header, text="Starting…")
            status.pack(side="left")
            progress = ttk.Progressbar(header, length=180, mode="indeterminate")
            progress.pack(side="right")
            progress.start(12)
            output = tk.Text(page, wrap="word", state="disabled", bg="#111827", fg="#e5e7eb", insertbackground="white")
            output.pack(fill="both", expand=True, pady=6)
            input_row = ttk.Frame(page)
            input_row.pack(fill="x")
            entry = ttk.Entry(input_row)
            entry.pack(side="left", fill="x", expand=True)
            name = config.name
            session = CliSession(config, lambda kind, data: self.events.put((name, kind, data)))
            view = SessionView(session, page, output, status, progress, entry)
            self.sessions[name] = view
            entry.bind("<Return>", lambda _event: self._send(view))
            ttk.Button(input_row, text="Send", command=lambda: self._send(view)).pack(side="left", padx=4)
            ttk.Button(input_row, text="Stop", command=session.stop).pack(side="left", padx=4)
            ttk.Button(input_row, text="Restart", command=lambda: self._restart(view)).pack(side="left", padx=4)
            ttk.Button(input_row, text="Clear", command=lambda: self._clear(view)).pack(side="left", padx=4)
            ttk.Button(input_row, text="Save log…", command=lambda: self._save_log(view)).pack(side="left", padx=4)
            ttk.Button(input_row, text="Close", command=lambda: self._close_session(name)).pack(side="left")
            self.tabs.add(page, text=name)
            self.tabs.select(page)
            session.start()
        except (OSError, ValueError, RuntimeError) as error:
            if "config" in locals() and config.name in self.sessions:
                self._remove_session(config.name)
            messagebox.showerror("Could not start CLI", str(error))

    def _send(self, view: SessionView) -> None:
        value = view.entry.get()
        if not value:
            return
        try:
            view.session.send(value + "\n")
            view.entry.delete(0, "end")
        except (OSError, RuntimeError) as error:
            messagebox.showerror("Could not send input", str(error))

    def _restart(self, view: SessionView) -> None:
        if view.session.running:
            messagebox.showinfo("Session is running", "Stop the session before restarting it.")
            return
        try:
            view.session.start()
            view.entry.focus_set()
        except (OSError, ValueError, RuntimeError) as error:
            messagebox.showerror("Could not restart CLI", str(error))

    def _clear(self, view: SessionView) -> None:
        view.output.configure(state="normal")
        view.output.delete("1.0", "end")
        view.output.configure(state="disabled")
        view.log.seek(0)
        view.log.truncate()

    def _save_log(self, view: SessionView) -> None:
        path = filedialog.asksaveasfilename(defaultextension=".log", filetypes=(("Log files", "*.log"), ("All files", "*")))
        if path:
            try:
                view.log.flush()
                view.log.seek(0)
                with Path(path).open("w", encoding="utf-8") as destination:
                    while chunk := view.log.read(64 * 1024):
                        destination.write(chunk)
                view.log.seek(0, os.SEEK_END)
            except OSError as error:
                messagebox.showerror("Could not save log", str(error))

    def _append_output(self, view: SessionView, data: str) -> None:
        clean = ANSI_ESCAPE.sub("", data).replace("\r\n", "\n")
        view.log.write(clean)
        view.output.configure(state="normal")
        view.output.insert("end", clean)
        line_count = int(view.output.index("end-1c").split(".")[0])
        if line_count > self.MAX_OUTPUT_LINES:
            view.output.delete("1.0", f"{line_count - self.MAX_OUTPUT_LINES}.0")
        view.output.see("end")
        view.output.configure(state="disabled")
        matches = list(PROGRESS.finditer(clean))
        if matches:
            view.progress.stop()
            view.progress.configure(mode="determinate", value=float(matches[-1].group(1)))

    def _drain_events(self) -> None:
        while True:
            try:
                name, kind, data = self.events.get_nowait()
            except queue.Empty:
                break
            view = self.sessions.get(name)
            if not view:
                continue
            if kind == "output":
                self._append_output(view, str(data))
            elif kind == "state":
                state, detail = data
                view.status.configure(text=f"{state.value.title()}{': ' + detail if detail else ''}")
                if state in (SessionState.EXITED, SessionState.FAILED):
                    view.progress.stop()
            if kind == "exit" and int(data) != 0:
                view.status.configure(foreground="#b91c1c")
            elif kind == "state" and data[0] == SessionState.RUNNING:
                view.status.configure(foreground="#15803d")
        if not self._closing:
            self.after(50, self._drain_events)

    def _close_session(self, name: str) -> None:
        view = self.sessions[name]
        if view.session.running:
            view.session.stop()
        self._remove_session(name)

    def _remove_session(self, name: str) -> None:
        view = self.sessions.pop(name)
        self.tabs.forget(view.page)
        view.page.destroy()
        view.log.close()

    def _close(self) -> None:
        self._closing = True
        for view in self.sessions.values():
            view.session.stop()
            view.log.close()
        self.destroy()


def main() -> None:
    Dashboard().mainloop()


if __name__ == "__main__":
    main()
