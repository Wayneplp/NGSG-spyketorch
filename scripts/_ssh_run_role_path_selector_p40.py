#!/usr/bin/env python3
"""Deploy and run Single Role-path Selector on the P40 rental host.

Password via env NGSG_P40_PASSWORD (never hardcode in git).
"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import paramiko

HOST = "121.11.192.245"
PORT = 57229
USER = "root"
REPO_CANDIDATES = [
    "/root/autodl-tmp/NGSG-spyketorch-4a958ae",
    "/root/autodl-tmp/NGSG-spyketorch",
    "/root/NGSG-spyketorch",
]
ROOT = Path(__file__).resolve().parents[1]

UPLOAD_PATHS = [
    "src/analysis/role_path_selector.py",
    "src/analysis/switch_selector.py",
    "src/continual/readout.py",
    "src/continual/neuron_partition.py",
    "src/continual/reserve_activation.py",
    "src/models/paper_mozafari.py",
    "scripts/eval_single_role_path_selector.py",
    "scripts/eval_partition_group_diagnosis.py",
    "scripts/eval_method_v1_routing.py",
    "tests/test_role_path_selector.py",
]


def password() -> str:
    value = os.environ.get("NGSG_P40_PASSWORD", "").strip()
    if not value:
        raise RuntimeError("Set NGSG_P40_PASSWORD in the environment before connecting.")
    return value


def connect() -> paramiko.SSHClient:
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(HOST, port=PORT, username=USER, password=password(), timeout=30)
    return client


def run(client: paramiko.SSHClient, cmd: str, timeout: int = 300) -> tuple[int, str, str]:
    _, stdout, stderr = client.exec_command(cmd, timeout=timeout)
    out = stdout.read().decode("utf-8", errors="replace")
    err = stderr.read().decode("utf-8", errors="replace")
    return stdout.channel.recv_exit_status(), out, err


def checked(client: paramiko.SSHClient, cmd: str, timeout: int = 300) -> str:
    code, out, err = run(client, cmd, timeout=timeout)
    if code:
        raise RuntimeError(f"remote failed ({code}):\n{out}\n{err}")
    return out


def _write_remote_text(sftp: paramiko.SFTPClient, remote: str, text: str) -> None:
    payload = text.replace("\r\n", "\n").replace("\r", "\n").encode("utf-8")
    with sftp.file(remote, "wb") as handle:
        handle.write(payload)


def resolve_repo(client: paramiko.SSHClient) -> str:
    for path in REPO_CANDIDATES:
        code, out, _ = run(client, f"test -d {path} && echo OK")
        if code == 0 and "OK" in out:
            return path
    # Fall back: create under autodl-tmp if present, else /root
    code, _, _ = run(client, "test -d /root/autodl-tmp && echo OK")
    base = "/root/autodl-tmp" if code == 0 else "/root"
    repo = f"{base}/NGSG-spyketorch"
    checked(client, f"mkdir -p {repo}")
    return repo


def probe(client: paramiko.SSHClient) -> None:
    print("=== host ===")
    print(checked(client, "hostname; uname -a; date"))
    print("=== gpu ===")
    code, out, err = run(client, "nvidia-smi -L; nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv")
    print(out or err)
    print("=== python / conda ===")
    code, out, err = run(
        client,
        "source /root/miniconda3/etc/profile.d/conda.sh 2>/dev/null; "
        "conda env list 2>/dev/null | head; "
        "which python; python -c 'import torch; print(torch.__version__, torch.cuda.is_available())' 2>&1 | tail -5",
    )
    print(out or err)
    print("=== disks / repos ===")
    print(checked(client, "ls -la /root/autodl-tmp 2>/dev/null | head -40; ls -la /root 2>/dev/null | head -40"))
    repo = resolve_repo(client)
    print(f"REPO={repo}")
    code, out, err = run(
        client,
        f"find {repo}/experiments -name 'model_after_task2.pt' 2>/dev/null | head -40; "
        f"find {repo}/experiments -path '*role_train*' -name 'result.json' 2>/dev/null | head -40",
    )
    print("checkpoints:")
    print(out or "(none)")


def deploy(client: paramiko.SSHClient, repo: str) -> None:
    checked(
        client,
        f"mkdir -p {repo}/src/analysis {repo}/src/continual {repo}/src/models "
        f"{repo}/scripts {repo}/tests {repo}/logs {repo}/diagnostics",
    )
    sftp = client.open_sftp()
    try:
        for rel in UPLOAD_PATHS:
            local = ROOT / rel
            if not local.exists():
                print(f"skip missing {rel}")
                continue
            remote = f"{repo}/{rel.replace(chr(92), '/')}"
            remote_dir = "/".join(remote.split("/")[:-1])
            checked(client, f"mkdir -p {remote_dir}")
            _write_remote_text(sftp, remote, local.read_text(encoding="utf-8"))
            print(f"uploaded {remote}")
    finally:
        sftp.close()
    print(
        checked(
            client,
            f"cd {repo}; "
            "source /root/miniconda3/etc/profile.d/conda.sh 2>/dev/null; "
            "conda activate ngsg 2>/dev/null || true; "
            "python -m py_compile src/analysis/role_path_selector.py scripts/eval_single_role_path_selector.py && "
            "python -m unittest tests.test_role_path_selector -v",
        )
    )


def find_role_train_run(client: paramiko.SSHClient, repo: str) -> str | None:
    code, out, _ = run(
        client,
        f"python - <<'PY'\n"
        f"from pathlib import Path\n"
        f"root = Path('{repo}') / 'experiments'\n"
        f"cands = []\n"
        f"if root.exists():\n"
        f"    for p in root.rglob('model_after_task2.pt'):\n"
        f"        run = p.parent.parent\n"
        f"        if (run / 'result.json').exists() and (run / 'resolved_config.json').exists():\n"
        f"            name = run.name.lower()\n"
        f"            score = (2 if 'role_train' in name else 0) + (1 if 'neurocomputing' in name else 0)\n"
        f"            cands.append((score, run.stat().st_mtime, str(run)))\n"
        f"cands.sort(reverse=True)\n"
        f"print(cands[0][2] if cands else '')\n"
        f"PY",
    )
    path = (out or "").strip().splitlines()
    path = path[-1].strip() if path else ""
    return path or None


def launch(client: paramiko.SSHClient, repo: str, run_dir: str, session: str = "rolepath") -> None:
    log = f"{repo}/logs/single_role_path_selector_p40.log"
    cmd = (
        f"cd {repo} && "
        "source /root/miniconda3/etc/profile.d/conda.sh && conda activate ngsg && "
        "python -c 'import torch; assert torch.cuda.is_available(), \"no cuda\"' && "
        f"python scripts/eval_single_role_path_selector.py "
        f"--run-dir {run_dir} --device cuda --selector-seed 0 "
        f"--validation-samples-per-task 2000 "
        f"--output {run_dir}/diagnostics/single_role_path_selector.json"
    )
    checked(client, f"tmux has-session -t {session} 2>/dev/null && tmux kill-session -t {session} || true")
    checked(
        client,
        f"tmux new-session -d -s {session} bash -lc {repr(cmd + f' 2>&1 | tee {log}; echo EXIT:$? | tee -a {log}')}",
    )
    print(f"launched tmux session={session} log={log}")


def status(client: paramiko.SSHClient, repo: str, run_dir: str | None = None) -> None:
    print(checked(client, "tmux ls 2>/dev/null || echo no-tmux"))
    print(checked(client, f"tail -n 40 {repo}/logs/single_role_path_selector_p40.log 2>/dev/null || echo no-log"))
    if run_dir:
        code, out, _ = run(client, f"test -f {run_dir}/diagnostics/single_role_path_selector.json && echo READY")
        if "READY" in out:
            print(checked(client, f"python -m json.tool {run_dir}/diagnostics/single_role_path_selector.json | head -120"))


def fetch(client: paramiko.SSHClient, run_dir: str, local_out: Path) -> None:
    remote = f"{run_dir}/diagnostics/single_role_path_selector.json"
    local_out.parent.mkdir(parents=True, exist_ok=True)
    sftp = client.open_sftp()
    try:
        sftp.get(remote, str(local_out))
    finally:
        sftp.close()
    print(f"fetched -> {local_out}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--action", choices=["probe", "deploy", "launch", "status", "fetch", "all"], default="all")
    parser.add_argument("--run-dir", default="")
    parser.add_argument("--session", default="rolepath")
    parser.add_argument("--fetch-to", type=Path, default=ROOT / "results" / "single_role_path_selector_p40.json")
    args = parser.parse_args()

    client = connect()
    try:
        repo = resolve_repo(client)
        if args.action in {"probe", "all"}:
            probe(client)
        if args.action in {"deploy", "all"}:
            deploy(client, repo)
        run_dir = args.run_dir.strip() or find_role_train_run(client, repo)
        if args.action in {"launch", "all"}:
            if not run_dir:
                raise RuntimeError(
                    "No role_train checkpoint found under experiments/. "
                    "Train role_train first or pass --run-dir."
                )
            print(f"using run_dir={run_dir}")
            launch(client, repo, run_dir, session=args.session)
        if args.action in {"status", "all"}:
            # brief wait then status
            if args.action == "all":
                time.sleep(5)
            status(client, repo, run_dir or None)
        if args.action == "fetch":
            if not run_dir:
                raise RuntimeError("--run-dir required for fetch when auto-detect fails")
            fetch(client, run_dir, args.fetch_to)
    finally:
        client.close()


if __name__ == "__main__":
    main()
