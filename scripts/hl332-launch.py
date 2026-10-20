#!/usr/bin/env python3
"""Bind a create-only static dry-launch receipt, then launch that exact Fab argv."""
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess

DEFAULT_IMAGE = "nvidia/cuda:12.6.3-devel-ubuntu24.04"
DEFAULT_CORES = 8
DEFAULT_MEMORY = "16g"
DEFAULT_WALL_SECONDS = 7200
DEFAULT_ESTIMATE_GIB = "32"
SCRIPT_INPUTS = ("scripts/hl332-build.sh", "scripts/hl332-container.sh", "scripts/hl332-launch.py")
ARTIFACTS = ("result.txt", "source-tree.txt", "source-commit.txt", "toolchain.txt", "image-digests.json", "cpu-controls.log", "cpu-build.log", "cpu-configure.log", "mixed-configure.log", "mixed-build.log", "container.log", "absent-backend.log", "zero-cases.log", "link-closure.txt", "package-sha256.txt")


def digest(data):
    return hashlib.sha256(data).hexdigest()


def bind(args):
    repo = Path(args.repo).resolve(strict=True)
    runner = Path(args.runner).resolve(strict=True)
    def git(*argv):
        return subprocess.check_output(["git", "-C", str(repo), *argv], text=True).strip()
    if git("status", "--porcelain", "--untracked-files=all"):
        raise SystemExit("REFUSE: freeze a clean committed source before deriving launch identities")
    tip = git("rev-parse", "HEAD")
    tree = git("rev-parse", "HEAD^{tree}")
    key = "hl332-" + tip[:12]
    inputs = {}
    for name in SCRIPT_INPUTS:
        path = repo / name
        actual = path.read_bytes()
        frozen = subprocess.check_output(["git", "-C", str(repo), "show", f"{tip}:{name}"])
        if actual != frozen:
            raise SystemExit("REFUSE: source input differs from frozen object: " + name)
        inputs[name] = digest(actual)
        if path.suffix == ".sh":
            subprocess.run(["bash", "-n", str(path)], check=True)
        else:
            ast.parse(actual, filename=str(path))
    native = runner.read_text()
    subprocess.run(["bash", "-n", str(runner)], check=True)
    # Exercise the installed consumer's actual options parser, not a guessed
    # replacement grammar. Only this parser slice runs; no placement/probe code.
    start = native.index("declare -A GATE_OPTIONS=(")
    stop = native.index("\nCMD=(", start)
    parser = native[start:stop]
    argv = ["fab", "build", "--repo", str(repo), "--tree", tree, "--key", key,
            "--cores-max", str(args.cores), "--compiler-use", "none"]
    for name in ARTIFACTS:
        argv += ["--artifact", "hl332-result/" + name]
    command = ["bash", "scripts/hl332-build.sh", tree, tip, key, args.image,
               str(args.cores), args.memory, str(args.wall_seconds)]
    argv += ["--", *command]
    harness = ('die(){ printf "%s\\n" "$*" >&2; exit 2; }; EXPECTED_TREE=""; SPREAD=0; VERB_ROLE="";\n'
               + parser + '\nprintf "%s\\0" "$REPO" "$EXPECTED_TREE" "$KEYOVERRIDE" "$CORES_MAX" "$COMPILER_USE_DECLARED" "${ARTIFACTS[@]}" "$@"\n')
    subprocess.run(["bash", "-n"], input=harness.encode(), check=True)
    parsed = subprocess.check_output(["bash", "-c", harness, "hl332-grammar", *argv[2:]]).split(b"\0")[:-1]
    expected = [str(repo), tree, key, str(args.cores), "none", *["hl332-result/" + n for n in ARTIFACTS], *command]
    if parsed != [v.encode() for v in expected]:
        raise SystemExit("REFUSE: native consumer parsed different argv/path/source joins")
    rendered = shlex.join(command)
    if shlex.split(rendered) != command:
        raise SystemExit("REFUSE: shell argv roundtrip differs")
    subprocess.run(["bash", "-n"], input=(rendered + "\n").encode(), check=True)
    if "--tree $EXPECTED_TREE does not match the exact shipped tree $TREE" not in native:
        raise SystemExit("REFUSE: installed consumer lacks the reviewed exact-snapshot tree gate")
    return {"status": "DRY_LAUNCH_PASS", "repo": str(repo), "commit": tip, "tree": tree,
            "runner": str(runner), "runner_sha256": digest(runner.read_bytes()), "inputs": inputs,
            "argv": argv, "rendered_command": rendered,
            "environment": {"FAB_BUILD_EST_GB": DEFAULT_ESTIMATE_GIB},
            "scope": "native option parser, clean snapshot identities, literal argv/path/source joins, shell/Python syntax; no remote effects"}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=("dry-run", "launch"))
    p.add_argument("--repo", required=True)
    p.add_argument("--runner", required=True)
    p.add_argument("--receipt", required=True)
    p.add_argument("--image", default=DEFAULT_IMAGE)
    p.add_argument("--cores", type=int, default=DEFAULT_CORES)
    p.add_argument("--memory", default=DEFAULT_MEMORY)
    p.add_argument("--wall-seconds", type=int, default=DEFAULT_WALL_SECONDS)
    args = p.parse_args()
    if args.cores < 1 or args.wall_seconds < 1:
        p.error("resource bounds must be positive")
    proof = bind(args)
    receipt = Path(args.receipt).resolve()
    if receipt.is_relative_to(Path(args.repo).resolve()):
        p.error("receipt must be outside the shipped source snapshot")
    if args.mode == "dry-run":
        with receipt.open("x") as f:
            json.dump(proof, f, indent=2)
            f.write("\n")
        print(json.dumps(proof, indent=2))
        return
    if json.loads(receipt.read_text()) != proof:
        raise SystemExit("REFUSE: dry-launch receipt does not bind this exact launch")
    if os.environ.get("FAB_EXCLUDE_HOSTS"):
        raise SystemExit("REFUSE: let Fab place this bounded build; no excluded hosts")
    env = dict(os.environ, **proof["environment"])
    os.execvpe(proof["argv"][0], proof["argv"], env)


if __name__ == "__main__":
    main()
