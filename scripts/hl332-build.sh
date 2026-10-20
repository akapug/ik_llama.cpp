#!/usr/bin/env bash
# Invoked only by Fab from the exact frozen source snapshot.
set -euo pipefail
expected_tree=$1
expected_commit=$2
key=$3
image=$4
cores_max=$5
memory=$6
wall_seconds=$7
[ "$(git rev-parse 'HEAD^{tree}')" = "$expected_tree" ]
[ "$(git rev-parse HEAD)" = "$expected_commit" ]
[ -z "$(git status --porcelain --untracked-files=all)" ]
[[ $key =~ ^hl332-[0-9a-f]{12}$ ]]
[[ $cores_max =~ ^[1-9][0-9]*$ && $wall_seconds =~ ^[1-9][0-9]*$ ]]
cores=${FAB_JOBS:?Fab must declare the admitted CPU allocation}
[[ $cores =~ ^[1-9][0-9]*$ ]]
(( cores <= cores_max ))
out="$PWD/hl332-result"
# Create-only: a failed or earlier run is evidence, never overwritten or deleted.
mkdir "$out"
printf '%s\n' "$expected_tree" > "$out/source-tree.txt"
printf '%s\n' "$expected_commit" > "$out/source-commit.txt"
printf '%s\n' "$image" > "$out/image-request.txt"
# Container images/dependencies are the only network input of this build.
docker pull "$image" 2>&1 | tee "$out/image-pull.log"
image_id=$(docker image inspect --format '{{.Id}}' "$image")
docker image inspect --format '{{json .RepoDigests}}' "$image_id" > "$out/image-digests.json"
printf '%s\n' "$image_id" > "$out/image-id.txt"
# No GPU/device mounts, no host packages or config writes. The selected immutable
# image ID is used rather than re-resolving a tag between inspection and execution.
docker run --rm --runtime=runc --name "$key" \
  --cpus "$cores" --memory "$memory" --memory-swap "$memory" --pids-limit 1024 \
  -e NVIDIA_VISIBLE_DEVICES=void -e OMP_NUM_THREADS="$cores" \
  -v "$PWD:/source:ro" -v "$out:/result:rw" \
  "$image_id" timeout --signal=TERM --kill-after=30s "$wall_seconds" \
  bash /source/scripts/hl332-container.sh /result "$cores" "$(id -u)" "$(id -g)" \
  2>&1 | tee "$out/container.log"
