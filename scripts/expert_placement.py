#!/usr/bin/env python3
"""Make per-expert placement files from routed-token counts.

Counts file: TSV, one row per expert tensor: name (blk.N.ffn_{up,down,gate}_exps.weight), then one
routed-token count per expert. Rows of one layer carry the same counts (every routed token reads
up, gate and down of its expert), so the layer row is the first row seen for that layer.

  hot   GPU hot set for LLAMA_HOT_EXPERTS: the globally most-routed experts that fit --vram-gib.
        All experts have the same size, so the global top-N by count maximizes routed reads
        served from the GPU. Output lines: blk.N<TAB>e1,e2,...
  numa  DRAM/PMem split for LLAMA_EXPERT_PLACEMENT: the coldest experts go to the PMem node until
        --pmem-gib is used. Experts in --gpu-hot (if given) are read by decode from the GPU bank, so
        their CPU copies go to PMem first. Output lines: blk.N<TAB>*<TAB>dram, blk.N<TAB>list<TAB>pmem.
  eval  Share of routed reads that a hot file (GPU) or placement file (per node) serves under
        another counts file, to test whether a selection holds on held-out text.
"""

import argparse
import re
import sys

# DeepSeek-V4-Flash: up, gate and down are MXFP4 [4096, 2048] per expert = 4456448 bytes each
DEFAULT_EXPERT_BYTES = 3 * 4456448
GIB = 1 << 30


def read_counts(path):
    layers = {}
    with open(path) as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2:
                continue
            m = re.match(r"blk\.(\d+)\.", parts[0])
            if not m:
                continue
            layer = int(m.group(1))
            if layer not in layers:
                layers[layer] = [int(x) for x in parts[1:]]
    if not layers:
        sys.exit(f"{path}: no blk.N rows")
    return layers


def ids_text(ids):
    """Compact comma list with a-b ranges."""
    ids = sorted(ids)
    out, i = [], 0
    while i < len(ids):
        j = i
        while j + 1 < len(ids) and ids[j + 1] == ids[j] + 1:
            j += 1
        out.append(str(ids[i]) if i == j else f"{ids[i]}-{ids[j]}")
        i = j + 1
    return ",".join(out)


def parse_ids(spec, n):
    if spec == "*":
        return list(range(n))
    out = []
    for part in spec.split(","):
        if not part:
            continue
        a, _, b = part.partition("-")
        out.extend(range(int(a), int(b or a) + 1))
    return out


def read_hot(path, n_expert):
    hot = {}
    with open(path) as f:
        for line in f:
            line = line.split("#", 1)[0].strip()
            if not line:
                continue
            key, ids = line.split()[:2]
            hot.setdefault(int(key.removeprefix("blk.")), set()).update(parse_ids(ids, n_expert))
    return hot


def read_placement(path, n_expert):
    place = {}
    with open(path) as f:
        for line in f:
            line = line.split("#", 1)[0].strip()
            if not line:
                continue
            key, ids, node = line.split()[:3]
            m = re.match(r"(?:blk\.)?(\d+)", key)
            layer = place.setdefault(int(m.group(1)), {})
            for e in parse_ids(ids, n_expert):
                layer[e] = node
    return place


def cmd_hot(a):
    counts = read_counts(a.counts)
    budget = int(a.vram_gib * GIB) // a.expert_bytes
    ranked = sorted(((c, layer, e) for layer, row in counts.items() for e, c in enumerate(row)), reverse=True)
    pick = {}
    for c, layer, e in ranked[:budget]:
        pick.setdefault(layer, []).append(e)
    total = sum(sum(r) for r in counts.values())
    served = sum(counts[layer][e] for layer, es in pick.items() for e in es)
    n = sum(len(v) for v in pick.values())
    print(f"# hot experts from {a.counts}: {n} experts, {n * a.expert_bytes / GIB:.2f} GiB "
          f"(expert {a.expert_bytes} B); in-sample share of routed reads on GPU {served / total:.3f}")
    for layer in sorted(pick):
        cov = sum(counts[layer][e] for e in pick[layer]) / max(1, sum(counts[layer]))
        print(f"blk.{layer}\t{ids_text(pick[layer])}\t# {len(pick[layer])} experts, layer share {cov:.3f}")


def cmd_numa(a):
    counts = read_counts(a.counts)
    n_expert = len(next(iter(counts.values())))
    gpu = read_hot(a.gpu_hot, n_expert) if a.gpu_hot else {}
    budget = int(a.pmem_gib * GIB) // a.expert_bytes
    # GPU-resident experts first (decode never reads their CPU copy), then the coldest by count
    ranked = sorted(((e not in gpu.get(layer, ()), c, layer, e)
                     for layer, row in counts.items() for e, c in enumerate(row)))
    pmem = {}
    for _, c, layer, e in ranked[:budget]:
        pmem.setdefault(layer, []).append(e)
    total = sum(sum(r) for r in counts.values())
    cpu_total = sum(c for layer, row in counts.items() for e, c in enumerate(row) if e not in gpu.get(layer, ()))
    pm_reads = sum(counts[layer][e] for layer, es in pmem.items() for e in es if e not in gpu.get(layer, ()))
    n = sum(len(v) for v in pmem.values())
    print(f"# placement from {a.counts}: {n} experts ({n * a.expert_bytes / GIB:.2f} GiB) on node {a.pmem_node}, "
          f"rest on node(s) {a.dram_nodes}")
    print(f"# in-sample decode reads served by PMem: {pm_reads / total:.3f} of all routed reads, "
          f"{pm_reads / max(1, cpu_total):.3f} of CPU-side reads" + (f" (GPU hot set {a.gpu_hot})" if gpu else ""))
    for layer in sorted(counts):
        print(f"blk.{layer}\t*\t{a.dram_nodes}")
        if layer in pmem:
            print(f"blk.{layer}\t{ids_text(pmem[layer])}\t{a.pmem_node}")


def cmd_eval(a):
    counts = read_counts(a.counts)
    n_expert = len(next(iter(counts.values())))
    total = sum(sum(r) for r in counts.values())
    if a.hot:
        hot = read_hot(a.hot, n_expert)
        served = sum(counts[l][e] for l, es in hot.items() if l in counts for e in es)
        print(f"{a.hot}: GPU serves {served / total:.3f} of routed reads in {a.counts}")
    if a.placement:
        place = read_placement(a.placement, n_expert)
        gpu = read_hot(a.hot, n_expert) if a.hot else {}
        per = {}
        for l, row in counts.items():
            for e, c in enumerate(row):
                node = "gpu" if e in gpu.get(l, ()) else place.get(l, {}).get(e, "unplaced")
                per[node] = per.get(node, 0) + c
        for node, c in sorted(per.items()):
            print(f"{a.placement}: {node}: {c / total:.3f} of routed reads")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    h = sub.add_parser("hot")
    h.add_argument("counts")
    h.add_argument("--vram-gib", type=float, required=True)
    h.add_argument("--expert-bytes", type=int, default=DEFAULT_EXPERT_BYTES)
    h.set_defaults(fn=cmd_hot)
    n = sub.add_parser("numa")
    n.add_argument("counts")
    n.add_argument("--pmem-gib", type=float, required=True)
    n.add_argument("--pmem-node", default="2")
    n.add_argument("--dram-nodes", default="0")
    n.add_argument("--gpu-hot", help="LLAMA_HOT_EXPERTS file: these experts go to PMem first")
    n.add_argument("--expert-bytes", type=int, default=DEFAULT_EXPERT_BYTES)
    n.set_defaults(fn=cmd_numa)
    e = sub.add_parser("eval")
    e.add_argument("counts")
    e.add_argument("--hot")
    e.add_argument("--placement")
    e.set_defaults(fn=cmd_eval)
    a = p.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
