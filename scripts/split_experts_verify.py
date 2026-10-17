#!/usr/bin/env python3
"""Round-trip check of a split-experts GGUF (scripts/split_experts.py) against its source GGUF.

For every MoE layer of SPLIT:
  - blk.N.ffn_exps_part.weight is I32 [n_expert, 2]; row 0 = partition tag index, row 1 = index in the partition;
  - the map is a bijection: every expert lands in exactly one partition at a distinct index, and every slot of every
    partition tensor is used by exactly one expert;
  - each partition holds the same roles (up/gate/down or gate_up) with ne0/ne1 of the source tensor;
  - with --bytes: where a partition tensor kept the source type, every expert slice is byte-identical to the source
    slice of the same global expert (a requantized partition is reported with its type and skipped).
Every non-expert tensor must exist in both files with the same shape (and, with --bytes, the same type and bytes).
Exit 0 only if everything holds. Multi-shard inputs: pass the first shard of each.
usage: split_experts_verify.py ORIG.gguf SPLIT.gguf [--bytes] [--expect-types hot=MXFP4,cold=IQ2_KS]
"""
import argparse, re, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'gguf-py'))
import numpy as np  # noqa: E402
import gguf  # noqa: E402

TAGS = ['hot', 'warm', 'cold', 'p3', 'p4', 'p5', 'p6', 'p7']
FULL_RE = re.compile(r'^blk\.(\d+)\.ffn_(up|gate|down|gate_up)_exps\.weight$')
PART_RE = re.compile(r'^blk\.(\d+)\.ffn_(up|gate|down|gate_up)_exps_(' + '|'.join(TAGS) + r')\.weight$')
MAP_RE = re.compile(r'^blk\.(\d+)\.ffn_exps_part\.weight$')
SHARD_RE = re.compile(r'^(.*)-(\d{5})-of-(\d{5})\.gguf$')


def tensors(first):
    m = SHARD_RE.match(str(first))
    paths = [Path(first)] if not m else \
        [Path(f'{m.group(1)}-{i:05d}-of-{m.group(3)}.gguf') for i in range(1, int(m.group(3)) + 1)]
    out = {}
    for p in paths:
        for t in gguf.GGUFReader(p).tensors:
            out[t.name] = t
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('orig')
    ap.add_argument('split')
    ap.add_argument('--bytes', action='store_true')
    ap.add_argument('--expect-types', default='')
    a = ap.parse_args()
    expect = {}
    for kv in filter(None, a.expect_types.split(',')):
        tag, _, typ = kv.partition('=')
        expect[tag] = typ.upper()

    src, dst = tensors(a.orig), tensors(a.split)
    errs = []
    full = {}
    for name, t in src.items():
        m = FULL_RE.match(name)
        if m:
            full.setdefault(int(m.group(1)), {})[m.group(2)] = t
    parts, maps = {}, {}
    for name, t in dst.items():
        m = PART_RE.match(name)
        if m:
            parts.setdefault(int(m.group(1)), {}).setdefault(m.group(3), {})[m.group(2)] = t
        elif MAP_RE.match(name):
            maps[int(MAP_RE.match(name).group(1))] = t

    stats = {}
    for il in sorted(set(full) | set(parts) | set(maps)):
        where = f'blk.{il}'
        if il not in full:
            errs.append(f'{where}: split file has partitions but the source has no expert tensors')
            continue
        if il not in maps or il not in parts:
            errs.append(f'{where}: no ffn_exps_part map or no partition tensors (layer not split)')
            continue
        roles = full[il]
        n_exp = int(next(iter(roles.values())).shape[2])
        mt = maps[il]
        if mt.tensor_type != gguf.GGMLQuantizationType.I32 or [int(x) for x in mt.shape] != [n_exp, 2]:
            errs.append(f'{where}: map is {mt.tensor_type.name} {list(mt.shape)}, expected I32 [{n_exp}, 2]')
            continue
        mp = np.asarray(mt.data, dtype=np.int32).reshape(2, n_exp)
        tag_of, idx_of = mp[0], mp[1]
        size = {}
        for tag, rs in parts[il].items():
            if set(rs) != set(roles):
                errs.append(f'{where}.{tag}: roles {sorted(rs)} != source roles {sorted(roles)}')
            ks = {int(t.shape[2]) for t in rs.values()}
            if len(ks) != 1:
                errs.append(f'{where}.{tag}: roles disagree on the expert count {sorted(ks)}')
            size[TAGS.index(tag)] = min(ks)
            for role, t in rs.items():
                s = roles.get(role)
                if s is not None and [int(x) for x in t.shape[:2]] != [int(x) for x in s.shape[:2]]:
                    errs.append(f'{where}.{tag}.{role}: shape {list(t.shape)} vs source {list(s.shape)}')
                if tag in expect and t.tensor_type.name != expect[tag]:
                    errs.append(f'{where}.{tag}.{role}: type {t.tensor_type.name}, expected {expect[tag]}')
        seen = {p: np.zeros(k, dtype=np.int64) for p, k in size.items()}
        for e in range(n_exp):
            p, k = int(tag_of[e]), int(idx_of[e])
            if p not in size or not 0 <= k < size[p]:
                errs.append(f'{where}: expert {e} -> ({p}, {k}) points outside the partitions {size}')
                continue
            seen[p][k] += 1
        for p, s in seen.items():
            bad = np.nonzero(s != 1)[0]
            if len(bad):
                errs.append(f'{where}.{TAGS[p]}: slots {bad[:8].tolist()} used {s[bad[:8]].tolist()} times (want 1)')
        for p, k in size.items():
            stats[TAGS[p]] = stats.get(TAGS[p], 0) + k
        if not a.bytes:
            continue
        for tag, rs in parts[il].items():
            p = TAGS.index(tag)
            ids = np.nonzero(tag_of == p)[0]
            for role, t in rs.items():
                s = roles[role]
                if t.tensor_type != s.tensor_type:
                    stats[f'requant {tag}.{role} {s.tensor_type.name}->{t.tensor_type.name}'] = 1
                    continue
                sb = np.asarray(s.data).reshape(n_exp, -1)
                tb = np.asarray(t.data).reshape(int(t.shape[2]), -1)
                for e in ids:
                    if not np.array_equal(sb[e], tb[idx_of[e]]):
                        errs.append(f'{where}.{tag}.{role}: expert {e} (slot {idx_of[e]}) differs from the source slice')
                        break

    for name, t in src.items():
        if FULL_RE.match(name):
            continue
        u = dst.get(name)
        if u is None:
            errs.append(f'{name}: missing from the split file')
        elif list(u.shape) != list(t.shape):
            errs.append(f'{name}: shape {list(u.shape)} vs {list(t.shape)}')
        elif a.bytes and (u.tensor_type != t.tensor_type or not np.array_equal(np.asarray(u.data), np.asarray(t.data))):
            errs.append(f'{name}: type/bytes differ ({t.tensor_type.name} -> {u.tensor_type.name})')
    extra = [n for n in dst if n not in src and not PART_RE.match(n) and not MAP_RE.match(n)]
    if extra:
        errs.append(f'split file has unexpected tensors: {extra[:8]}')

    print(f'layers split: {len(maps)}; experts per partition: ' +
          ', '.join(f'{k}={v}' for k, v in stats.items() if not k.startswith('requant')))
    for k in sorted(k for k in stats if k.startswith('requant')):
        print(' ', k)
    for e in errs[:40]:
        print('FAIL', e)
    print('RESULT', 'PASS' if not errs else f'FAIL ({len(errs)} problems)')
    sys.exit(0 if not errs else 1)


if __name__ == '__main__':
    main()
