#!/usr/bin/env python3
"""Split the routed-expert tensors of a GGUF into per-partition tensors (e.g. hot / warm / cold), so each partition
can get its own quant type (llama-quantize --custom-q "ffn_.*_exps_cold=iq2_ks,...") and its own memory placement.

For every MoE layer, blk.N.ffn_{up,gate,down,gate_up}_exps.weight [ne0, ne1, n_expert] becomes
blk.N.ffn_<role>_exps_<tag>.weight [ne0, ne1, K_tag] (verbatim byte copies of the expert slices, sorted by expert id)
plus blk.N.ffn_exps_part.weight, I32 [n_expert, 2]: row 0 = partition tag index, row 1 = index inside the partition.
Tags (index order, as the loader knows them): hot warm cold p3 p4 p5 p6 p7. Everything else is copied unchanged.
The unrequantized split computes exactly what the original computes (ik_llama.cpp branch split-exps).

Partition lists (--part TAG=FILE, repeatable, first listed wins on overlap; experts in no list go to --rest):
  TSV  : 'blk.N<TAB>e1,e2,a-b' per line (the LLAMA_HOT_EXPERTS format), '#' comments
  JSON : {"layers": {"model.layers.N.ffn.experts": [ids...]}} (vLLM skew alloc files) or {"blk.N": [ids...]}

Multi-shard input: pass the first shard (*-00001-of-0000K.gguf); every shard is written with the same split layout.
Shards without expert tensors are hard-linked (--link-unchanged, default; their split.tensors.count is stale, which
the loader ignores: it reads the count from the first shard only).

--imatrix IN.dat --imatrix-out OUT.dat slices an imatrix (.dat, the format llama-quantize reads) the same way: entries
for the partition tensors with the matching expert rows (the gate entry falls back to up, as quantize does).

Per-partition types (--type, repeatable): the split itself only copies bytes; the types are applied by llama-quantize
with --custom-q rules that this script writes (--quant-rules FILE, also with --dry-run), and runs with --quantize-bin.
  --type TAG[.ROLE][@LAYERS]=TYPE   ROLE in up/gate/down/gate_up, LAYERS like 0-9,20; TYPE a ggml type name
                                    (case-insensitive: q2_K, iq2_ks, q3_K ...) or 'keep' (= the source type, bytes copied)
  First matching --type wins; a partition tensor no rule matches keeps its source type. The rules also pin every
  non-expert tensor to its source type, so quantize changes nothing but the listed partitions. Example (hot keeps the
  native MXFP4, cold q2_K with q3_K down on layers 0-9):
    --type hot=keep --type cold.down@0-9=q3_K --type cold=q2_K

usage: split_experts.py IN.gguf OUT.gguf --part hot=hot.tsv [--part warm=warm.json] [--rest cold] [--imatrix ...]
                        [--type SPEC ... --quant-rules RULES.txt [--quantize-bin DIR --quant-out OUT-mixed.gguf]]
"""
import argparse, json, os, re, struct, subprocess, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'gguf-py'))
import numpy as np  # noqa: E402
import gguf  # noqa: E402

TAGS = ['hot', 'warm', 'cold', 'p3', 'p4', 'p5', 'p6', 'p7']
EXP_RE = re.compile(r'^blk\.(\d+)\.ffn_(up|gate|down|gate_up)_exps\.weight$')
SHARD_RE = re.compile(r'^(.*)-(\d{5})-of-(\d{5})\.gguf$')


def parse_ids(spec):
    out = []
    for part in spec.split(','):
        part = part.strip()
        if not part:
            continue
        a, _, b = part.partition('-')
        out.extend(range(int(a), int(b or a) + 1))
    return out


def load_list(path):
    """-> {layer: [expert ids]}"""
    text = Path(path).read_text()
    res = {}
    if text.lstrip().startswith('{'):
        d = json.loads(text)
        layers = d.get('layers', d)
        for key, ids in layers.items():
            m = re.search(r'(?:model\.layers|blk)\.(\d+)', key)
            if not m:
                raise SystemExit(f'{path}: cannot read a layer number from key {key!r}')
            res.setdefault(int(m.group(1)), []).extend(int(i) for i in ids)
        return res
    for ln, line in enumerate(text.splitlines(), 1):
        line = line.split('#', 1)[0].strip()
        if not line:
            continue
        f = line.split()
        m = re.fullmatch(r'(?:blk\.)?(\d+)', f[0])
        if not m or len(f) != 2:
            raise SystemExit(f"{path}:{ln}: expected 'blk.N<TAB>e1,e2,a-b'")
        res.setdefault(int(m.group(1)), []).extend(parse_ids(f[1]))
    return res


def shard_paths(first, search=()):
    """all shards of a split GGUF; a shard missing next to the first one is looked up in the --shard-search dirs"""
    m = SHARD_RE.match(str(first))
    if not m:
        return [Path(first)], None
    n = int(m.group(3))
    out = []
    for i in range(1, n + 1):
        p = Path(f'{m.group(1)}-{i:05d}-of-{n:05d}.gguf')
        if not p.exists():
            p = next((Path(d) / p.name for d in search if (Path(d) / p.name).exists()), p)
        if not p.exists():
            raise SystemExit(f'shard {p.name} not found next to {first} nor in --shard-search {list(search)}')
        out.append(p)
    return out, n


def out_paths(out, n):
    if n is None:
        return [Path(out)]
    base = re.sub(r'(-\d{5}-of-\d{5})?\.gguf$', '', str(out))
    return [Path(f'{base}-{i:05d}-of-{n:05d}.gguf') for i in range(1, n + 1)]


def str_bytes(s):
    b = s.encode()
    return struct.pack('<Q', len(b)) + b


def kv_bytes(key, vtype, payload):
    return str_bytes(key) + struct.pack('<I', vtype) + payload


def build_plan(readers, lists, rest_tag):
    """-> {layer: (n_expert, part_of[e] tag index, idx_of[e], {tag: sorted ids})}"""
    layers = {}
    for r in readers:
        for t in r.tensors:
            m = EXP_RE.match(t.name)
            if m:
                layers.setdefault(int(m.group(1)), int(t.shape[2]))
    plan = {}
    rest = TAGS.index(rest_tag)
    for il, n_exp in sorted(layers.items()):
        part_of = [-1] * n_exp
        for tag, lst in lists:
            for e in lst.get(il, []):
                if not 0 <= e < n_exp:
                    raise SystemExit(f'layer {il}: expert {e} out of range (n_expert {n_exp})')
                if part_of[e] < 0:
                    part_of[e] = TAGS.index(tag)
        part_of = [p if p >= 0 else rest for p in part_of]
        members = {}
        for e, p in enumerate(part_of):
            members.setdefault(p, []).append(e)
        idx_of = [0] * n_exp
        for p, ids in members.items():
            for k, e in enumerate(ids):
                idx_of[e] = k
        plan[il] = (n_exp, part_of, idx_of, members)
    for tag, lst in lists:
        extra = sorted(set(lst) - set(plan))
        if extra:
            print(f'warning: list {tag} names layers without expert tensors: {extra[:8]}', file=sys.stderr)
    return plan


def write_shard(r, out, plan, n_total, info, map_here):
    align = 32
    if 'general.alignment' in r.fields:
        align = int(r.fields['general.alignment'].contents())
    kvs = []
    for name, f in r.fields.items():
        if name.startswith('GGUF.'):
            continue
        if name == 'split.tensors.count':
            kvs.append(kv_bytes(name, f.types[0], np.array([n_total], dtype=f.parts[-1].dtype).tobytes()))
            continue
        kvs.append(b''.join(p.tobytes() for p in f.parts))
    kvs.append(kv_bytes('split_experts.plan', gguf.GGUFValueType.STRING, str_bytes(info)))

    # tensor list: (name, shape ne0.., ggml type, nbytes, [(src offset, length)...])
    items = []
    for t in r.tensors:
        m = EXP_RE.match(t.name)
        if not m:
            items.append((t.name, [int(x) for x in t.shape], int(t.tensor_type), int(t.n_bytes), [(int(t.data_offset), int(t.n_bytes))]))
            continue
        il, role = int(m.group(1)), m.group(2)
        n_exp, part_of, idx_of, members = plan[il]
        nb2 = int(t.n_bytes) // n_exp
        assert nb2 * n_exp == int(t.n_bytes)
        for p in sorted(members):
            ids = members[p]
            ranges = []
            for e in ids:
                off = int(t.data_offset) + e * nb2
                if ranges and ranges[-1][0] + ranges[-1][1] == off:
                    ranges[-1] = (ranges[-1][0], ranges[-1][1] + nb2)
                else:
                    ranges.append((off, nb2))
            items.append((f'blk.{il}.ffn_{role}_exps_{TAGS[p]}.weight', [int(t.shape[0]), int(t.shape[1]), len(ids)],
                          int(t.tensor_type), nb2 * len(ids), ranges))
        if il in map_here:
            map_here.discard(il)
            arr = np.array(part_of + idx_of, dtype=np.int32).tobytes()
            items.append((f'blk.{il}.ffn_exps_part.weight', [n_exp, 2], int(gguf.GGMLQuantizationType.I32), len(arr), arr))

    tinfo = []
    off = 0
    for name, shape, typ, nbytes, _ in items:
        tinfo.append(str_bytes(name) + struct.pack('<I', len(shape)) + b''.join(struct.pack('<Q', d) for d in shape)
                     + struct.pack('<I', typ) + struct.pack('<Q', off))
        off += (nbytes + align - 1) // align * align
    head = b'GGUF' + struct.pack('<IQQ', 3, len(items), len(kvs)) + b''.join(kvs) + b''.join(tinfo)
    head += b'\0' * ((-len(head)) % align)

    tmp = Path(str(out) + '.tmp')
    src = r.data
    chunk = 64 << 20
    with open(tmp, 'wb') as fo:
        fo.write(head)
        for name, shape, typ, nbytes, ranges in items:
            if isinstance(ranges, bytes):
                fo.write(ranges)
            else:
                for so, ln in ranges:
                    for c in range(so, so + ln, chunk):
                        fo.write(memoryview(src[c:min(c + chunk, so + ln)]))
            fo.write(b'\0' * ((-nbytes) % align))
    os.replace(tmp, out)
    return len(items)


def count_items(r, plan, maps):
    n = len(maps)
    for t in r.tensors:
        m = EXP_RE.match(t.name)
        n += len(plan[int(m.group(1))][3]) if m else 1
    return n


def fix_zero_rows(name, rows):
    """experts never routed while the imatrix was collected have all-zero (or non-finite) importance, and q2_K/q3_K
    quantization with such weights fails validation: give them the mean of the tensor's healthy expert rows"""
    bad = ~np.isfinite(rows).all(axis=1) | (rows == 0).all(axis=1)
    if bad.any():
        if bad.all():
            print(f'warning: imatrix {name}: no healthy expert row, left as is', file=sys.stderr)
            return rows, 0
        rows = rows.copy()
        rows[bad] = rows[~bad].mean(axis=0)
    return rows, int(bad.sum())


def slice_imatrix(src, dst, plan, fix_zeros=True):
    data = Path(src).read_bytes()
    pos = 0

    def rd(fmt):
        nonlocal pos
        v = struct.unpack_from(fmt, data, pos)
        pos += struct.calcsize(fmt)
        return v

    (n,) = rd('<i')
    entries = {}
    order = []
    for _ in range(n):
        (ln,) = rd('<i')
        name = data[pos:pos + ln].decode()
        pos += ln
        ncall, nval = rd('<ii')
        vals = np.frombuffer(data, dtype=np.float32, count=nval, offset=pos)
        pos += 4 * nval
        entries[name] = (ncall, vals)
        order.append(name)
    trailer = data[pos:]
    out = dict(entries)
    made = n_fixed = 0
    for il, (n_exp, part_of, idx_of, members) in plan.items():
        for role in ('up', 'gate', 'down', 'gate_up'):
            name = f'blk.{il}.ffn_{role}_exps.weight'
            src_name = name if name in entries else (f'blk.{il}.ffn_up_exps.weight' if role == 'gate' else None)
            if src_name not in entries:
                continue
            ncall, vals = entries[src_name]
            if len(vals) % n_exp:
                raise SystemExit(f'imatrix {src_name}: {len(vals)} values are not a multiple of {n_exp} experts')
            rows = vals.reshape(n_exp, -1)
            if fix_zeros:
                rows, nb = fix_zero_rows(src_name, rows)
                n_fixed += nb if src_name == name else 0
            for p, ids in members.items():
                out[f'blk.{il}.ffn_{role}_exps_{TAGS[p]}.weight'] = (ncall, np.ascontiguousarray(rows[ids]).reshape(-1))
                made += 1
    with open(dst, 'wb') as fo:
        fo.write(struct.pack('<i', len(out)))
        for name, (ncall, vals) in out.items():
            b = name.encode()
            fo.write(struct.pack('<i', len(b)) + b + struct.pack('<ii', ncall, len(vals)))
            fo.write(np.asarray(vals, dtype=np.float32).tobytes())
        fo.write(trailer)
    print(f'imatrix: {len(entries)} entries in, {made} partition entries added -> {dst}' +
          (f'; {n_fixed} all-zero/non-finite expert rows set to the tensor mean' if fix_zeros else ''))


ROLES = ('up', 'gate', 'down', 'gate_up')
# tensors llama-quantize never converts (int types); everything else that is >= 2D and ends in 'weight' it may touch
NO_QUANT = {'I8', 'I16', 'I32', 'I64', 'BF16_R16'}


def ggml_type_names():
    """lower-case name -> ggml's own spelling (llama-quantize matches type names exactly)."""
    src = Path(__file__).resolve().parent.parent / 'ggml' / 'src' / 'ggml.c'
    names = re.findall(r'\.type_name\s*=\s*"([^"]+)"', src.read_text(errors='replace')) if src.exists() else []
    out = {n.lower(): n for n in names if n != 'DEPRECATED'}
    if not out:  # no ggml source next to the script: the common spelling (k-quants keep a capital K)
        for t in gguf.GGMLQuantizationType:
            n = t.name.lower()
            out[n] = re.sub(r'^(q\d)_k$', r'\1_K', n)
    return out


def parse_type_spec(spec, names):
    """'TAG[.ROLE][@LAYERS]=TYPE' -> (tag, role or None, layer set or None, type name or 'keep')"""
    m = re.fullmatch(r'([a-z0-9]+)(?:\.([a-z_]+))?(?:@([0-9,\-]+))?=([A-Za-z0-9_]+)', spec)
    if not m or m.group(1) not in TAGS or (m.group(2) and m.group(2) not in ROLES):
        raise SystemExit(f'--type {spec}: expected TAG[.ROLE][@LAYERS]=TYPE, TAG in {TAGS}, ROLE in {ROLES}')
    typ = m.group(4)
    if typ.lower() != 'keep':
        if typ.lower() not in names:
            raise SystemExit(f'--type {spec}: unknown ggml type {typ!r}')
        typ = names[typ.lower()]
    else:
        typ = 'keep'
    return m.group(1), m.group(2), set(parse_ids(m.group(3))) if m.group(3) else None, typ


def alt(ids):
    return '(' + '|'.join(str(i) for i in sorted(ids)) + ')'


def quant_rules(readers, plan, specs, names):
    """-> (list of 'regex=type' rules for llama-quantize --custom-q, {(tag, type): n_tensors})"""
    src_name = lambda t: names.get(t.tensor_type.name.lower(), t.tensor_type.name.lower())
    exp, pin, summary = {}, {}, {}
    for r in readers:
        for t in r.tensors:
            m = EXP_RE.match(t.name)
            if m:
                il, role = int(m.group(1)), m.group(2)
                for p in plan[il][3]:
                    tag = TAGS[p]
                    typ = next((ty for tg, ro, ls, ty in specs if tg == tag and ro in (None, role) and (ls is None or il in ls)), 'keep')
                    typ = src_name(t) if typ == 'keep' else typ
                    exp.setdefault((role, tag, typ), []).append(il)
                    summary[(tag, typ)] = summary.get((tag, typ), 0) + 1
                continue
            if t.tensor_type.name in NO_QUANT or len(t.shape) < 2 or not t.name.endswith('weight'):
                continue
            mb = re.match(r'^blk\.(\d+)\.(.*)$', t.name)
            key = (mb.group(2), src_name(t)) if mb else (t.name, src_name(t))
            pin.setdefault(key, []).append(int(mb.group(1)) if mb else None)
    rules = [rf'^blk\.{alt(ls)}\.ffn_{role}_exps_{tag}\.weight$={typ}' for (role, tag, typ), ls in sorted(exp.items())]
    for (stem, typ), ls in sorted(pin.items()):
        rules.append(rf'^{re.escape(stem)}$={typ}' if ls == [None] else rf'^blk\.{alt(ls)}\.{re.escape(stem)}$={typ}')
    bad = [x for x in rules if ',' in x or x.count('=') != 1]
    if bad:
        raise SystemExit(f'cannot express these rules for --custom-q (comma or extra "="): {bad[:3]}')
    return rules, summary


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('input')
    ap.add_argument('output', nargs='?')
    ap.add_argument('--part', action='append', default=[], metavar='TAG=FILE')
    ap.add_argument('--rest', default='cold', choices=TAGS)
    ap.add_argument('--imatrix')
    ap.add_argument('--imatrix-out')
    ap.add_argument('--no-imatrix-fix', action='store_true', help='keep all-zero expert rows (quantize then fails on them)')
    ap.add_argument('--shard-search', action='append', default=[], metavar='DIR', help='other dirs holding shards of the input')
    ap.add_argument('--no-link-unchanged', action='store_true')
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--skip-split', action='store_true', help='OUTPUT shards already written by an earlier run: only quantize')
    ap.add_argument('--type', action='append', default=[], metavar='TAG[.ROLE][@LAYERS]=TYPE')
    ap.add_argument('--quant-rules', metavar='FILE', help='write the llama-quantize --custom-q rule string here')
    ap.add_argument('--quantize-bin', metavar='DIR', help='run DIR/llama-quantize on the split output with the rules')
    ap.add_argument('--quant-out', metavar='OUT.gguf')
    ap.add_argument('--threads', type=int, default=os.cpu_count())
    a = ap.parse_args()

    lists = []
    for spec in a.part:
        tag, _, path = spec.partition('=')
        if tag not in TAGS or not path:
            raise SystemExit(f'--part {spec}: expected TAG=FILE with TAG in {TAGS}')
        if tag == a.rest:
            raise SystemExit(f'--part {tag} is also the --rest partition')
        lists.append((tag, load_list(path)))

    ins, n_split = shard_paths(a.input, a.shard_search)
    readers = [gguf.GGUFReader(p) for p in ins]
    plan = build_plan(readers, lists, a.rest)
    if not plan:
        raise SystemExit('no ffn_*_exps tensors found')

    sizes = {}
    for r in readers:
        for t in r.tensors:
            m = EXP_RE.match(t.name)
            if m:
                n_exp, part_of, _, members = plan[int(m.group(1))]
                for p, ids in members.items():
                    sizes[TAGS[p]] = sizes.get(TAGS[p], 0) + int(t.n_bytes) // n_exp * len(ids)
    counts = {}
    for n_exp, part_of, _, members in plan.values():
        for p, ids in members.items():
            counts[TAGS[p]] = counts.get(TAGS[p], 0) + len(ids)
    info = ' '.join(f'{tag}={counts[tag]}' for tag in TAGS if tag in counts) + f' experts over {len(plan)} layers; ' + \
        ' '.join(f'--part {s}' for s in a.part) + f' --rest {a.rest}'
    print('plan:', info)
    for tag in TAGS:
        if tag in sizes:
            print(f'  {tag:5s} {counts[tag]:6d} experts {sizes[tag] / 2**30:9.2f} GiB (source types)')

    rules = None
    if a.type or a.quant_rules or a.quantize_bin:
        names = ggml_type_names()
        specs = [parse_type_spec(x, names) for x in a.type]
        rules, summary = quant_rules(readers, plan, specs, names)
        for (tag, typ), n in sorted(summary.items()):
            print(f'  type  {tag:5s} {typ:8s} {n:5d} tensors')
        if a.quant_rules:
            Path(a.quant_rules).write_text(','.join(rules) + '\n')
            print(f'{len(rules)} quantize rules -> {a.quant_rules}')
    if a.quantize_bin and not (a.output and a.quant_out):
        raise SystemExit('--quantize-bin needs OUTPUT and --quant-out')

    if a.imatrix:
        if not a.imatrix_out:
            raise SystemExit('--imatrix needs --imatrix-out')
        slice_imatrix(a.imatrix, a.imatrix_out, plan, not a.no_imatrix_fix)
    if a.dry_run or not a.output:
        return

    outs = out_paths(a.output, n_split)
    outs[0].parent.mkdir(parents=True, exist_ok=True)
    # each layer's map tensor goes to the shard that holds the layer's first expert tensor
    maps = [set() for _ in readers]
    for il in plan:
        first = next(i for i, r in enumerate(readers) if any(EXP_RE.match(t.name) and int(EXP_RE.match(t.name).group(1)) == il for t in r.tensors))
        maps[first].add(il)
    n_total = sum(count_items(r, plan, m) for r, m in zip(readers, maps))
    for r, src, dst, mp in zip(readers, ins, outs, maps):
        if a.skip_split:
            if not dst.exists():
                raise SystemExit(f'--skip-split: {dst} is missing')
            continue
        has_exps = any(EXP_RE.match(t.name) for t in r.tensors)
        if n_split and not has_exps and not a.no_link_unchanged and src != ins[0]:
            if dst.exists() or dst.is_symlink():
                dst.unlink()
            try:
                os.link(src, dst)
                how = 'hard link'
            except OSError:
                os.symlink(os.path.abspath(src), dst)
                how = 'symlink'
            print(f'{dst.name}: no expert tensors, {how} to {src}')
            continue
        n = write_shard(r, dst, plan, n_total, info, set(mp))
        print(f'{dst.name}: {n} tensors written')
    if a.quantize_bin:
        # Q8_0 is only the fallback type: every tensor quantize may touch is pinned by a rule
        cmd = [str(Path(a.quantize_bin) / 'llama-quantize'), '--allow-requantize']
        if a.imatrix_out:
            cmd += ['--imatrix', a.imatrix_out]
        if n_split:
            # shards without expert tensors are not rewritten: link them at the output names; --partial-requant then
            # skips every existing output shard (their tensors keep the source types, which the rules pin anyway)
            cmd += ['--keep-split', '--partial-requant']
            base = re.sub(r'(-\d{5}-of-\d{5})?\.gguf$', '', a.quant_out)
            Path(base).parent.mkdir(parents=True, exist_ok=True)
            for i, src in enumerate(ins):
                q = Path(f'{base}-{i + 1:05d}-of-{n_split:05d}.gguf')
                if q.exists() or q.is_symlink():
                    q.unlink()  # a shard left by an earlier run would be skipped
                if any(EXP_RE.match(t.name) for t in readers[i].tensors):
                    continue
                try:
                    os.link(src, q)
                except OSError:
                    os.symlink(os.path.abspath(src), q)
                print(f'{q.name}: no expert tensors, linked to {src} (quantize skips it)')
            a.quant_out = base + '.gguf'  # quantize --keep-split appends -0000N-of-0000M.gguf to the name without .gguf
        cmd += ['--custom-q', ','.join(rules), str(outs[0]), a.quant_out, 'Q8_0', str(a.threads)]
        print('running', ' '.join(cmd[:3]), '...', ' '.join(cmd[-4:]), flush=True)
        rc = subprocess.run(cmd).returncode
        if rc:
            raise SystemExit(f'llama-quantize rc={rc}')


if __name__ == '__main__':
    main()
