#!/usr/bin/env python3
"""Self-contained test of split_experts.py + split_experts_verify.py on a tiny synthetic GGUF (no model needed):
partition lists in both formats with an overlap (first list wins), the map round trip (expert -> (tag, index) ->
the same bytes), a corrupted map that the verifier must reject, and imatrix slicing.
usage: split_experts_test.py   (exit 0 = pass)"""
import json, struct, subprocess, sys, tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'gguf-py'))
import numpy as np  # noqa: E402
import gguf  # noqa: E402

N_EXP, NE0, NE1 = 8, 32, 4
TAGS = ['hot', 'warm', 'cold']
fails = []


def check(cond, what):
    print(('ok   ' if cond else 'FAIL ') + what)
    if not cond:
        fails.append(what)


def run(*args):
    r = subprocess.run([sys.executable, '-I', *map(str, args)], capture_output=True, text=True)
    return r.returncode, r.stdout + r.stderr


def main():
    rng = np.random.default_rng(7)
    d = Path(tempfile.mkdtemp(prefix='split-exps-test-'))
    src = d / 'tiny.gguf'
    w = gguf.GGUFWriter(src, 'deepseek4')
    w.add_uint32('deepseek4.expert_count', N_EXP)
    full = {}
    w.add_tensor('token_embd.weight', rng.standard_normal((10, NE0), dtype=np.float32))
    for il in range(3):
        dt = np.float16 if il == 2 else np.float32
        for role, shape in (('up', (N_EXP, NE1, NE0)), ('gate', (N_EXP, NE1, NE0)), ('down', (N_EXP, NE0, NE1))):
            a = rng.standard_normal(shape).astype(dt)
            full[(il, role)] = a
            w.add_tensor(f'blk.{il}.ffn_{role}_exps.weight', a)
        w.add_tensor(f'blk.{il}.ffn_norm.weight', rng.standard_normal(NE0, dtype=np.float32))
    w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()

    (d / 'hot.tsv').write_text('# test\nblk.0\t1,3,5-6\nblk.2\t0-7\n')
    (d / 'warm.json').write_text(json.dumps({'layers': {'model.layers.0.ffn.experts': [3, 7], 'blk.1': [2, 4]}}))
    # imatrix .dat: up and down entries for layer 0 (gate falls back to up)
    ent = {'blk.0.ffn_up_exps.weight': rng.random(N_EXP * NE0, dtype=np.float32),
           'blk.0.ffn_down_exps.weight': rng.random(N_EXP * NE1, dtype=np.float32)}
    with open(d / 'in.imatrix', 'wb') as f:
        f.write(struct.pack('<i', len(ent)))
        for k, v in ent.items():
            f.write(struct.pack('<i', len(k)) + k.encode() + struct.pack('<ii', 5, len(v)) + v.tobytes())
        f.write(struct.pack('<i', 3) + b'abc')  # trailer (last call + dataset name) copied verbatim

    out = d / 'tiny-split.gguf'
    rc, log = run(HERE / 'split_experts.py', src, out, '--part', f'hot={d}/hot.tsv', '--part', f'warm={d}/warm.json',
                  '--rest', 'cold', '--imatrix', d / 'in.imatrix', '--imatrix-out', d / 'out.imatrix')
    print(log.strip())
    check(rc == 0, 'split_experts.py runs')

    want = {0: {1: 0, 3: 0, 5: 0, 6: 0, 7: 1}, 1: {2: 1, 4: 1}, 2: {e: 0 for e in range(N_EXP)}}
    r = gguf.GGUFReader(out)
    t = {x.name: x for x in r.tensors}
    for il in range(3):
        mp = np.asarray(t[f'blk.{il}.ffn_exps_part.weight'].data).reshape(2, N_EXP)
        part = [want[il].get(e, 2) for e in range(N_EXP)]
        check(mp[0].tolist() == part, f'blk.{il} map tags {mp[0].tolist()} == {part} (hot list wins the overlap on 3)')
        for p in set(part):
            ids = [e for e in range(N_EXP) if part[e] == p]
            check(mp[1][ids].tolist() == list(range(len(ids))), f'blk.{il}.{TAGS[p]} indices are 0..{len(ids) - 1} in expert order')
            for role in ('up', 'gate', 'down'):
                got = np.asarray(t[f'blk.{il}.ffn_{role}_exps_{TAGS[p]}.weight'].data)
                check(np.array_equal(got, full[(il, role)][ids]), f'blk.{il}.{role}_{TAGS[p]}: experts {ids} copied verbatim')
        check(f'blk.{il}.ffn_up_exps.weight' not in t, f'blk.{il}: full tensors gone')
    check('blk.2.ffn_up_exps_cold.weight' not in t, 'blk.2: an empty partition has no tensor')

    rc, log = run(HERE / 'split_experts_verify.py', src, out, '--bytes')
    print(log.strip())
    check(rc == 0 and 'RESULT PASS' in log, 'verifier passes the split file')

    # corrupt: blk.0 experts 0 and 2 (both cold) point at the same slot
    bad = d / 'tiny-bad.gguf'
    raw = bytearray(out.read_bytes())
    mt = t['blk.0.ffn_exps_part.weight']
    idx_base = int(mt.data_offset) + 4 * N_EXP  # data_offset is absolute in the file
    raw[idx_base + 4 * 2: idx_base + 4 * 3] = raw[idx_base: idx_base + 4]
    bad.write_bytes(bytes(raw))
    rc, log = run(HERE / 'split_experts_verify.py', src, bad, '--bytes')
    print(log.strip())
    check(rc != 0 and 'used' in log, 'verifier rejects a map with a slot used twice')

    # imatrix slices
    data = (d / 'out.imatrix').read_bytes()
    pos, got = 4, {}
    for _ in range(struct.unpack_from('<i', data, 0)[0]):
        (ln,) = struct.unpack_from('<i', data, pos); pos += 4
        name = data[pos:pos + ln].decode(); pos += ln
        _, nv = struct.unpack_from('<ii', data, pos); pos += 8
        got[name] = np.frombuffer(data, dtype=np.float32, count=nv, offset=pos); pos += 4 * nv
    check(data[pos:] == struct.pack('<i', 3) + b'abc', 'imatrix trailer kept')
    part0 = [want[0].get(e, 2) for e in range(N_EXP)]
    for p in set(part0):
        ids = [e for e in range(N_EXP) if part0[e] == p]
        for role, srcname, ne in (('up', 'up', NE0), ('gate', 'up', NE0), ('down', 'down', NE1)):
            v = got.get(f'blk.0.ffn_{role}_exps_{TAGS[p]}.weight')
            exp = ent[f'blk.0.ffn_{srcname}_exps.weight'].reshape(N_EXP, ne)[ids].reshape(-1)
            check(v is not None and np.array_equal(v, exp), f'imatrix blk.0.{role}_{TAGS[p]} = source rows {ids}')

    print('RESULT', 'PASS' if not fails else f'FAIL ({len(fails)})')
    sys.exit(1 if fails else 0)


if __name__ == '__main__':
    main()
