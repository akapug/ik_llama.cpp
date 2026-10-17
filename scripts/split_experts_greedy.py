#!/usr/bin/env python3
"""Greedy decode of every prompt in a JSON list through a running llama-server; one JSON line per prompt with the
token strings and the top-2 probabilities (exact floats), for a token-for-token compare of two models/configs.
usage: split_experts_greedy.py PROMPTS.json OUT.jsonl [--npredict 32]   (PORT env, default 8790)
       split_experts_greedy.py --compare A.jsonl B.jsonl"""
import json, os, sys, urllib.request

if sys.argv[1] == '--compare':
    a = [json.loads(l) for l in open(sys.argv[2])]
    b = [json.loads(l) for l in open(sys.argv[3])]
    ok = len(a) == len(b) and len(a) > 0
    for x, y in zip(a, b):
        tok = x['tokens'] == y['tokens']
        prob = x['probs'] == y['probs']
        diff = next((i for i, (u, v) in enumerate(zip(x['tokens'], y['tokens'])) if u != v), None)
        print(f"prompt {x['prompt']}: {x['n_prompt']} prompt tok, {len(x['tokens'])} gen: tokens {'MATCH' if tok else f'DIFF at {diff}'}, "
              f"probs {'BIT-IDENTICAL' if prob else 'differ'}")
        ok = ok and tok and prob
    print('RESULT', 'PASS' if ok else 'FAIL')
    sys.exit(0 if ok else 1)

prompts = json.load(open(sys.argv[1]))
n = int(sys.argv[sys.argv.index('--npredict') + 1]) if '--npredict' in sys.argv else 32
url = f"http://127.0.0.1:{os.environ.get('PORT', '8790')}/completion"
with open(sys.argv[2], 'w') as fo:
    for i, p in enumerate(prompts):
        body = {'prompt': p + '\n', 'n_predict': n, 'temperature': 0, 'cache_prompt': False, 'ignore_eos': True, 'n_probs': 2}
        req = urllib.request.Request(url, json.dumps(body).encode(), {'Content-Type': 'application/json'})
        d = json.loads(urllib.request.urlopen(req, timeout=7200).read())
        cp = d.get('completion_probabilities') or []
        toks = [e.get('content', e.get('token')) for e in cp]
        probs = [[q.get('prob', q.get('logprob')) for q in (e.get('probs') or e.get('top_probs') or [])] for e in cp]
        fo.write(json.dumps({'prompt': i, 'n_prompt': d['timings']['prompt_n'], 'tokens': toks, 'probs': probs}) + '\n')
        print(f"prompt {i}: {d['timings']['prompt_n']} prompt tok, {len(toks)} generated", flush=True)
