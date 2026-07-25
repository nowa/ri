#!/usr/bin/env python3
"""Compare pi vs ri differential payload dumps."""
import json
import sys

def canon(v):
    if isinstance(v, dict):
        return {k: canon(v[k]) for k in sorted(v)}
    if isinstance(v, list):
        return [canon(x) for x in v]
    return v

def diff_paths(a, b, path=""):
    out = []
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            p = f"{path}.{k}" if path else k
            if k not in a:
                out.append(f"  {p}: only in ri = {json.dumps(b[k])[:120]}")
            elif k not in b:
                out.append(f"  {p}: only in pi = {json.dumps(a[k])[:120]}")
            else:
                out.extend(diff_paths(a[k], b[k], p))
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            out.append(f"  {path}: length pi={len(a)} ri={len(b)}")
        for i, (x, y) in enumerate(zip(a, b)):
            out.extend(diff_paths(x, y, f"{path}[{i}]"))
    elif a != b:
        out.append(f"  {path}: pi={json.dumps(a)[:120]} ri={json.dumps(b)[:120]}")
    return out

pi = json.load(open(sys.argv[1]))
ri = json.load(open(sys.argv[2]))

match = mismatch = status_diff = 0
for case_id in pi:
    p, r = pi[case_id], ri.get(case_id)
    if r is None:
        print(f"## {case_id}: MISSING in ri dump")
        mismatch += 1
        continue
    p_err, r_err = "error" in p, "error" in r
    if p_err != r_err:
        status_diff += 1
        print(f"## {case_id}: STATUS pi={'error: ' + p['error'][:100] if p_err else 'payload'} | ri={'error: ' + r['error'][:100] if r_err else 'payload'}")
        continue
    if p_err:
        match += 1  # both errored
        continue
    if canon(p["payload"]) == canon(r["payload"]):
        match += 1
        continue
    mismatch += 1
    print(f"## {case_id}: PAYLOAD DIFF")
    for line in diff_paths(canon(p["payload"]), canon(r["payload"]))[:25]:
        print(line)

print(f"\nTOTAL: {len(pi)} cases — {match} match, {mismatch} payload diffs, {status_diff} status diffs")
