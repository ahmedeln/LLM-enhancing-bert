import sys
import json
import numpy as np
from collections import Counter
import re


def iter_json_values(s):
    """Yield JSON values found in string s even when concatenated without newlines.
    Uses json.JSONDecoder.raw_decode to parse successive values."""
    decoder = json.JSONDecoder()
    idx = 0
    s_len = len(s)
    while idx < s_len:
        # skip whitespace
        while idx < s_len and s[idx].isspace():
            idx += 1
        # skip stray commas separating concatenated JSON values
        while idx < s_len and s[idx] == ',':
            idx += 1
        if idx >= s_len:
            break
        try:
            obj, end = decoder.raw_decode(s, idx)
        except Exception:
            # can't decode further from this position
            break
        yield obj
        idx = end


def extract_json_objects(s):
    """Extract substring representations of JSON objects by scanning for
    balanced braces. This is tolerant of objects that span newlines or are
    concatenated with other tokens."""
    objs = []
    i = 0
    n = len(s)
    while i < n:
        if s[i] != '{':
            i += 1
            continue
        start = i
        i += 1
        depth = 1
        in_str = False
        esc = False
        while i < n and depth > 0:
            ch = s[i]
            if in_str:
                if esc:
                    esc = False
                else:
                    if ch == '\\':
                        esc = True
                    elif ch == '"':
                        in_str = False
            else:
                if ch == '"':
                    in_str = True
                elif ch == '{':
                    depth += 1
                elif ch == '}':
                    depth -= 1
            i += 1
        if depth == 0:
            objs.append(s[start:i])
        else:
            break
    return objs


if len(sys.argv) < 2:
    print("Usage: summarize_features.py <features_jsonl>")
    sys.exit(1)

path = sys.argv[1]

lines = 0
total_tokens = 0
layer_counts = Counter()
emb_dim = None
values_sum = 0.0
values_sq_sum = 0.0
values_count = 0

with open(path, 'r', encoding='utf-8') as f:
    whole = f.read()

raw_lines = len(whole.splitlines()) if whole else 0
lines = raw_lines

# Parse all JSON values found in the whole file (handles concatenated
# objects, arrays and fragments split across lines).
found_any_global = False
# First, try extracting well-formed JSON objects by scanning for balanced
# braces; this recovers objects even when arrays or values are split across
# lines or concatenated in odd ways.
objs = extract_json_objects(whole)

for obj_str in objs:
    try:
        obj = json.loads(obj_str)
    except Exception:
        continue
    found_any_global = True
    # If this object is a wrapper with a 'features' list, iterate those entries
    if 'features' in obj and isinstance(obj['features'], list):
        for feat in obj['features']:
            if not isinstance(feat, dict):
                continue
            # process each feature entry as a token-like object
            if 'token' in feat and 'layers' in feat:
                total_tokens += 1
                layers = feat.get('layers', [])
                layer_counts.update([len(layers)])
                for l in layers:
                    vals = l.get('values') or l.get('vector') or []
                    if emb_dim is None and isinstance(vals, list) and len(vals) > 0:
                        emb_dim = len(vals)
                    if vals:
                        arr = np.array(vals, dtype=np.float32)
                        values_sum += arr.sum()
                        values_sq_sum += (arr * arr).sum()
                        values_count += arr.size
        continue
    if not isinstance(obj, dict):
        # If it's a top-level list/object other than dict, handle basic lists
        if isinstance(obj, list):
            total_tokens += 1
            if emb_dim is None:
                emb_dim = len(obj)
            arr = np.array(obj, dtype=np.float32)
            values_sum += arr.sum()
            values_sq_sum += (arr * arr).sum()
            values_count += arr.size
        continue

    # If it's an object with token/layers (standard extractor format)
    if 'token' in obj and 'layers' in obj:
        total_tokens += 1
        layers = obj.get('layers', [])
        layer_counts.update([len(layers)])
        for l in layers:
            vals = l.get('values') or l.get('vector') or []
            if emb_dim is None and isinstance(vals, list) and len(vals) > 0:
                emb_dim = len(vals)
            if vals:
                arr = np.array(vals, dtype=np.float32)
                values_sum += arr.sum()
                values_sq_sum += (arr * arr).sum()
                values_count += arr.size
        continue

    # Other object formats that may include 'values' or nested arrays
    if 'values' in obj and isinstance(obj['values'], list):
        vals = obj['values']
        total_tokens += 1
        if emb_dim is None and len(vals) > 0:
            emb_dim = len(vals)
        arr = np.array(vals, dtype=np.float32)
        values_sum += arr.sum()
        values_sq_sum += (arr * arr).sum()
        values_count += arr.size

# Fallback: if we didn't find any JSON objects with vectors, try scanning
# the whole file for long runs of floats (handles lines split across newlines
# where arrays were emitted without braces).
if not found_any_global:
    floats = re.findall(r'[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?', whole)
    if len(floats) >= 16:
        try:
            arr = np.array([float(x) for x in floats], dtype=np.float32)
            total_tokens += 1
            if emb_dim is None:
                emb_dim = arr.size
            values_sum += arr.sum()
            values_sq_sum += (arr * arr).sum()
            values_count += arr.size
        except Exception:
            pass

print("File:", path)
print("Lines (JSONL rows):", lines)
print("Token-like entries seen:", total_tokens)
print("Embedding dim (inferred):", emb_dim)
if layer_counts:
    most_common = layer_counts.most_common(3)
    print("Layers per-token counts (top 3):", most_common)
if values_count:
    mean = values_sum / values_count
    var = (values_sq_sum / values_count) - (mean * mean)
    std = np.sqrt(max(var, 0.0))
    print(f"Values count: {values_count}, mean={mean:.6f}, std={std:.6f}")
else:
    print("No numeric vectors found to summarize.")
