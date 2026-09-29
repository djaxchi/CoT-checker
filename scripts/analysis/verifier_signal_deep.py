#!/usr/bin/env python3
"""Exploratory frozen-verifier input interventions on the existing diagnostic."""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.analysis.verifier_signal_experiment import (  # noqa: E402
    load_bundle,
    seed_roster,
    write_json,
)
from scripts.analysis.verifier_signal_local import compare_scores, validate_checkpoint  # noqa: E402
from scripts.validate_instruct_leaderboard import cell_tag  # noqa: E402
from src.analysis.verifier_signal import family_metrics, sha256, summarize  # noqa: E402
from src.harness.learners import build_learner  # noqa: E402
from src.repstore import split_fingerprint  # noqa: E402
from src.repstore.store import ShardedRepSplit  # noqa: E402


def attention_decomposition(a0: np.ndarray, v0: np.ndarray,
                            a1: np.ndarray, v1: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Exact symmetric decomposition: value change plus routing change."""
    return (a0 + a1) * (v1 - v0) / 2, (v0 + v1) * (a1 - a0) / 2


def patch_states(recipient: np.ndarray, donor: np.ndarray, positions: list[int]) -> np.ndarray:
    if recipient.shape != donor.shape:
        raise ValueError('paired token states must have identical shapes')
    output = recipient.copy()
    output[positions] = donor[positions]
    return output


def patch_norms(recipient: np.ndarray, donor: np.ndarray) -> np.ndarray:
    """Transfer per-token L2 norms while retaining recipient directions."""
    a = np.linalg.norm(recipient, axis=-1, keepdims=True)
    b = np.linalg.norm(donor, axis=-1, keepdims=True)
    if (a == 0).any() or (b == 0).any():
        raise ValueError('zero token norm')
    return recipient * (b / a)


@torch.inference_mode()
def logits(model, states: list[np.ndarray], mode: str = 'full',
           tokens: list[list[str]] | None = None, rep: str = 'step_tokens') -> np.ndarray:
    output = []
    for start in range(0, len(states), 32):
        batch = states[start:start+32]
        if rep != 'step_tokens':
            x = np.stack([s[-1] if rep == 'last_token' else s.mean(0) for s in batch])
            # The training adapter stores derived pooled vectors in float16.
            x = x.astype(np.float16).astype(np.float32)
            z = model(torch.from_numpy(x), None)
        else:
            x = np.zeros((len(batch), max(map(len, batch)), batch[0].shape[1]), dtype=np.float32)
            mask = np.zeros(x.shape[:2], dtype=np.float32)
            for i, state in enumerate(batch):
                x[i, :len(state)] = state[::-1] if mode == 'reverse_states' else state
                mask[i, :len(state)] = 1
                if mode == 'last_only':
                    mask[i, :len(state)-1] = 0
                elif mode == 'drop_last':
                    mask[i, len(state)-1] = 0
                elif mode in ('digits_only', 'no_digits'):
                    digits = np.array([t.strip().isdigit() for t in tokens[start+i]])
                    mask[i, :len(state)] = digits if mode == 'digits_only' else ~digits
                if mask[i].sum() == 0:
                    raise ValueError('empty intervention mask')
            z = model(torch.from_numpy(x), torch.from_numpy(mask))
        output.extend(z.tolist())
    return np.array(output)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.manual_seed(20260929)
    bundle = ROOT / 'experiments/verifier_signal_v1_available64'
    assets = ROOT / 'results/verifier_signal_v1'
    config, rows = load_bundle(bundle)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(assets / 'tokenizer', local_files_only=True)
    view = ShardedRepSplit(assets / 'cluster_reference/store/diagnostic')
    meta = view.meta()
    assert [r['uid'] for r in rows] == [m['uid'] for m in meta]
    states = [view.item(i)[m['step_start_idx']:] for i, m in enumerate(meta)]
    tokens = [[tokenizer.decode([t]) for t in tokenizer(r['candidate_step'], add_special_tokens=False)['input_ids']]
              for r in rows]
    assert all(len(s) == len(t) for s, t in zip(states, tokens))
    store_hash = split_fingerprint(assets / 'cluster_reference/store/diagnostic')
    data_hash = sha256(bundle / 'examples.jsonl')
    lookup = {(r['family_id'], r['prefix_variant'], r['candidate_variant'], r['style']): i
              for i, r in enumerate(rows) if r['role'] == 'target'}
    cells = []
    patch_records = []
    attention_records = []
    for (rep, learner), seeds in seed_roster(bundle, config).items():
        for seed in seeds:
            tag = cell_tag(rep, learner, seed)
            folder = assets / 'local_assets/cells' / tag
            reference = json.loads((assets / 'cluster_reference/scores' / f'{tag}.json').read_text())
            result = validate_checkpoint(folder, reference, (rep, learner, seed), data_hash, store_hash)
            model = build_learner(learner, result['dim'], dropout=0.1).eval()
            model.load_state_dict(torch.load(folder / 'model.pt', map_location='cpu', weights_only=True))
            full = logits(model, states, rep=rep)
            probs = torch.sigmoid(torch.tensor(full)).numpy()
            compare_scores(dict(zip([r['uid'] for r in rows], probs)), reference['scores'])
            modes = {'full': full}
            if learner in ('attn_query', 'transformer:d512,l2,f2048,h8'):
                for mode in ('last_only', 'drop_last', 'digits_only', 'no_digits', 'reverse_states'):
                    modes[mode] = logits(model, states, mode, tokens)
            attention = []
            if learner == 'attn_query':
                with torch.inference_mode():
                    for state in states:
                        x = torch.from_numpy(state)
                        a = torch.softmax(x @ model.q * model.scale, dim=0).numpy()
                        v = (x @ model.head.weight[0]).numpy()
                        attention.append((a, v))
                    bias = model.head.bias.item()
                modes['uniform_attention'] = np.array([v.mean()+bias for _, v in attention])
                assert np.max(np.abs(np.array([a @ v+bias for a,v in attention])-full)) < 1e-4
                for r, tok, (a,v) in zip(rows, tokens, attention):
                    attention_records.append(dict(cell=tag, uid=r['uid'], tokens=tok,
                                                  weights=a.tolist(), values=v.tolist(), bias=bias))
            mode_summaries = {}
            for mode, values in modes.items():
                probability = torch.sigmoid(torch.tensor(values)).numpy()
                metrics = family_metrics(rows, dict(zip([r['uid'] for r in rows], probability)))
                mode_summaries[mode] = summarize(metrics, 2000, 731)
            cells.append(dict(cell=tag, rep=rep, learner=learner, seed=seed,
                              logits={mode: dict(zip([r['uid'] for r in rows], z.tolist())) for mode,z in modes.items()},
                              summary=mode_summaries))
            if learner in ('attn_query', 'transformer:d512,l2,f2048,h8'):
                hybrids, specs = [], []
                for i, row in enumerate(rows):
                    if row['role'] != 'target' or row['style'] != 'plain' or row['local_invalid']:
                        continue
                    j = lookup[row['family_id'], 1-row['prefix_variant'], row['candidate_variant'], 'plain']
                    assert rows[j]['local_invalid'] == 1 and tokens[i] == tokens[j]
                    groups = {'first': [0], 'last': [len(tokens[i])-1],
                              'digits': [k for k,t in enumerate(tokens[i]) if t.strip().isdigit()],
                              'non_digits': [k for k,t in enumerate(tokens[i]) if not t.strip().isdigit()],
                              'all': list(range(len(tokens[i])))}
                    record = dict(cell=tag, family_id=row['family_id'], arm=row['arm'], domain=row['domain'],
                                  partition=row['partition'], candidate_variant=row['candidate_variant'],
                                  valid_uid=row['uid'], invalid_uid=rows[j]['uid'],
                                  full_gap=float(full[j]-full[i]), patches={})
                    if attention:
                        value, routing = attention_decomposition(*attention[i], *attention[j])
                        assert abs(value.sum()+routing.sum()-(full[j]-full[i])) < 1e-4
                        record['decomposition'] = dict(value=float(value.sum()), routing=float(routing.sum()),
                                                       token_values=value.tolist(), token_routing=routing.tolist())
                    patch_records.append(record)
                    for name, positions in groups.items():
                        hybrids += [patch_states(states[i], states[j], positions),
                                    patch_states(states[j], states[i], positions)]
                        specs.append((record, name, i, j))
                    hybrids += [patch_norms(states[i], states[j]), patch_norms(states[j], states[i])]
                    specs.append((record, 'norms_only', i, j))
                    hybrids += [patch_norms(states[j], states[i]), patch_norms(states[i], states[j])]
                    specs.append((record, 'directions_only', i, j))
                patched = logits(model, hybrids)
                for k, (record, name, i, j) in enumerate(specs):
                    forward, backward = patched[2*k:2*k+2]
                    record['patches'][name] = {'valid_to_invalid_shift': float(forward-full[i]),
                                              'invalid_to_valid_shift': float(full[j]-backward),
                                              'symmetric_shift': float((forward-full[i]+full[j]-backward)/2)}
                    if name == 'all':
                        assert abs(forward-full[j]) < 1e-4 and abs(backward-full[i]) < 1e-4
            print('Investigated', tag, flush=True)
    write_json(args.out / 'readouts.json', {'exploratory': True, 'dataset_sha256': data_hash,
               'store_fingerprint': store_hash, 'cells': cells})
    write_json(args.out / 'patches.json', patch_records)
    write_json(args.out / 'attention.json', attention_records)
    # Average paired candidates and seeds within family before descriptive bootstrap.
    grouped = defaultdict(lambda: defaultdict(list))
    for record in patch_records:
        learner = record['cell'].split('__')[1]
        metrics = {'full_gap': record['full_gap']}
        metrics.update({f'patch_{key}': val['symmetric_shift'] for key,val in record['patches'].items()})
        if 'decomposition' in record:
            metrics.update({key: record['decomposition'][key] for key in ('value', 'routing')})
        key = (learner, record['family_id'], record['arm'], record['domain'], record['partition'])
        for metric, value in metrics.items():
            grouped[key][metric].append(value)
    by_model = defaultdict(list)
    for (learner, family, arm, domain, partition), values in grouped.items():
        by_model[learner].append(dict(family_id=family, arm=arm, domain=domain,
            partition=partition, style='plain', metrics={k:float(np.mean(v)) for k,v in values.items()}))
    write_json(args.out / 'patch_summary.json', {key:summarize(val, 2000, 731) for key,val in by_model.items()})


if __name__ == '__main__':
    main()
