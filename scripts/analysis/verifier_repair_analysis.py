#!/usr/bin/env python3
"""Prespecified family contrasts for genuine versus false repair."""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from scripts.analysis.verifier_repair_experiment import load  # noqa: E402
from scripts.analysis.verifier_signal_experiment import (  # noqa: E402
    load_bundle,
    seed_roster,
    write_json,
)
from src.analysis.verifier_signal import sha256, summarize  # noqa: E402


def metrics_for_family(rows: list[dict], scores: dict[str,float], nll: dict[str,float]) -> dict:
    ids={(r['prefix_wrong'],r['conclusion_wrong'],r['form'],r['stage']):r['uid'] for r in rows}
    s={k:scores[u] for k,u in ids.items()}
    true=s[1,0,'repair','target']
    false=s[1,1,'repair','target']
    gap=false-true
    nll_gap=nll[ids[1,1,'repair','target']]-nll[ids[1,0,'repair','target']]
    m={'genuine_repair_score':true,'false_repair_score':false,'repair_discrimination':gap,
       'repair_preference_correct':float(gap>0),'repair_preference_tied':float(gap==0),
       'repair_vs_jump':s[1,0,'direct','target']-true,
       'repair_wrong_vs_clean_prefix':true-s[0,0,'repair','target'],
       'genuine_followup_score':s[1,0,'repair','followup'],
       'false_repair_followup_score':s[1,1,'repair','followup'],
       'followup_global_contrast':s[1,1,'repair','followup']-s[1,0,'repair','followup'],
       'followup_repair_vs_jump_history':s[1,0,'repair','followup']-s[1,0,'direct','followup'],
       'math_nll_discrimination':nll_gap,
       'likelihood_preference_correct':float(nll_gap>0),
       'likelihood_preference_tied':float(nll_gap==0),
       'both_prefer_genuine':float(gap>0 and nll_gap>0),
       'only_probe_prefers_genuine':float(gap>0 and nll_gap<=0),
       'only_likelihood_prefers_genuine':float(gap<=0 and nll_gap>0),
       'neither_prefers_genuine':float(gap<=0 and nll_gap<=0)}
    for p in (0,1):
        for c in (0,1):
            m[f'wording_p{p}_c{c}']=s[p,c,'repair','target']-s[p,c,'check','target']
    for form in ('direct','recompute','check'):
        m[f'{form}_correct_score']=s[1,0,form,'target']
        m[f'{form}_incorrect_score']=s[1,1,form,'target']
        m[f'{form}_discrimination']=s[1,1,form,'target']-s[1,0,form,'target']
    first=rows[0]
    return dict(family_id=first['family_id'],domain=first['domain'],partition=first['partition'],
                arm='repair',style='primary',metrics=m)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle',type=Path,default=ROOT/'experiments/verifier_repair_v1')
    p.add_argument('--extraction',type=Path,required=True)
    p.add_argument('--scores',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args()
    config,rows=load(args.bundle)
    likelihoods=[]
    for path in sorted((args.extraction/'store/repair').glob('shard_*/likelihoods.json')):
        likelihoods+=json.loads(path.read_text())
    if len(likelihoods)!=len(rows) or {r['uid'] for r in likelihoods}!={r['uid'] for r in rows}:
        raise ValueError('incomplete likelihoods')
    nll={r['uid']:r['math_nll_mean'] for r in likelihoods}
    if not all(np.isfinite(v) and v>=0 for v in nll.values()):
        raise ValueError('invalid likelihood')
    groups=defaultdict(list)
    for r in rows:
        groups[r['family_id']].append(r)
    roster_bundle=ROOT/'experiments/verifier_signal_v1_available64'
    old_config,_=load_bundle(roster_bundle)
    expected={(rep,learner,seed) for (rep,learner),seeds in seed_roster(roster_bundle,old_config).items() for seed in seeds}
    per_cell=[]
    by_model=defaultdict(list)
    seen=set()
    store_hashes=set()
    for path in sorted(args.scores.glob('*.json')):
        data=json.loads(path.read_text())
        identity=(data['rep'],data['learner'],data['seed'])
        if identity in seen or data['dataset_sha256']!=sha256(args.bundle/'examples.jsonl'):
            raise ValueError('duplicate or mismatched score data')
        seen.add(identity)
        store_hashes.add(data['store_fingerprint'])
        if set(data['scores'])!=set(nll) or not all(np.isfinite(v) and 0<=v<=1 for v in data['scores'].values()):
            raise ValueError('invalid scores')
        fm=[metrics_for_family(members,data['scores'],nll) for members in groups.values()]
        per_cell.append(dict(cell=data['cell'],family_metrics=fm,summary=summarize(fm,2000,731)))
        by_model[identity[:2]].append((data['seed'],fm))
    if seen!=expected or len(store_hashes)!=1:
        raise ValueError('wrong checkpoint roster or mixed stores')
    averaged=[]
    for (rep,learner),members in by_model.items():
        fm=[]
        for i,item in enumerate(members[0][1]):
            row=dict(item)
            row['metrics']={k:float(np.mean([values[i]['metrics'][k] for _,values in members])) for k in item['metrics']}
            fm.append(row)
        averaged.append(dict(rep=rep,learner=learner,seeds=sorted(s for s,_ in members),
                             family_metrics=fm,summary=summarize(fm,2000,731)))
    args.out.mkdir(parents=True,exist_ok=False)
    write_json(args.out/'analysis.json',dict(dataset_sha256=sha256(args.bundle/'examples.jsonl'),
        per_cell=per_cell,seed_averaged=averaged,bootstrap_unit='numeric family within domain after averaging seeds',
        likelihood_files_sha256={p.name+':'+p.parent.name:sha256(p) for p in sorted((args.extraction/'store/repair').glob('shard_*/likelihoods.json'))}))
    lines=['# Genuine versus false repair: primary results','',
           'Held-out numeric families; 24 per domain. Scores are uncalibrated sigmoid outputs.','',
           '| Probe | Seeds | Domain | Genuine | False | False minus genuine, 95% interval | Correct preference | Repair versus jump |',
           '|---|---|---|---:|---:|---|---:|---:|']
    for c in averaged:
        for domain in config['domains']:
            r={m['metric']:m for m in c['summary'] if m['partition']=='test' and m['domain']==domain}
            gap=r['repair_discrimination']
            lines.append(f"| {c['rep']} / {c['learner']} | {c['seeds']} | {domain} | {r['genuine_repair_score']['mean']:.4f} | {r['false_repair_score']['mean']:.4f} | {gap['mean']:.4f} [{gap['ci95'][0]:.4f}, {gap['ci95'][1]:.4f}] | {r['repair_preference_correct']['mean']:.3f} | {r['repair_vs_jump']['mean']:.4f} |")
    (args.out/'summary.md').write_text('\n'.join(lines)+'\n')
    print('Wrote',args.out/'summary.md')


if __name__=='__main__':
    main()
