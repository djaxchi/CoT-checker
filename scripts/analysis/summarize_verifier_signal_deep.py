#!/usr/bin/env python3
"""Summarize exploratory readout interventions with family-level uncertainty."""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from src.analysis.verifier_signal import read_rows, summarize  # noqa: E402


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    args=parser.parse_args()
    rows=read_rows(ROOT/'experiments/verifier_signal_v1_available64/examples.jsonl')
    data=json.loads((args.run/'readouts.json').read_text())
    groups=defaultdict(list)
    for row in rows:
        if row['role']=='target':
            groups[row['family_id'],row['arm'],row['domain'],row['partition'],row['style']].append(row)
    by_model=defaultdict(lambda:defaultdict(list))
    for cell in data['cells']:
        for mode,logits in cell['logits'].items():
            for key,members in groups.items():
                z=np.array([logits[r['uid']] for r in members])
                probability=1/(1+np.exp(-z))
                labels=np.array([r['local_invalid'] for r in members])
                lookup={(r['prefix_variant'],r['candidate_variant']):p for r,p in zip(members,probability)}
                d0=lookup[0,1]-lookup[0,0]
                d1=lookup[1,0]-lookup[1,1]
                metrics={'valid_mean_score':probability[labels==0].mean(),
                         'invalid_mean_score':probability[labels==1].mean(),
                         'valid_mean_logit':z[labels==0].mean(),
                         'invalid_mean_logit':z[labels==1].mean(),
                         'both_preferences_correct':float(d0>0 and d1>0),
                         'local_contrast_mean':(d0+d1)/2}
                if mode != 'full':
                    original=np.array([cell['logits']['full'][r['uid']] for r in members])
                    metrics['mean_absolute_logit_change']=np.abs(z-original).mean()
                by_model[cell['rep'],cell['learner'],mode][key].append(metrics)
    output=[]
    for (rep,learner,mode),families in by_model.items():
        averaged=[]
        for (family,arm,domain,partition,style),members in families.items():
            averaged.append(dict(family_id=family,arm=arm,domain=domain,partition=partition,style=style,
                metrics={k:float(np.mean([m[k] for m in members])) for k in members[0]}))
        output.append(dict(rep=rep,learner=learner,mode=mode,summary=summarize(averaged,2000,731)))
    (args.run/'behavior_summary.json').write_text(json.dumps(output,indent=2,allow_nan=False)+'\n')
    print('Wrote',args.run/'behavior_summary.json')


if __name__=='__main__':
    main()
