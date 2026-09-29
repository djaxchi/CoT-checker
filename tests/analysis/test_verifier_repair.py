import json
from pathlib import Path

import numpy as np
import pytest
import torch

from scripts.analysis.verifier_repair_experiment import token_statistics
from src.analysis.verifier_repair import build_rows


def test_crossed_roster_and_exact_arithmetic():
    config=json.loads(Path('experiments/verifier_repair_v1/config.json').read_text())
    rows=build_rows(config)
    assert rows==build_rows(config) and len(rows)==2048
    assert len({r['uid'] for r in rows})==2048
    for r in rows:
        a,b,x=(r['witness'][k] for k in ['a','b','x'])
        answer=x+r['conclusion_wrong'] if r['domain']=='affine' else a*x+a*r['conclusion_wrong']
        actual=a*answer+b==a*x+b if r['domain']=='affine' else answer==a*x
        assert actual==(not r['conclusion_wrong'])
        if r['stage']=='followup':
            assert r['label']==0 and r['candidate_math_invalid']==0
        elif r['form']=='direct':
            assert r['label']==(r['prefix_wrong']!=r['conclusion_wrong'])
        else:
            assert r['label']==r['conclusion_wrong']
    families={}
    for r in rows:
        families.setdefault(r['family_id'],set()).add(r['partition'])
    assert len(families)==64 and all(len(v)==1 for v in families.values())
    assert sum(v=={'test'} for v in families.values())==48


def test_likelihood_predicts_next_token_not_current_token():
    ids=[0,1,2,1]
    z=torch.tensor([[4.,0.,0.],[0.,0.,4.],[0.,2.,0.],[0.,0.,4.]])
    stat=token_statistics(z,ids,2,[(0,1),(1,2)],1)
    first=float(np.log1p(2*np.exp(-4)))
    last=float(np.log1p(2*np.exp(-2)))
    assert stat['nll_mean']==pytest.approx((first+last)/2,abs=1e-6)
    assert stat['math_nll_mean']==pytest.approx(last,abs=1e-6)
    assert stat['n_candidate_tokens']==2
    assert np.isfinite(stat['token_entropies']).all()


def test_repair_metrics_separate_labels_and_likelihood():
    from scripts.analysis.verifier_repair_analysis import metrics_for_family
    config=json.loads(Path('experiments/verifier_repair_v1/config.json').read_text())
    config['families_per_domain']=1
    config['domains']=['affine']
    rows=build_rows(config)
    scores={r['uid']:0.1+0.8*r['label'] for r in rows}
    nll={r['uid']:1.+r['conclusion_wrong'] for r in rows}
    m=metrics_for_family(rows,scores,nll)['metrics']
    assert m['repair_discrimination']==pytest.approx(.8)
    assert m['repair_vs_jump']==pytest.approx(.8)
    assert m['wording_p1_c0']==0
    assert m['followup_global_contrast']==0
    assert m['both_prefer_genuine']==1
    reversed_nll={uid:3.-value for uid,value in nll.items()}
    m=metrics_for_family(rows,scores,reversed_nll)['metrics']
    assert m['only_probe_prefers_genuine']==1


def test_full_analysis_mixed_seed_roster(tmp_path, monkeypatch):
    from scripts.analysis.verifier_repair_analysis import main
    from scripts.analysis.verifier_repair_experiment import load
    from scripts.analysis.verifier_signal_experiment import load_bundle, seed_roster
    from scripts.validate_instruct_leaderboard import cell_tag
    from src.analysis.verifier_signal import sha256
    bundle=Path('experiments/verifier_repair_v1').resolve()
    _,rows=load(bundle)
    extraction=tmp_path/'extraction'
    scores=tmp_path/'scores'
    scores.mkdir()
    for shard in range(4):
        folder=extraction/'store/repair'/f'shard_{shard:02d}'
        folder.mkdir(parents=True)
        (folder/'likelihoods.json').write_text(json.dumps([
            {'uid':r['uid'],'math_nll_mean':1.+r['conclusion_wrong']} for r in rows[shard::4]]))
    old=Path('experiments/verifier_signal_v1_available64')
    config,_=load_bundle(old)
    for (rep,learner),seeds in seed_roster(old,config).items():
        for seed in seeds:
            tag=cell_tag(rep,learner,seed)
            (scores/f'{tag}.json').write_text(json.dumps(dict(cell=tag,rep=rep,learner=learner,seed=seed,
                dataset_sha256=sha256(bundle/'examples.jsonl'),store_fingerprint='stub',
                scores={r['uid']:0.1+0.8*r['label'] for r in rows})))
    out=tmp_path/'analysis'
    monkeypatch.setattr('sys.argv',['analysis','--bundle',str(bundle),'--extraction',str(extraction),
                                    '--scores',str(scores),'--out',str(out)])
    main()
    result=json.loads((out/'analysis.json').read_text())
    assert len(result['per_cell'])==16 and len(result['seed_averaged'])==6
    for cell in result['seed_averaged']:
        gaps=[r for r in cell['summary'] if r['metric']=='repair_discrimination']
        assert all(r['mean']==pytest.approx(.8) for r in gaps)
    next(scores.glob('*.json')).unlink()
    with pytest.raises(ValueError,match='wrong checkpoint roster'):
        main()
