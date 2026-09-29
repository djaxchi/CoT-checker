#!/usr/bin/env python3
"""Build, extract, and locally score the frozen repair diagnostic."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from scripts.analysis.verifier_signal_experiment import write_json  # noqa: E402
from src.analysis.verifier_repair import build_rows  # noqa: E402
from src.analysis.verifier_signal import read_rows, sha256  # noqa: E402


def load(bundle: Path) -> tuple[dict,list[dict]]:
    config=json.loads((bundle/'config.json').read_text())
    rows=read_rows(bundle/'examples.jsonl')
    manifest=json.loads((bundle/'manifest.json').read_text())
    for name,digest in manifest['sha256'].items():
        if sha256(bundle/name)!=digest:
            raise ValueError('frozen bundle changed')
    if rows!=build_rows(config) or len({r['uid'] for r in rows})!=len(rows):
        raise ValueError('data does not reproduce')
    return config,rows


def token_statistics(logits, ids: list[int], start: int, offsets: list[tuple[int,int]],
                     math_start: int) -> dict:
    """Teacher-forced likelihoods: state t-1 predicts token t, never token t itself."""
    import torch
    z=logits[start-1:len(ids)-1].float()
    lp=torch.log_softmax(z,dim=-1)
    targets=torch.tensor(ids[start:],device=lp.device)
    chosen=lp.gather(1,targets[:,None]).squeeze(1)
    entropy=-(lp.exp()*lp).sum(-1)
    math_mask=torch.tensor([end>math_start for _,end in offsets],device=lp.device)
    if len(chosen)!=len(offsets) or not math_mask.any():
        raise ValueError('token statistics alignment failure')
    return dict(token_logprobs=chosen.cpu().tolist(),token_entropies=entropy.cpu().tolist(),
                nll_mean=float(-chosen.mean()),nll_sum=float(-chosen.sum()),
                math_nll_mean=float(-chosen[math_mask].mean()),n_candidate_tokens=len(chosen))


def extract(args,config,rows):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from scripts.encode_prm800k_token_store import encode_split, tokenize_with_offsets
    tokenizer=AutoTokenizer.from_pretrained(config['model'],revision=config['revision'],local_files_only=True)
    model=AutoModelForCausalLM.from_pretrained(config['model'],revision=config['revision'],
                                              local_files_only=True,torch_dtype=torch.bfloat16).to('cuda').eval()
    if model.config._commit_hash!=config['revision'] or model.config.hidden_size!=config['dim']:
        raise ValueError('wrong backbone')
    subset=rows[args.shard_idx::4]
    records=[]
    for row in subset:
        ids,start=tokenize_with_offsets(tokenizer,row,config['max_seq_len'])
        if len(ids)-start>config['t_max']:
            raise ValueError('candidate exceeds probe cap')
    # Reuse the unchanged encoder; observe the same forward outputs to collect likelihoods.
    class ObservedModel:
        def __init__(self):
            self.config=model.config
            self.cursor=0
        def __call__(self,inputs,**kwargs):
            output=model(inputs,**kwargs)
            for i in range(inputs.shape[0]):
                row=subset[self.cursor]
                ids,start=tokenize_with_offsets(tokenizer,row,config['max_seq_len'])
                enc=tokenizer(row['candidate_step'],add_special_tokens=False,return_offsets_mapping=True)
                if enc['input_ids']!=ids[start:]:
                    raise ValueError('candidate tokenizer mismatch')
                stat=token_statistics(output.logits[i],ids,start,enc['offset_mapping'],row['math_start'])
                records.append(dict(uid=row['uid'],**stat))
                self.cursor+=1
            return output
    destination=args.out/'store/repair'/f'shard_{args.shard_idx:02d}'
    if destination.exists():
        raise ValueError('refusing to overwrite extraction')
    observer=ObservedModel()
    pad=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    encode_split(args.bundle/'examples.jsonl',args.out/'store','repair',tokenizer,observer,
                 torch.device('cuda'),config['layer'],config['max_seq_len'],4,pad,args.shard_idx,4,
                 config['model'],None,span_only=True)
    assert observer.cursor==len(subset)
    write_json(destination/'likelihoods.json',records)
    write_json(destination/'extraction.json',dict(dataset_sha256=sha256(args.bundle/'examples.jsonl'),
        config_sha256=sha256(args.bundle/'config.json'),model_revision=config['revision'],
        torch_version=torch.__version__,shard_idx=args.shard_idx,n_rows=len(subset)))


def score(args,config,rows):
    import torch

    from scripts.analysis.verifier_signal_experiment import load_bundle, seed_roster
    from scripts.analysis.verifier_signal_local import validate_checkpoint
    from scripts.onpolicy.score_cells_on_split import score_cell
    from scripts.validate_instruct_leaderboard import cell_tag
    from src.repstore import split_fingerprint
    from src.repstore.store import ShardedRepSplit
    torch.set_num_threads(4)
    assets=ROOT/'results/verifier_signal_v1'
    old_bundle=ROOT/'experiments/verifier_signal_v1_available64'
    old_config,_=load_bundle(old_bundle)
    old_hash=sha256(old_bundle/'examples.jsonl')
    old_store=split_fingerprint(assets/'cluster_reference/store/diagnostic')
    split=args.extraction/'store/repair'
    view=ShardedRepSplit(split)
    actual=view.meta()
    if len(actual)!=len(rows) or {m['uid']:m['label'] for m in actual}!={r['uid']:r['label'] for r in rows}:
        raise ValueError('extracted row/label mismatch')
    manifests=[json.loads(p.read_text()) for p in sorted(split.glob('shard_*/extraction.json'))]
    if len(manifests)!=4 or {m['shard_idx'] for m in manifests}!=set(range(4)):
        raise ValueError('incomplete extraction')
    for m in manifests:
        if m['dataset_sha256']!=sha256(args.bundle/'examples.jsonl') or m['config_sha256']!=sha256(args.bundle/'config.json') or m['model_revision']!=config['revision']:
            raise ValueError('extraction provenance mismatch')
    args.out.mkdir(parents=True,exist_ok=False)
    for (rep,learner),seeds in seed_roster(old_bundle,old_config).items():
        for seed in seeds:
            tag=cell_tag(rep,learner,seed)
            cell=assets/'local_assets/cells'/tag
            ref=json.loads((assets/'cluster_reference/scores'/f'{tag}.json').read_text())
            result=validate_checkpoint(cell,ref,(rep,learner,seed),old_hash,old_store)
            values,_,meta=score_cell(cell,result,split,None,torch.device('cpu'),32,config['t_max'])
            mapped={m['uid']:float(s) for m,s in zip(meta,values)}
            if set(mapped)!={r['uid'] for r in rows} or not all(0<=s<=1 for s in mapped.values()):
                raise ValueError('invalid score output')
            write_json(args.out/f'{tag}.json',dict(cell=tag,rep=rep,learner=learner,seed=seed,
                scores=mapped,model_sha256=sha256(cell/'model.pt'),results_sha256=sha256(cell/'results.json'),
                dataset_sha256=sha256(args.bundle/'examples.jsonl'),store_fingerprint=split_fingerprint(split),
                source_test_validation='inherited from hash-matched v1 reference',torch_version=torch.__version__))
            print('Scored',tag,flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase',choices=['build','validate','extract','score'])
    p.add_argument('--bundle',type=Path,default=ROOT/'experiments/verifier_repair_v1')
    p.add_argument('--out',type=Path)
    p.add_argument('--extraction',type=Path)
    p.add_argument('--shard-idx',type=int,choices=range(4),default=0)
    args=p.parse_args()
    if args.phase=='build':
        config=json.loads((args.bundle/'config.json').read_text())
        rows=build_rows(config)
        if (args.bundle/'examples.jsonl').exists():
            raise ValueError('refusing to replace frozen bundle')
        (args.bundle/'examples.jsonl').write_text(''.join(json.dumps(r,sort_keys=True)+'\n' for r in rows))
        write_json(args.bundle/'manifest.json',dict(n_rows=len(rows),n_families=len({r['family_id'] for r in rows}),
            sha256={name:sha256(args.bundle/name) for name in ['config.json','examples.jsonl']},
            human_review='pending; exact arithmetic and generation tested'))
        return
    config,rows=load(args.bundle)
    if args.phase=='validate':
        print('Validated',len(rows),'rows')
    elif args.out is None:
        p.error('--out required')
    elif args.phase=='extract':
        extract(args,config,rows)
    elif args.extraction is None:
        p.error('--extraction required')
    else:
        score(args,config,rows)


if __name__=='__main__':
    main()
