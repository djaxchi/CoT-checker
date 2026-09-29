"""Frozen crossed repair-language and arithmetic-validity diagnostic."""
from __future__ import annotations

import random


def build_rows(config: dict) -> list[dict]:
    rng=random.Random(config['seed'])
    rows=[]
    for domain in config['domains']:
        used=set()
        for index in range(config['families_per_domain']):
            while True:
                a,b,x=rng.randint(2,9),rng.randint(3,40),rng.randint(2,35)
                key=(a,b,x) if domain=='affine' else (a,x)
                if key not in used:
                    used.add(key)
                    break
            family=f'{domain}_{index:03d}'
            true=a*x
            problem=(f'Solve {a}*x + {b} = {true+b} for real x.' if domain=='affine'
                     else f'Compute the product {a}*{x}.')
            for prefix_wrong in (0,1):
                intermediate=true+a*prefix_wrong
                prefix=(f'Subtracting {b} from both sides gives {a}*x = {intermediate}.' if domain=='affine'
                        else f'Multiplying {a} by {x} gives {intermediate}.')
                for conclusion_wrong in (0,1):
                    answer=x+conclusion_wrong if domain=='affine' else true+a*conclusion_wrong
                    arithmetic_rhs=a*answer if domain=='affine' else answer
                    direct=f'x = {answer}.' if domain=='affine' else f'The product is {answer}.'
                    recompute=(f'{true+b} - {b} = {arithmetic_rhs}. Dividing by {a} gives x = {answer}.'
                               if domain=='affine' else f'{a}*{x} = {answer}. The product is {answer}.')
                    follow=(f'Adding 1 gives x + 1 = {answer+1}.' if domain=='affine'
                            else f'Adding 1 to that product gives {answer+1}.')
                    for form in config['forms']:
                        wrapper={'direct':'','recompute':'','check':'Let me check the calculation. ',
                                 'repair':'Let me correct the calculation. '}[form]
                        candidate=wrapper+(direct if form=='direct' else recompute)
                        for stage in ('target','followup'):
                            transition_invalid=(int(prefix_wrong!=conclusion_wrong) if form=='direct'
                                                else conclusion_wrong) if stage=='target' else 0
                            row=dict(uid=f'{family}_p{prefix_wrong}_c{conclusion_wrong}_{form}_{stage}',
                                     family_id=family,problem_id=family,solution_id=None,
                                     step_idx=1 if stage=='target' else 2,
                                     partition='dev' if index<config['dev_per_domain'] else 'test',
                                     domain=domain,form=form,stage=stage,
                                     prefix_wrong=prefix_wrong,conclusion_wrong=conclusion_wrong,
                                     witness=dict(a=a,b=b,x=x), problem=problem,
                                     prefix=prefix if stage=='target' else prefix+'\n\n'+candidate,
                                     candidate_step=candidate if stage=='target' else follow,
                                     math_start=len(wrapper) if stage=='target' else 0,
                                     candidate_math_invalid=conclusion_wrong if stage=='target' else 0,
                                     transition_invalid=transition_invalid,label=transition_invalid,
                                     recovered=int(prefix_wrong==1 and conclusion_wrong==0 and form!='direct'))
                            rows.append(row)
    return rows
