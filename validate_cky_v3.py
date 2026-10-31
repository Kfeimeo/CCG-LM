"""V3 matrix and complete LM gradient validation; retains all strict failures."""
from __future__ import annotations
import argparse,itertools,json,traceback
from pathlib import Path
from v3_study import forest_pair,model_pair
from utils import atomic_json


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output-dir',default='reports/cky_v3')
    p.add_argument('--implementation',default='lazy_tensorized');p.add_argument('--cpu-only',action='store_true');p.add_argument('--matrix-only',action='store_true');p.add_argument('--gradient-only',action='store_true');p.add_argument('--resume',action='store_true')
    a=p.parse_args();root=Path(a.output_dir);root.mkdir(parents=True,exist_ok=True)
    devices=[('cpu','fp32')]+([] if a.cpu_only else [('cuda','fp32'),('cuda','bf16')])
    correctness=dict(matrix=[],edge_cases=[],implementation=a.implementation,status='running')
    if a.resume and (root/'correctness.json').exists():correctness=json.loads((root/'correctness.json').read_text(encoding='utf-8'))
    done={(x['batch'],x['words'],x['k'],x['r'],x['vector'],x['device'],x['precision'],x['policy']) for x in correctness['matrix']}
    try:
        if not a.gradient_only:
            for device,precision in devices:
                for b,w,k,r,v,policy in itertools.product([1,2,4],[1,4,8,16,24],[1,2,4],[4,12],[False,True],['normal','explore']):
                    key=(b,w,k,r,v,device,precision,policy)
                    if key in done:continue
                    correctness['matrix'].append(forest_pair(b,w,k,r,v,device,precision,policy,a.implementation))
                    if len(correctness['matrix'])%60==0:
                        print('matrix',len(correctness['matrix']),device,precision,flush=True);atomic_json(correctness,root/'correctness.json')
                for policy,v in itertools.product(['ties','near_ties','prune','empty','all_inf','partial_inf'],[False,True]):
                    correctness['edge_cases'].append(forest_pair(2,8,4,12,v,device,precision,policy,a.implementation))
            correctness['status']='passed';atomic_json(correctness,root/'correctness.json')
        if not a.matrix_only:
            gradients=dict(cases=[],fp32_tolerance={'atol':3e-6,'rtol':.0003},bf16_tolerance={'atol':.00015,'rtol':.04})
            for device,precision in devices:
                for mode in ('forest_scalar4','forest_vector4'):
                    z=model_pair(mode,device,precision,implementation=a.implementation);gradients['cases'].append(z)
                    atomic_json(gradients,root/'gradient_comparison.json')
                    if precision=='fp32':assert z['gradient']['strict_failed_parameter_count']==0,z['gradient']['strict_failed_parameters']
            if not a.cpu_only:
                for precision,batch in itertools.product(['fp32','bf16'],[1,4]):
                    z=model_pair(device='cuda',precision=precision,main=True,batch=batch,implementation=a.implementation)
                    gradients['cases'].append(z);atomic_json(gradients,root/'gradient_comparison.json')
                    print('main',precision,batch,'strict failures',z['gradient']['strict_failed_parameter_count'],flush=True)
                    if precision=='fp32':assert z['gradient']['strict_failed_parameter_count']==0,z['gradient']['strict_failed_parameters']
            gradients['status']='complete; BF16 strict mismatches preserved';atomic_json(gradients,root/'gradient_comparison.json')
    except Exception as exc:
        atomic_json(dict(error=repr(exc),traceback=traceback.format_exc(),implementation=a.implementation,matrix_completed=len(correctness['matrix'])),root/'validation_failure.json');raise
    print('V3 validation completed',flush=True)

if __name__=='__main__':main()
