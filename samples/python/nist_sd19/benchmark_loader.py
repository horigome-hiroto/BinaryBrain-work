"""Time data loading only; no claim about network/GPU performance.

Compare retained in-memory baseline interface with .npy mmap using the same
frozen data, batches, seed, uint8 one-hot and torch DataLoader implementation.
"""
import argparse
import dataclasses
import json
from pathlib import Path
import statistics
import time
import numpy as np
from config import ExperimentConfig
from experiment_dataset import load_or_create_common_dataset,make_data_loaders

def benchmark(config, repeats=3):
    data=load_or_create_common_dataset(config)
    baseline=dict(data)
    for k,v in data.items():
        if isinstance(v,np.ndarray): baseline[k]=np.array(v,copy=True)
    results={}
    for mode,d in [('baseline_in_memory',baseline),('sd19_mmap',data)]:
        timings=[]
        for _ in range(repeats+1):
            loaders=make_data_loaders(config,d)
            start=time.perf_counter();count=0
            for x,t in loaders['train']:
                # Same conversion used by the original training loop.
                np.asarray(x,dtype=np.float32);np.asarray(t).astype(np.float32)
                count+=len(x)
            timings.append(time.perf_counter()-start)
        results[mode]=dict(images=count,seconds=timings[1:],median_seconds=statistics.median(timings[1:]))
    return dict(scope='loader and original numpy conversion only, no network, GPU, or forward/backward',
                shape=[config.input_h,config.input_w],results=results)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run-dir',required=True);p.add_argument('--output',default='loader_benchmark.json')
    a=p.parse_args();saved=json.loads((Path(a.run_dir)/'config.json').read_text())
    c=ExperimentConfig(**{k:v for k,v in saved.items() if k in ExperimentConfig.__dataclass_fields__})
    Path(a.output).write_text(json.dumps(benchmark(c),indent=2))
