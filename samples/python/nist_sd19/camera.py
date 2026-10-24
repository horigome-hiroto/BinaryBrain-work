"""640x132 synthetic camera visualization using square-patch inference.

Uses the training scale without rescaling the canvas. Patches have replicate
padding on the right/bottom. No halo/blending: seams lack context and are NOT
whole-frame inference. Camera results must be compared with this limitation.
"""
import argparse
import importlib
import json
from pathlib import Path
import numpy as np
from PIL import Image
from config import ExperimentConfig
from experiment_dataset import load_or_create_common_dataset
from experiment_common import require_binarybrain

def main():
    p=argparse.ArgumentParser();p.add_argument('--run-dir',required=True)
    p.add_argument('--checkpoint',choices=('best_dense','best_lut'),default='best_lut')
    p.add_argument('--output',default='camera_result');p.add_argument('--seed',type=int,default=0)
    args=p.parse_args();run=Path(args.run_dir)
    saved=json.loads((run/'config.json').read_text())
    cfg=ExperimentConfig(**{k:v for k,v in saved.items() if k in ExperimentConfig.__dataclass_fields__})
    bb=require_binarybrain()
    module=importlib.import_module('fpga_magazin_network_samepadding' if cfg.model_name=='fpga_magazine' else 'g_research_miou_replicate_fixed')
    net,_,_,_,_=module.create_network(bb,cfg)
    if not bb.load_networks(str(run/'checkpoints'),net,name=args.checkpoint,verbose=False):
        raise FileNotFoundError(args.checkpoint)
    net.send_command('switch_model '+('dense' if args.checkpoint=='best_dense' else 'lut'))
    data=load_or_create_common_dataset(cfg)
    canvas=np.zeros((132,640),np.float32)
    truth=np.full((132,640),10,np.uint8)
    rng=np.random.default_rng(args.seed);s=cfg.input_h
    # Each patch includes the same normalized occupancy as its square training image.
    for y in range(0,132-s+1,s):
        for x in range(0,640-s+1,s):
            i=int(rng.integers(len(data['source_imgs_test'])))
            canvas[y:y+s,x:x+s]=data['source_imgs_test'][i,0]
            truth[y:y+s,x:x+s]=data['teacher_imgs_test_full'][i].argmax(0)
    pred=np.zeros((132,640),np.uint8)
    for y in range(0,132,s):
        for x in range(0,640,s):
            h,w=min(s,132-y),min(s,640-x)
            patch=np.pad(canvas[y:y+h,x:x+w],((0,s-h),(0,s-w)),mode='edge')
            result=net.forward(bb.FrameBuffer.from_numpy(patch[None,None]),train=False).numpy()
            pred[y:y+h,x:x+w]=result[0].argmax(0)[:h,:w]
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    Image.fromarray(np.uint8(canvas*255)).save(out/'input.png')
    np.save(out/'prediction.npy',pred);np.save(out/'teacher.npy',truth)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(3,1,figsize=(16,9))
    ax[0].imshow(canvas,cmap='gray');ax[1].imshow(truth,cmap='tab20',vmin=0,vmax=10);ax[2].imshow(pred,cmap='tab20',vmin=0,vmax=10)
    fig.tight_layout();fig.savefig(out/'camera.png');plt.close(fig)
    (out/'metadata.json').write_text(json.dumps(dict(run_dir=str(run),checkpoint=args.checkpoint,seed=args.seed,
        inference='nonoverlapping square patches; replicate edge; context seams',canvas_hw=[132,640],patch_size=s),indent=2))

if __name__=='__main__':main()
