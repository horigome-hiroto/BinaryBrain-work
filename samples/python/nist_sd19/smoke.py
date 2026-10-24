"""Small real BinaryBrain validation; never starts the full experiment.

Requires an already generated SD19 dataset. Reports unavailable dependencies as
errors, rather than substituting a different network or pretending dry-run works.
"""
import argparse
import importlib
import json
from pathlib import Path
import time
import numpy as np
from config import build_arg_parser, config_from_args
from experiment_dataset import load_or_create_common_dataset, make_data_loaders
from experiment_common import require_binarybrain, set_global_seed, evaluate_segmentation, _train_epoch_whole

MODULES = dict(fpga_magazine='fpga_magazin_network_samepadding',g_research='g_research_miou_replicate_fixed')

def main():
    parser=build_arg_parser('Short SD19 runtime check',include_fpga_blocks=True)
    parser.add_argument('--model',choices=MODULES,default='fpga_magazine')
    parser.add_argument('--report',default='smoke_result.json')
    args=parser.parse_args()
    config=config_from_args(args,model_name=args.model,architecture_variant='samepadding_smoke',fpga_dw_pw_blocks=args.fpga_dw_pw_blocks)
    bb=require_binarybrain();set_global_seed(config.run_seed)
    module=importlib.import_module(MODULES[args.model])
    started=time.perf_counter()
    net,shape,dtype,layers,name=module.create_network(bb,config)
    assert tuple(shape)==(11,args.size,args.size)
    report=dict(model=args.model,size=args.size,layers=layers,build_seconds=time.perf_counter()-started, modes={})
    # Check both routes before training, not just the DenseAffine switch route.
    x=np.zeros((1,1,args.size,args.size),np.float32)
    for mode in ('dense','lut'):
        net.send_command('switch_model '+mode)
        y=net.forward(bb.FrameBuffer.from_numpy(x),train=True)
        assert y.numpy().shape==(1,11,args.size,args.size)
        net.backward(bb.FrameBuffer.from_numpy(np.ones((1,11,args.size,args.size),np.float32)))
        report['modes'][mode]='forward/backward passed'
    data=load_or_create_common_dataset(config,force=args.force_dataset)
    # No additional sources are selected: take a tiny view of the frozen split.
    tiny=dict(data)
    for key in data:
        if key.startswith(('source_imgs_','teacher_imgs_')):
            tiny[key]=data[key][:min(2,config.batch_size)]
    loaders=make_data_loaders(config,tiny)
    net.send_command('switch_model dense');net.send_command('parameter_lock false')
    optimizer=bb.OptimizerAdam(learning_rate=config.learning_rate)
    optimizer.set_variables(net.get_parameters(),net.get_gradients())
    started=time.perf_counter()
    report['train_loss']=_train_epoch_whole(bb,net,loaders['train'],output_shape=shape,model_kind=args.model,optimizer=optimizer,desc='smoke only',evaluation_hw=(config.eval_hw,)*2)
    report['train_seconds']=time.perf_counter()-started
    result=evaluate_segmentation(bb,net,loaders['val'],output_shape=shape,model_kind=args.model,title='smoke',evaluation_hw=(config.eval_hw,)*2)
    from experiment_metrics import scores_to_jsonable
    report['validation']=dict(loss=result['loss'],scores=scores_to_jsonable(result['scores']))
    Path(args.report).write_text(json.dumps(report,indent=2))

if __name__=='__main__': main()
