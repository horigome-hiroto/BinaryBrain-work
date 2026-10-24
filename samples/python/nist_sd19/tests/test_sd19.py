import dataclasses
import hashlib
import importlib
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from PIL import Image
import pytest
from config import ExperimentConfig, build_arg_parser, config_from_args
from experiment_dataset import source_manifest, load_or_create_common_dataset, preprocess
from experiment_metrics import SegmentationMetrics,crop_teacher_for_common_eval,crop_prediction_for_common_eval
from experiment_lut import fpga_magazine_stage_specs,g_research_stage_specs

@pytest.fixture
def cfg(tmp_path):
    root=tmp_path/'raw'
    # Artificial native-sized images, NOT real SD19 validation data.
    for label in range(10):
        directory=root/('%02x'%(48+label))/'hsf_0';directory.mkdir(parents=True)
        for i in range(6):
            a=np.full((128,128),255,np.uint8)
            a[15+i:95,30:55+label]=0
            a[55:62,55:70+i]=0
            Image.fromarray(a).save(directory/('%d.png'%i))
    return ExperimentConfig(sd19_root=str(root),common_data_dir=str(tmp_path/'cache'),
        num_train_tiles=20,num_val_tiles=10,num_test_tiles=10)

def test_fixed_disjoint_content_split_and_cache(cfg):
    manifest=source_manifest(cfg)
    groups=[{r['pixels_sha256'] for r in manifest['splits'][s]} for s in ('train','val','test')]
    assert not (groups[0]&groups[1] or groups[0]&groups[2] or groups[1]&groups[2])
    for size in (22,28,32,44,64,128):
        c=dataclasses.replace(cfg,input_h=size,input_w=size,tile_size=size)
        assert source_manifest(c)['manifest_id']==manifest['manifest_id']
        d=load_or_create_common_dataset(c)
        for split,n in [('train',20),('val',10),('test',10)]:
            x=d['source_imgs_'+split];t=d['teacher_imgs_'+split+'_full']
            assert x.shape==(n,1,size,size) and t.shape==(n,11,size,size)
            assert t.dtype==np.uint8 and t.sum(1).max()==1
            fg=x[:,0]>=cfg.threshold
            assert np.array_equal(t[:,:10].sum(1),fg)
            bg=t[:,10][~fg]
            if split=='train': assert abs(bg.mean()-.8)<.03
            else: assert bg.min()==1 and t.sum(1).min()==1
        cached=load_or_create_common_dataset(c)
        assert np.array_equal(d['teacher_imgs_train_full'],cached['teacher_imgs_train_full'])
    assert len(list(Path(cfg.common_data_dir).glob('dataset_*/preview.png')))==6

def test_defaults_and_eval_separation():
    p=build_arg_parser('test',include_fpga_blocks=True)
    args=p.parse_args([])
    c=config_from_args(args,model_name='fpga_magazine',architecture_variant='samepadding',fpga_dw_pw_blocks=36)
    assert (c.num_train_tiles,c.num_val_tiles,c.num_test_tiles)==(4096,128,1024)
    assert (c.batch_size,c.dense_epochs,c.distill_epochs,c.cnv_epochs)==(16,128,2,4)
    assert c.frame_modulation_size==5 and c.bin_mode
    assert c.bg_keep_train==.8 and c.bg_keep_eval==1
    for size in (22,28,32,44,64,128):
        a=p.parse_args(['--size',str(size)])
        conf=config_from_args(a,model_name='x',architecture_variant='samepadding')
        assert conf.eval_hw==size
    with pytest.raises(ValueError):
        config_from_args(p.parse_args(['--size','44','--eval-hw','45']),model_name='x',architecture_variant='x')

def test_teacher_mask_and_metrics():
    t=np.zeros((1,11,44,44),np.uint8);t[:,10]=1;t[:,10,10:30,12:28]=0;t[:,3,10:30,12:28]=1
    m=SegmentationMetrics();m.calculate_numpy(t,t)
    score=m.get_scores();assert score['pixel_acc']==score['all_miou']==score['foreground_miou']==1
    ignored=t.copy();ignored[:,:,0,0]=0
    m.clear();m.calculate_numpy(t,ignored);assert m.get_scores()['valid_pixel_count']==44*44-1
    assert crop_teacher_for_common_eval(t,40).shape==(1,11,40,40)
    assert crop_prediction_for_common_eval(t,'g_research',[11,44,44],44).shape==t.shape

def test_architecture_descriptions():
    g=g_research_stage_specs();assert len(g)==31 and sum(x['spatial_3x3'] for x in g)==29
    assert [s['output_channels'] for s in g[-2:]]==[72,11]
    for blocks in (1,2,3,27,36,40):
        a=fpga_magazine_stage_specs(blocks)
        assert len(a)==blocks+2
        assert a[0]['output_channels']==32 and a[-1]['output_channels']==11
        if blocks>1: assert a[blocks-1]['output_channels']==64
        assert a[blocks]['input_channels']==(32 if blocks==1 else 64)
        for size in (22,28,32,44,64,128):
            c=ExperimentConfig(input_h=size,input_w=size,fpga_dw_pw_blocks=blocks)
            for mod in ('fpga_magazin_network_samepadding','g_research_miou_replicate_fixed'):
                assert importlib.import_module(mod).describe_without_binarybrain(c)['output_shape']==[11,size,size]

def test_source_edit_detection(cfg):
    m=source_manifest(cfg);r=m['splits']['train'][0]
    Path(m['root'],r['path']).write_bytes(b'changed')
    with pytest.raises(ValueError,match='Source modified'):
        load_or_create_common_dataset(cfg)

def test_occupancy_and_aspect(cfg):
    path=next(Path(cfg.sd19_root).rglob('*.png'))
    c=dataclasses.replace(cfg,occupancy=.7)
    x=preprocess(path,c);yy,xx=np.where(x>=c.threshold)
    assert abs(max(np.ptp(yy)+1,np.ptp(xx)+1)-round(44*.7))<=1


def test_retained_training_functions():
    import ast
    import experiment_common
    tree=ast.parse(Path(experiment_common.__file__).read_text())
    hashes=json.loads((Path(__file__).parents[1]/'provenance.json').read_text())['unchanged_function_ast_sha256']
    for node in tree.body:
        if isinstance(node,ast.FunctionDef) and node.name in hashes:
            assert hashlib.sha256(ast.dump(node,include_attributes=False).encode()).hexdigest()==hashes[node.name]


def test_original_loader_interface(cfg):
    from experiment_dataset import make_data_loaders
    c=dataclasses.replace(cfg,batch_size=4)
    data=load_or_create_common_dataset(c)
    loaders=make_data_loaders(c,data)
    x,t=next(iter(loaders['train']))
    assert tuple(x.shape)==(4,1,44,44) and tuple(t.shape)==(4,11,44,44)
    assert np.array_equal(next(iter(loaders['test']))[1].numpy(), data['teacher_imgs_test_full'][:4])

# A recording constructor, NOT a substitute BinaryBrain runtime. This executes
# the actual factories and checks their wiring/shape algebra without C++ kernels.
def test_factory_wiring_symbolically():
    from types import SimpleNamespace
    convolutions=[]
    class Layer:
        def __init__(self,shape=None,**kw): self.shape=shape
        def set_input_shape(self,shape): return self.shape or shape
        def set_name(self,name): pass
        def send_command(self,command): pass
    class Seq(Layer):
        def __init__(self,layers=None): self.layers=layers or []
        def append(self,x): self.layers.append(x)
        def set_input_shape(self,shape):
            for layer in self.layers: shape=layer.set_input_shape(shape)
            return shape
    class Switch(Layer):
        def __init__(self,models,**kw): self.models=models
        def set_input_shape(self,shape):
            dense=self.models['dense'].set_input_shape(shape)
            lut=self.models['lut'].set_input_shape(shape)
            assert dense==lut
            return dense
    class Conv(Layer):
        def __init__(self,sub_layer,filter_size=(1,1),**kw):
            self.sub=sub_layer;self.filter=filter_size;self.kw=kw;convolutions.append(self)
        def set_input_shape(self,shape):
            c,h,w=shape
            out=self.sub.set_input_shape([c*self.filter[0]*self.filter[1],1,1])[0]
            if self.filter==(3,3):
                assert self.kw['padding']=='same' and self.kw['border_mode']=='replicate'
            return [out,h,w]
    bb=SimpleNamespace(DType=SimpleNamespace(BIT='bit',FP32='float'),Sequential=Seq,
        Convolution2d=Conv,Switcher=Switch,DenseAffine=Layer,BatchNormalization=Layer,
        Binarize=Layer,DifferentiableLut=Layer,RealToBinary=Layer,BinaryToReal=Layer)
    for size in (22,28,32,44,64,128):
        for name,blocks in [('fpga_magazin_network_samepadding',36),('g_research_miou_replicate_fixed',36),('fpga_magazin_network_samepadding',1)]:
            cfg=ExperimentConfig(input_h=size,input_w=size,fpga_dw_pw_blocks=blocks)
            net,shape,_,layers,_=importlib.import_module(name).create_network(bb,cfg)
            assert shape==[11,size,size]
            assert layers==(blocks+2 if name.startswith('fpga') else 31)
