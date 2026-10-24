"""SD19 by_class PNG loader; immutable source split shared by all resolutions.

Uses the original uint8 one-hot teachers and DataLoader interface. One-hot
creation, hashing and resizing happen once, never in the training batch loop.
"""
import hashlib
import json
import os
from pathlib import Path
import time
import numpy as np
from PIL import Image

SCHEMA = 1

def _digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()[:20]


def _atomic_json(path, value):
    path = Path(path)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2, sort_keys=True))
    temp.replace(path)


def source_manifest(config):
    """Freeze selected original images, independent of size/network/preprocessing.

    Content duplicates are removed across ALL classes and splits. PNG 30..39
    are hex ASCII codes for digits. Prefer hsf_* directories to avoid the
    redundant train_* aggregate in the official by_class distribution.
    """
    root = Path(config.sd19_root).expanduser().resolve()
    if not config.sd19_root:
        raise ValueError('sd19_root cannot be empty')
    if not (root/'30').is_dir() and not (root/'by_class'/'30').is_dir():
        download_sd19(root)
    if (root/'by_class').is_dir():
        root = root/'by_class'
    counts = [config.num_train_tiles, config.num_val_tiles, config.num_test_tiles]
    key = _digest(dict(root=str(root), seed=config.dataset_seed, counts=counts, schema=SCHEMA))
    base = Path(config.common_data_dir)
    base.mkdir(parents=True, exist_ok=True)
    target = base/('sources_'+key+'.json')
    if target.exists():
        return json.loads(target.read_text())
    candidates = []
    for label in range(10):
        directory = root/('%02x' % (48+label))
        if not directory.is_dir():
            raise FileNotFoundError('Missing SD19 digit directory: '+str(directory))
        groups = sorted(directory.glob('hsf_*'))
        files = sorted(f for group in (groups or [directory]) for f in group.rglob('*')
                       if f.suffix.lower() == '.png')
        candidates.extend((label,path) for path in files)
    rng = np.random.default_rng(config.dataset_seed)
    order = rng.permutation(len(candidates))
    chosen, seen = [], {}
    # Read/decode only as many originals as needed, not the entire archive.
    # Still reject content duplicates across every selected split.
    for index in order:
        label,path = candidates[index]
        raw_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        with Image.open(path) as im:
            if im.size != (128,128):
                raise ValueError('Expected native 128x128 SD19 image: '+str(path))
            pixel_hash = hashlib.sha256(im.convert('L').tobytes()).hexdigest()
        if pixel_hash in seen:
            if seen[pixel_hash] != label:
                raise ValueError('Conflicting selected labels for identical pixels: '+str(path))
            continue
        seen[pixel_hash] = label
        chosen.append(dict(path=str(path.relative_to(root)),label=label,
                           sha256=raw_hash,pixels_sha256=pixel_hash))
        if len(chosen) == sum(counts): break
    if len(chosen) < sum(counts):
        raise ValueError('Not enough unique SD19 sources: %d needed, %d available' % (sum(counts),len(chosen)))
    a,b,c = counts
    result = dict(schema=SCHEMA, root=str(root), dataset_seed=config.dataset_seed,
                  selection='sorted candidates, seeded permutation, selected pixel deduplication; source-level split (not writer-disjoint)',
                  splits=dict(train=chosen[:a], val=chosen[a:a+b], test=chosen[a+b:a+b+c]))
    result['manifest_id'] = _digest(result)
    _atomic_json(target,result)
    return result


def preprocess(path, config):
    with Image.open(path) as im:
        image = np.asarray(im.convert('L'), dtype=np.float32)/255
    if config.polarity == 'dark':
        image = 1-image
    fg = image >= config.threshold
    if not fg.any():
        raise ValueError('No foreground at configured threshold: '+str(path))
    yy,xx = np.where(fg)
    cropped = image[yy.min():yy.max()+1,xx.min():xx.max()+1]
    edge = max(1, round(config.occupancy * min(config.input_h,config.input_w)))
    scale = edge/max(cropped.shape)
    h,w = [max(1,round(v*scale)) for v in cropped.shape]
    method = getattr(Image.Resampling, config.interpolation.upper())
    resized = np.asarray(Image.fromarray(cropped).resize((w,h),method),dtype=np.float32)
    canvas = np.zeros((config.input_h,config.input_w),dtype=np.float32)
    y,x = (config.input_h-h)//2,(config.input_w-w)//2
    canvas[y:y+h,x:x+w] = np.clip(resized,0,1)
    return canvas


def _settings(config, manifest):
    return dict(schema=SCHEMA, manifest_id=manifest['manifest_id'], hw=[config.input_h,config.input_w],
                occupancy=config.occupancy, threshold=config.threshold, polarity=config.polarity,
                interpolation=config.interpolation, bg_keep_train=config.bg_keep_train,
                bg_keep_eval=config.bg_keep_eval, dataset_seed=config.dataset_seed)


def common_dataset_path(config):
    manifest = source_manifest(config)
    return str(Path(config.common_data_dir)/('dataset_'+_digest(_settings(config,manifest))))


def load_or_create_common_dataset(config, *, force=False, dry_run=False):
    if dry_run:
        return build_synthetic_common_dataset(config)
    started = time.perf_counter()
    manifest = source_manifest(config)
    settings = _settings(config,manifest)
    target = Path(common_dataset_path(config))
    if force and target.exists():
        # Keep previous generated datasets and results; do not delete them.
        target.rename(target.with_name(target.name+'_archive_'+str(time.time_ns())))
    target.mkdir(parents=True,exist_ok=True)
    if not (target/'metadata.json').exists():
        for si,split in enumerate(('train','val','test')):
            records = manifest['splits'][split]
            n,h,w = len(records),config.input_h,config.input_w
            src = np.lib.format.open_memmap(target/(split+'_source.npy'),mode='w+',dtype=np.float32,shape=(n,1,h,w))
            teach = np.lib.format.open_memmap(target/(split+'_teacher.npy'),mode='w+',dtype=np.uint8,shape=(n,11,h,w))
            keep = config.bg_keep_train if split=='train' else config.bg_keep_eval
            for i,record in enumerate(records):
                path = Path(manifest['root'])/record['path']
                if hashlib.sha256(path.read_bytes()).hexdigest() != record['sha256']:
                    raise ValueError('Source modified since split creation: '+str(path))
                image = preprocess(path,config)
                fg = image >= config.threshold
                if not fg.any():
                    raise ValueError('Digit disappeared after resize: '+str(path))
                src[i,0] = image
                teach[i] = 0
                teach[i,record['label']] = fg
                # Independent stable RNG per original and split. Same source split for all sizes.
                seed = int(record['pixels_sha256'][:16],16) ^ int(config.dataset_seed) ^ si
                rng = np.random.default_rng(seed)
                bg = ~fg
                if keep < 1:
                    bg = bg & (rng.random((h,w)) < keep)
                teach[i,10] = bg
            src.flush(); teach.flush()
        _atomic_json(target/'sources.json',manifest)
        _atomic_json(target/'metadata.json',dict(settings, generation_seconds=time.perf_counter()-started))
    result = dict(split_info=manifest, dataset_seed=config.dataset_seed, teacher_dtype='uint8')
    for split in ('train','val','test'):
        result['source_imgs_'+split] = np.load(target/(split+'_source.npy'),mmap_mode='c')
        result['teacher_imgs_'+split+'_full'] = np.load(target/(split+'_teacher.npy'),mmap_mode='c')
    if not (target/'preview.png').exists():
        visualize(result,target/'preview.png')
    return result


def visualize(data,path):
    from experiment_common import _get_pyplot
    plt = _get_pyplot()
    if plt is None:
        raise RuntimeError('Matplotlib is required to visualize generated data')
    n = min(8,len(data['source_imgs_train']))
    fig,axes = plt.subplots(3,n,figsize=(2*n,6),squeeze=False)
    for i in range(n):
        image = data['source_imgs_train'][i,0]
        t = data['teacher_imgs_train_full'][i]
        mask = t.argmax(0).astype(np.int16)
        mask[~t.any(0)] = -1
        axes[0,i].imshow(image,cmap='gray',vmin=0,vmax=1)
        axes[1,i].imshow(mask,cmap='tab20',vmin=-1,vmax=10,interpolation='nearest')
        axes[2,i].imshow(t.any(0),cmap='gray',vmin=0,vmax=1)
        for row in range(3): axes[row,i].axis('off')
    fig.suptitle('Input / teacher (-1 ignored, 10 background) / supervised pixels')
    fig.tight_layout(); fig.savefig(path); plt.close(fig)

def build_synthetic_common_dataset(config, *, num_train=2, num_val=2, num_test=2):
    rng = np.random.default_rng(int(config.dataset_seed))

    def make(n, bg_keep):
        source = rng.random((n, 1, config.input_h, config.input_w), dtype=np.float32)
        teacher = np.zeros((n, config.num_classes, config.input_h, config.input_w), dtype=np.uint8)
        for i in range(n):
            cls = i % 10
            teacher[i, cls, config.input_h//4:3*config.input_h//4, config.input_w//4:3*config.input_w//4] = (source[i, 0, config.input_h//4:3*config.input_h//4, config.input_w//4:3*config.input_w//4] > 0.5).astype(np.uint8)
            bg = 1 - np.any(teacher[i, :10] > 0, axis=0).astype(np.uint8)
            if bg_keep < 1.0:
                bg = bg * (rng.random(bg.shape) < bg_keep).astype(np.uint8)
            teacher[i, 10] = bg
        return source, teacher

    source_train, teacher_train = make(num_train, config.bg_keep_train)
    source_val, teacher_val = make(num_val, config.bg_keep_eval)
    source_test, teacher_test = make(num_test, config.bg_keep_eval)
    return {
        "source_imgs_train": source_train,
        "teacher_imgs_train_full": teacher_train,
        "source_imgs_val": source_val,
        "teacher_imgs_val_full": teacher_val,
        "source_imgs_test": source_test,
        "teacher_imgs_test_full": teacher_test,
        "mnist_indices": {
            "train_pool": [],
            "val_pool": [],
            "test_pool": [],
            "train_tile_indices": [],
            "val_tile_indices": [],
            "test_tile_indices": [],
        },
        "split_info": {"synthetic": True},
        "dataset_seed": int(config.dataset_seed),
        "bg_keep_train": float(config.bg_keep_train),
        "bg_keep_eval": float(config.bg_keep_eval),
        "teacher_dtype": "uint8",
    }



def make_data_loaders(config, common_data):
    import torch

    class FullTeacherDataset(torch.utils.data.Dataset):
        def __init__(self, source_imgs, teacher_imgs_full):
            self.source_imgs = source_imgs
            self.teacher_imgs_full = teacher_imgs_full

        def __len__(self):
            return len(self.source_imgs)

        def __getitem__(self, index):
            return self.source_imgs[index], self.teacher_imgs_full[index]

    generator = torch.Generator()
    generator.manual_seed(int(config.run_seed))
    train_dataset = FullTeacherDataset(common_data["source_imgs_train"], common_data["teacher_imgs_train_full"])
    val_dataset = FullTeacherDataset(common_data["source_imgs_val"], common_data["teacher_imgs_val_full"])
    test_dataset = FullTeacherDataset(common_data["source_imgs_test"], common_data["teacher_imgs_test_full"])
    return {
        "train": torch.utils.data.DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            generator=generator,
            num_workers=0,
        ),
        "train_eval": torch.utils.data.DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=0,
        ),
        "val": torch.utils.data.DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=0,
        ),
        "test": torch.utils.data.DataLoader(
            test_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=0,
        ),
    }


# Official download support, integrated so the package needs only seven files.
"""Fetch official NIST SD19 archive with TLS verification and ZIP CRC checks."""
from pathlib import Path, PurePosixPath
import hashlib
import json
import urllib.request
import zipfile

SD19_DOWNLOAD_URL = 'https://s3.amazonaws.com/nist-srd/SD19/by_class.zip'

def download_sd19(root):
    root = Path(root).expanduser().resolve()
    if all((root/'by_class'/('%02x' % (48+i))).is_dir() for i in range(10)):
        return
    root.mkdir(parents=True,exist_ok=True)
    archive = root/'by_class.zip'
    if not archive.exists():
        partial = root/'by_class.zip.partial'
        print('Downloading official SD19 archive (large download): '+SD19_DOWNLOAD_URL,flush=True)
        try:
            with urllib.request.urlopen(SD19_DOWNLOAD_URL,timeout=60) as response, partial.open('wb') as out:
                expected = response.headers.get('Content-Length')
                size = 0
                while True:
                    chunk = response.read(1024*1024)
                    if not chunk: break
                    out.write(chunk); size += len(chunk)
                if expected is not None and size != int(expected):
                    raise IOError('Incomplete download')
            partial.replace(archive)
        except Exception as exc:
            raise RuntimeError('SD19 download failed. Allow s3.amazonaws.com in environment network settings, or manually extract by_class and pass --sd19-root. '+str(exc)) from exc
    with zipfile.ZipFile(archive) as z:
        # Extract only digits. ZipFile.open verifies CRC while fully reading.
        for member in z.infolist():
            path = PurePosixPath(member.filename)
            if path.is_absolute() or '..' in path.parts:
                raise ValueError('Unsafe ZIP path')
            if len(path.parts)<3 or path.parts[0]!='by_class' or path.parts[1] not in {('%02x'%(48+i)) for i in range(10)}:
                continue
            target = root.joinpath(*path.parts)
            if member.is_dir(): target.mkdir(parents=True,exist_ok=True); continue
            target.parent.mkdir(parents=True,exist_ok=True)
            if target.exists():
                continue  # Preserve manually supplied/extracted files.
            partial = target.with_suffix(target.suffix+'.partial')
            with z.open(member) as src, partial.open('wb') as out:
                while True:
                    chunk = src.read(1024*1024)
                    if not chunk: break
                    out.write(chunk)
            partial.replace(target)
    digest = hashlib.sha256()
    with archive.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    checksum = digest.hexdigest()
    (root/'download.json').write_text(json.dumps(dict(url=SD19_DOWNLOAD_URL,sha256=checksum,
        note='Local provenance hash; not a publisher-provided checksum. TLS and ZIP CRC verified.'),indent=2))
