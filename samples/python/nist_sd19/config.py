#!/usr/bin/env python
# coding: utf-8

"""Shared configuration for the MNIST LUT-Network comparison experiments."""

from dataclasses import asdict, dataclass
import argparse
import re


DATASET_SEED = 0
RUN_SEEDS = (0, 1, 2)

INPUT_H = 44
INPUT_W = 44
ROWS = 1
COLS = 1
TILE_SIZE = 44

NUM_CLASSES = 11
BACKGROUND_CLASS = 10
CLASS_NAMES = [str(i) for i in range(10)] + ["background"]

BG_KEEP_TRAIN = 0.8
BG_KEEP_EVAL = 1.0

TRAIN_MNIST_COUNT = 55000
VAL_MNIST_COUNT = 5000

NUM_TRAIN_TILES = 4096
NUM_VAL_TILES = 128
NUM_TEST_TILES = 1024

BATCH_SIZE = 16
DENSE_EPOCHS = 128
DISTILL_EPOCHS = 2
CNV_EPOCHS = 4
LEARNING_RATE = 0.001
LAYER_LEARNING_RATE = 0.001
DISTILL_LEARNING_RATE = 0.001
PLOT_EVERY = 1
DENSE_CLASS_IOU_BAR_EVERY = 10

# mIoU / Class IoU are calculated on this central square crop.  The model
# itself still produces its native output (26x26 for the current g_research
# network); this setting changes only the region used for evaluation.
EVAL_HW = 44

FRAME_MODULATION_SIZE = 5
BIN_MODE = True

FPGA_MAGAZINE_DW_PW_BLOCKS = 36
FPGA_MAGAZINE_FINAL_POINTWISE_LAYERS = 2
G_RESEARCH_SPATIAL_LAYERS = 29
G_RESEARCH_FINAL_POINTWISE_LAYERS = 2

COMMON_DATA_DIR = "./data/sd19/common"
RUN_ROOT = "./data/sd19/runs"


def slugify(text):
    text = str(text).strip()
    text = re.sub(r"[^0-9A-Za-z_.-]+", "_", text)
    text = re.sub(r"_+", "_", text)
    return text.strip("_") or "run"


@dataclass
class ExperimentConfig:
    sd19_root: str = "./data/sd19/raw"
    occupancy: float = 0.8
    threshold: float = 0.5
    polarity: str = "dark"
    interpolation: str = "bilinear"
    # Longest side of cropped digit occupies occupancy * image size.
    model_name: str = "fpga_magazine"
    architecture_variant: str = "samepadding_sd19"
    run_id: str = "default"
    run_seed: int = 0
    dataset_seed: int = DATASET_SEED

    input_h: int = INPUT_H
    input_w: int = INPUT_W
    rows: int = ROWS
    cols: int = COLS
    tile_size: int = TILE_SIZE

    num_classes: int = NUM_CLASSES
    background_class: int = BACKGROUND_CLASS
    bg_keep_train: float = BG_KEEP_TRAIN
    bg_keep_eval: float = BG_KEEP_EVAL

    train_mnist_count: int = TRAIN_MNIST_COUNT
    val_mnist_count: int = VAL_MNIST_COUNT
    num_train_tiles: int = NUM_TRAIN_TILES
    num_val_tiles: int = NUM_VAL_TILES
    num_test_tiles: int = NUM_TEST_TILES

    batch_size: int = BATCH_SIZE
    dense_epochs: int = DENSE_EPOCHS
    distill_epochs: int = DISTILL_EPOCHS
    cnv_epochs: int = CNV_EPOCHS
    learning_rate: float = LEARNING_RATE
    layer_learning_rate: float = LAYER_LEARNING_RATE
    distill_learning_rate: float = DISTILL_LEARNING_RATE
    plot_every: int = PLOT_EVERY
    dense_class_iou_bar_every: int = DENSE_CLASS_IOU_BAR_EVERY
    plot_enabled: bool = True
    eval_hw: int = EVAL_HW

    frame_modulation_size: int = FRAME_MODULATION_SIZE
    bin_mode: bool = BIN_MODE

    fpga_dw_pw_blocks: int = FPGA_MAGAZINE_DW_PW_BLOCKS
    fpga_final_pointwise_layers: int = FPGA_MAGAZINE_FINAL_POINTWISE_LAYERS
    g_research_spatial_layers: int = G_RESEARCH_SPATIAL_LAYERS
    g_research_final_pointwise_layers: int = G_RESEARCH_FINAL_POINTWISE_LAYERS

    common_data_dir: str = COMMON_DATA_DIR
    run_root: str = RUN_ROOT
    rtl_sim_path: str = "../../verilog/mnist/tb_mnist_semantic_segmentationSimple"

    def to_dict(self):
        return asdict(self)


def default_run_id(model_name, architecture_variant, run_seed):
    return slugify("%s_%s_seed%d" % (model_name, architecture_variant, run_seed))


def add_common_arguments(parser, *, include_fpga_blocks=False):
    parser.add_argument("--sd19-root", default="./data/sd19/raw")
    parser.add_argument("--size", type=int, choices=(22,28,32,44,64,128), default=INPUT_H)
    parser.add_argument("--occupancy", type=float, default=0.8)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--polarity", choices=("dark", "light"), default="dark")
    parser.add_argument("--interpolation", choices=("nearest", "bilinear", "bicubic"), default="bilinear")
    parser.add_argument("--bg-keep-train", type=float, default=BG_KEEP_TRAIN)
    parser.add_argument("--bg-keep-eval", type=float, default=BG_KEEP_EVAL)
    parser.add_argument("--frame-modulation-size", type=int, default=FRAME_MODULATION_SIZE)
    parser.add_argument("--bin-mode", action=argparse.BooleanOptionalAction, default=BIN_MODE)
    parser.add_argument("--seed", "--run-seed", dest="run_seed", type=int, default=0)
    parser.add_argument("--dataset-seed", type=int, default=DATASET_SEED)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--dense-epochs", type=int, default=DENSE_EPOCHS)
    parser.add_argument("--distill-epochs", type=int, default=DISTILL_EPOCHS)
    parser.add_argument("--cnv-epochs", type=int, default=CNV_EPOCHS)
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
    parser.add_argument("--layer-learning-rate", type=float, default=LAYER_LEARNING_RATE)
    parser.add_argument("--distill-learning-rate", type=float, default=DISTILL_LEARNING_RATE)
    parser.add_argument("--num-train-tiles", type=int, default=NUM_TRAIN_TILES)
    parser.add_argument("--num-val-tiles", type=int, default=NUM_VAL_TILES)
    parser.add_argument("--num-test-tiles", type=int, default=NUM_TEST_TILES)
    parser.add_argument("--common-data-dir", default=COMMON_DATA_DIR)
    parser.add_argument("--run-root", default=RUN_ROOT)
    parser.add_argument("--force-dataset", action="store_true")
    parser.add_argument("--prepare-dataset-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-training", action="store_true")
    parser.add_argument("--no-rtl", action="store_true")
    parser.add_argument("--copy-rtl-sim", action="store_true")
    parser.add_argument("--allow-existing-run-dir", action="store_true")
    parser.add_argument("--plot-every", type=int, default=PLOT_EVERY)
    parser.add_argument("--dense-class-iou-bar-every", type=int, default=DENSE_CLASS_IOU_BAR_EVERY)
    parser.add_argument(
        "--eval-hw",
        type=int,

        default=None,
        help=(
            "Central square region used for mIoU/Class-IoU evaluation. "
            "This changes evaluation only; the model output and training loss stay unchanged."
        ),
    )
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--plot", action="store_true")
    if include_fpga_blocks:
        parser.add_argument(
        "--fpga-dw-pw-blocks",
        type=int,

        default=FPGA_MAGAZINE_DW_PW_BLOCKS,
    )


def config_from_args(args, *, model_name, architecture_variant, fpga_dw_pw_blocks=None):
    if not 0 < args.occupancy <= 1 or not 0 < args.threshold < 1:
        raise ValueError("occupancy must be (0,1], threshold (0,1)")
    if any(not 0 <= v <= 1 for v in (args.bg_keep_train, args.bg_keep_eval)):
        raise ValueError("background keep probabilities must be [0,1]")
    if any(v <= 0 for v in (args.batch_size, args.frame_modulation_size, args.num_train_tiles, args.num_val_tiles, args.num_test_tiles)):
        raise ValueError("batch size, frame modulation, and split counts must be positive")
    if any(v < 0 for v in (args.dense_epochs, args.distill_epochs, args.cnv_epochs)):
        raise ValueError("epoch counts cannot be negative")
    if getattr(args, "fpga_dw_pw_blocks", 1) <= 0:
        raise ValueError("DW/PW block count must be positive")
    eval_hw = args.size if args.eval_hw is None else args.eval_hw
    if not 0 < eval_hw <= args.size or (args.size-eval_hw) % 2:
        raise ValueError("evaluation crop must fit and have an even size difference")
    if args.dataset_seed < 0 or args.run_seed < 0:
        raise ValueError("seeds must be nonnegative")
    run_id = args.run_id
    if run_id is None:
        run_id = default_run_id(model_name, architecture_variant, args.run_seed) + "_%dx%d_b%d_eval%d" % (args.size,args.size,fpga_dw_pw_blocks or 0,eval_hw)
    if fpga_dw_pw_blocks is None:
        fpga_dw_pw_blocks = FPGA_MAGAZINE_DW_PW_BLOCKS
    config = ExperimentConfig(
        sd19_root=args.sd19_root, occupancy=args.occupancy, threshold=args.threshold,
        polarity=args.polarity, interpolation=args.interpolation,
        input_h=args.size, input_w=args.size, tile_size=args.size,
        bg_keep_train=args.bg_keep_train, bg_keep_eval=args.bg_keep_eval,
        frame_modulation_size=args.frame_modulation_size, bin_mode=args.bin_mode,
        model_name=model_name,
        architecture_variant=architecture_variant,
        run_id=slugify(run_id),
        run_seed=args.run_seed,
        dataset_seed=args.dataset_seed,
        batch_size=args.batch_size,
        dense_epochs=args.dense_epochs,
        distill_epochs=args.distill_epochs,
        cnv_epochs=args.cnv_epochs,
        learning_rate=args.learning_rate,
        layer_learning_rate=args.layer_learning_rate,
        distill_learning_rate=args.distill_learning_rate,
        plot_every=args.plot_every,
        dense_class_iou_bar_every=args.dense_class_iou_bar_every,
        plot_enabled=not args.no_plot,
        eval_hw=eval_hw,
        num_train_tiles=args.num_train_tiles,
        num_val_tiles=args.num_val_tiles,
        num_test_tiles=args.num_test_tiles,
        common_data_dir=args.common_data_dir,
        run_root=args.run_root,
        fpga_dw_pw_blocks=fpga_dw_pw_blocks,
    )

    if args.run_id is None:
        import hashlib, json
        identity = config.to_dict()
        identity.pop("run_id")
        config.run_id += "_" + hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()[:10]
    return config


def build_arg_parser(description, *, include_fpga_blocks=False):
    parser = argparse.ArgumentParser(description=description)
    add_common_arguments(parser, include_fpga_blocks=include_fpga_blocks)
    return parser
