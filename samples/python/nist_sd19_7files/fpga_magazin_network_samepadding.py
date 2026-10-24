#!/usr/bin/env python
# coding: utf-8

"""FPGA Magazine LUT-Network comparison experiment (Same Padding).

Positive DW+PW block variants insert extra 32ch->32ch DW+PW middle
blocks immediately before the final 32ch->64ch DW+PW block.
All spatial 3x3 layers use same padding with replicate borders, so
the input HxW is preserved through every DW+PW block.
"""

import csv
import datetime
import json
import os

from experiment_common import (
    append_summary,
    distillation_layer,
    ensure_new_run_dir,
    evaluate_segmentation,
    export_rtl_artifacts,
    maybe_plot_metrics,
    require_binarybrain,
    save_run_config,
    set_global_seed,
    train_tail_model,
    train_whole_model,
    write_final_result_json,
    write_json,
    write_layer_info,
)
from experiment_config import ExperimentConfig, build_arg_parser, config_from_args, slugify
from experiment_dataset import load_or_create_common_dataset, make_data_loaders
from experiment_metrics import scores_to_jsonable
from experiment_lut import (
    fpga_magazine_stage_specs,
    summarize_architecture,
)


MODEL_KIND = "fpga_magazine"


# Training/data options below must match the original run when resuming.
# Display-only flags such as --no-plot and --no-rtl are intentionally omitted.
RESUME_CONFIG_KEYS = (
    "sd19_root", "occupancy", "threshold", "polarity", "interpolation",
    "model_name",
    "architecture_variant",
    "run_seed",
    "dataset_seed",
    "input_h",
    "input_w",
    "rows",
    "cols",
    "tile_size",
    "num_classes",
    "background_class",
    "bg_keep_train",
    "bg_keep_eval",
    "train_mnist_count",
    "val_mnist_count",
    "num_train_tiles",
    "num_val_tiles",
    "num_test_tiles",
    "batch_size",
    "dense_epochs",
    "distill_epochs",
    "cnv_epochs",
    "learning_rate",
    "layer_learning_rate",
    "distill_learning_rate",
    "frame_modulation_size",
    "bin_mode",
    "fpga_dw_pw_blocks",
    "fpga_final_pointwise_layers",
    "common_data_dir",
    "eval_hw",
)


def _layers_from_config(config):
    return int(config.fpga_dw_pw_blocks) + int(config.fpga_final_pointwise_layers)


def _checkpoint_name_before_cnv(start_cnv):
    """Return the checkpoint that represents the state immediately before cnvN."""
    return "best_dense" if int(start_cnv) == 0 else "cnv%d" % (int(start_cnv) - 1)


def _validate_resume_request(args, config):
    """Validate --resume-from-cnv and return its integer value, or None."""
    start_cnv = getattr(args, "resume_from_cnv", None)
    if start_cnv is None:
        return None

    start_cnv = int(start_cnv)
    layers = _layers_from_config(config)
    if not 0 <= start_cnv < layers:
        raise ValueError(
            "--resume-from-cnv must be between 0 and %d, got %d" % (layers - 1, start_cnv)
        )
    if args.skip_training:
        raise ValueError("--resume-from-cnv cannot be used with --skip-training")
    if args.force_dataset:
        raise ValueError(
            "Do not use --force-dataset while resuming: the existing shared dataset must be preserved."
        )
    if not args.allow_existing_run_dir:
        raise ValueError("Resuming requires --allow-existing-run-dir.")

    run_dir = os.path.join(config.run_root, config.run_id)
    if not os.path.isdir(run_dir):
        raise FileNotFoundError("Cannot resume because run directory does not exist: %s" % run_dir)
    return start_cnv


def _validate_resume_config(run_dir, config):
    """Reject a resume if model, data, or optimization settings differ."""
    config_path = os.path.join(run_dir, "config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError("Cannot resume because config.json is missing: %s" % config_path)

    with open(config_path) as f:
        saved = json.load(f)

    current = config.to_dict()
    mismatches = []
    for key in RESUME_CONFIG_KEYS:
        # Runs made before --eval-hw existed used the old common 26x26 crop.
        # Treat a missing key as 26 rather than silently accepting a new 24x24
        # setting during a resumed run.
        if key not in saved:
            if key == "eval_hw":
                saved_value = 26
            else:
                continue
        else:
            saved_value = saved.get(key)
        if saved_value != current.get(key):
            mismatches.append(
                "%s (saved=%r, requested=%r)" % (key, saved_value, current.get(key))
            )
    if mismatches:
        raise ValueError(
            "Resume settings do not match the existing run. Use the same experiment options.\n  "
            + "\n  ".join(mismatches)
        )


def _append_resume_history(run_dir, start_cnv, checkpoint_name):
    """Keep a small audit trail without overwriting the original config.json."""
    record = {
        "resumed_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "resume_from_cnv": int(start_cnv),
        "loaded_checkpoint": checkpoint_name,
    }
    path = os.path.join(run_dir, "resume_history.jsonl")
    with open(path, "a") as f:
        f.write(json.dumps(record, sort_keys=True) + "\n")


def _has_logged_test(run_dir, stage):
    """Return True when a final test row has already been recorded for a stage."""
    metrics_path = os.path.join(run_dir, "logs", "metrics.csv")
    if not os.path.exists(metrics_path):
        return False
    with open(metrics_path, newline="") as f:
        for row in csv.DictReader(f):
            if row.get("stage") == stage and row.get("phase") == "test":
                return True
    return False


def _best_dense_from_logs(run_dir):
    """Recover DenseAffine best-validation metadata for a resumed run."""
    metrics_path = os.path.join(run_dir, "logs", "metrics.csv")
    best_epoch = None
    best_score = -float("inf")
    if not os.path.exists(metrics_path):
        return None

    with open(metrics_path, newline="") as f:
        for row in csv.DictReader(f):
            if row.get("stage") != "dense" or row.get("phase") != "validation":
                continue
            try:
                epoch = int(float(row.get("epoch", "")))
                score = float(row.get("foreground_miou", ""))
            except (TypeError, ValueError):
                continue
            if score > best_score:
                best_epoch = epoch
                best_score = score

    if best_epoch is None:
        return None
    return {"best_epoch": best_epoch, "best_foreground_miou": best_score}


def _fast_forward_train_loader_rng(loaders, config, start_cnv):
    """Restore the train-loader shuffle position at the start of cnvN.

    BinaryBrain checkpoints do not include the torch DataLoader generator
    state.  Replaying completed train-loader passes preserves the original
    shuffle sequence after a process restart without loading image tensors.
    """
    completed_passes = int(config.dense_epochs) + int(start_cnv) * (
        int(config.distill_epochs) + int(config.cnv_epochs)
    )
    if completed_passes <= 0:
        return 0

    train_loader = loaders["train"]
    generator = getattr(train_loader, "generator", None)
    dataset_size = len(train_loader.dataset)
    if generator is None or dataset_size <= 0:
        return 0

    import torch

    class IndexOnlyDataset(torch.utils.data.Dataset):
        def __len__(self):
            return dataset_size

        def __getitem__(self, index):
            return int(index)

    rng_loader = torch.utils.data.DataLoader(
        IndexOnlyDataset(),
        batch_size=dataset_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
        drop_last=False,
    )
    for _ in range(completed_passes):
        for _batch_indices in rng_loader:
            pass
    return completed_passes


def _evaluation_hw_from_config(config):
    return (int(config.eval_hw), int(config.eval_hw))


def create_network(bb, config):
    bin_dtype = bb.DType.BIT if config.bin_mode else bb.DType.FP32

    # Spatial 3x3 layers preserve HxW using replicate padding.
    spatial_padding = "same"
    spatial_border_mode = "replicate"

    def create_lut_conv1(name, output_ch, filter_size=(1, 1), padding="valid", connection="serial", batch_norm=True, fw_dtype=bin_dtype, border_mode="constant"):
        return bb.Convolution2d(
            bb.DifferentiableLut(
                [output_ch, 1, 1],
                connection=connection,
                batch_norm=batch_norm,
                name=(name + "_lut_dl"),
                bin_dtype=fw_dtype,
            ),
            filter_size=filter_size,
            padding=padding,
            border_mode=border_mode,
            name=(name + "_lut_conv"),
            fw_dtype=fw_dtype,
        )

    def create_lut_conv2(name, output_ch, filter_size=(1, 1), padding="valid", connection="serial", batch_norm=True, fw_dtype=bin_dtype, border_mode="constant"):
        return bb.Convolution2d(
            bb.Sequential([
                bb.DifferentiableLut(
                    [output_ch * 6, 1, 1],
                    connection=connection,
                    batch_norm=batch_norm,
                    name=(name + "_lut_dl0"),
                    bin_dtype=fw_dtype,
                ),
                bb.DifferentiableLut(
                    [output_ch, 1, 1],
                    connection="serial",
                    batch_norm=batch_norm,
                    name=(name + "_lut_dl1"),
                    bin_dtype=fw_dtype,
                ),
            ]),
            filter_size=filter_size,
            padding=padding,
            border_mode=border_mode,
            name=(name + "_lut_conv"),
            fw_dtype=fw_dtype,
        )

    def create_lut_depthwise_pointwise(name, depthwise_lut_ch, pointwise_output_ch, padding="valid", batch_norm=True, fw_dtype=bin_dtype):
        return bb.Sequential([
            create_lut_conv1(
                name + "_dw",
                depthwise_lut_ch,
                filter_size=(3, 3),
                padding=padding,
                connection="random",
                batch_norm=batch_norm,
                fw_dtype=fw_dtype,
                border_mode=spatial_border_mode,
            ),
            create_lut_conv2(
                name + "_pw",
                pointwise_output_ch,
                filter_size=(1, 1),
                padding="valid",
                connection="serial",
                batch_norm=batch_norm,
                fw_dtype=fw_dtype,
            ),
        ])

    def create_dense_affine(name, output_ch, fw_dtype=bin_dtype):
        return bb.Sequential([
            bb.DenseAffine([output_ch, 1, 1], name=(name + "_dense_affine")),
            bb.BatchNormalization(name=(name + "_dense_bn")),
            bb.Binarize(name=(name + "_dense_act"), bin_dtype=fw_dtype),
        ])

    def create_dense_conv(name, output_ch, filter_size=(1, 1), padding="valid", fw_dtype=bin_dtype, border_mode="constant"):
        return bb.Convolution2d(
            create_dense_affine(name, output_ch, fw_dtype),
            filter_size=filter_size,
            padding=padding,
            border_mode=border_mode,
            name=(name + "_dense_conv"),
            fw_dtype=fw_dtype,
        )

    dw_pw_blocks = int(config.fpga_dw_pw_blocks)
    final_pointwise_layers = int(config.fpga_final_pointwise_layers)
    layers = dw_pw_blocks + final_pointwise_layers

    class SemanticSegmentationNetwork(bb.Sequential):
        def __init__(self):
            self.real2bin = bb.RealToBinary(frame_modulation_size=config.frame_modulation_size, bin_dtype=bin_dtype)
            self.bin2real = bb.BinaryToReal(frame_integration_size=config.frame_modulation_size, bin_dtype=bin_dtype)
            self.net_cnv = bb.Sequential()
            for i in range(layers):
                layer_name = "cnv%d" % i
                if i < dw_pw_blocks:
                    if i == 0:
                        dense_output_ch = 32
                        depthwise_lut_ch = 32
                        pointwise_output_ch = 32
                    elif i < dw_pw_blocks - 1:
                        dense_output_ch = 32
                        depthwise_lut_ch = 256
                        pointwise_output_ch = 32
                    else:
                        dense_output_ch = 64
                        depthwise_lut_ch = 256
                        pointwise_output_ch = 64
                    cnv_dense = create_dense_conv(
                        layer_name,
                        dense_output_ch,
                        filter_size=(3, 3),
                        padding=spatial_padding,
                        fw_dtype=bin_dtype,
                        border_mode=spatial_border_mode,
                    )
                    cnv_lut = create_lut_depthwise_pointwise(
                        layer_name,
                        depthwise_lut_ch=depthwise_lut_ch,
                        pointwise_output_ch=pointwise_output_ch,
                        padding=spatial_padding,
                        batch_norm=True,
                        fw_dtype=bin_dtype,
                    )
                else:
                    if i == dw_pw_blocks:
                        cnv_dense = create_dense_conv(layer_name, 36, filter_size=(1, 1), padding="valid", fw_dtype=bin_dtype)
                        cnv_lut = create_lut_conv2(layer_name, 36, filter_size=(1, 1), padding="valid", connection="serial", batch_norm=True, fw_dtype=bin_dtype)
                    else:
                        cnv_dense = create_dense_conv(layer_name, 11, filter_size=(1, 1), padding="valid", fw_dtype=bin_dtype)
                        cnv_lut = create_lut_conv2(layer_name, 11, filter_size=(1, 1), padding="valid", connection="serial", batch_norm=True, fw_dtype=bin_dtype)
                self.net_cnv.append(bb.Switcher({"dense": cnv_dense, "lut": cnv_lut}, init_model_name="dense"))
            super(SemanticSegmentationNetwork, self).__init__([self.real2bin, self.net_cnv, self.bin2real])

        def set_input_shape(self, shape):
            shape = self.real2bin.set_input_shape(shape)
            shape = self.net_cnv.set_input_shape(shape)
            shape = self.bin2real.set_input_shape(shape)
            return shape

        def forward(self, x, train):
            x = self.real2bin.forward(x, train)
            x = self.net_cnv.forward(x, train)
            x = self.bin2real.forward(x, train)
            return x

        def backward(self, dy):
            dy = self.bin2real.backward(dy)
            dy = self.net_cnv.backward(dy)
            return dy

    net = SemanticSegmentationNetwork()
    net_name = slugify("%s_%s" % (config.model_name, config.architecture_variant))
    net.set_name(net_name)
    net.send_command("switch_model dense")
    output_shape = net.set_input_shape([1, config.input_h, config.input_w])
    # Same Padding keeps the spatial size identical to the configured input.
    expected_hw = (int(config.input_h), int(config.input_w))
    if tuple(output_shape[-2:]) != expected_hw:
        raise AssertionError("FPGA %d-block output shape %s != expected HW %s" % (dw_pw_blocks, output_shape, expected_hw))
    net.send_command("binary true")
    return net, output_shape, bin_dtype, layers, net_name


def describe_without_binarybrain(config):
    # Dry-run must describe the Same Padding output, not the old valid-padding shrinkage.
    output_hw = (int(config.input_h), int(config.input_w))
    output_shape = [config.num_classes, output_hw[0], output_hw[1]]
    return summarize_architecture(
        config.model_name,
        config.architecture_variant,
        fpga_magazine_stage_specs(config.fpga_dw_pw_blocks),
        output_shape,
    )



def _load_saved_config_for_retest(run_dir, eval_hw):
    """Rebuild the original ExperimentConfig and override only eval_hw.

    The saved config preserves the original block count, shared-dataset
    location, seed, network dimensions, frame settings, and binary mode.
    """
    config_path = os.path.join(run_dir, "config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError("Cannot re-test because config.json is missing: %s" % config_path)
    with open(config_path) as f:
        saved = json.load(f)

    field_names = set(ExperimentConfig.__dataclass_fields__.keys())
    values = {key: saved[key] for key in field_names if key in saved}
    config = ExperimentConfig(**values)
    if config.model_name != MODEL_KIND:
        raise ValueError(
            "--retest-final expects an fpga_magazine run, but config.json has model_name=%r"
            % config.model_name
        )
    if not str(config.architecture_variant).startswith("samepadding_"):
        raise ValueError(
            "--retest-final with this script requires a Same Padding FPGA Magazine run; "
            "saved architecture_variant=%r" % config.architecture_variant
        )
    config.eval_hw = config.eval_hw if eval_hw is None else int(eval_hw)
    return config


def _retest_output_path(run_dir, evaluation_hw):
    h, w = evaluation_hw
    directory = os.path.join(run_dir, "re_evaluations")
    os.makedirs(directory, exist_ok=True)
    return os.path.join(directory, "final_test_center_%dx%d.json" % (h, w))


def run_saved_final_test(args):
    """Evaluate best_dense / best_lut checkpoints without re-training.

    The original metrics.csv, class_iou.csv, final_results.json, summary.csv,
    checkpoints, and RTL files are intentionally left unchanged.
    """
    if not args.run_id:
        raise ValueError("--retest-final requires --run-id of the completed run")
    if args.resume_from_cnv is not None:
        raise ValueError("--retest-final cannot be combined with --resume-from-cnv")
    if args.force_dataset:
        raise ValueError("Do not use --force-dataset with --retest-final")

    run_dir = os.path.join(args.run_root, slugify(args.run_id))
    if not os.path.isdir(run_dir):
        raise FileNotFoundError("Run directory does not exist: %s" % run_dir)

    config = _load_saved_config_for_retest(run_dir, args.eval_hw)
    evaluation_hw = _evaluation_hw_from_config(config)

    bb = require_binarybrain()
    set_global_seed(config.run_seed)
    try:
        bb.set_device(0)
    except Exception:
        pass

    common_data = load_or_create_common_dataset(config, force=False)
    from experiment_common import configure_timing
    configure_timing(run_dir)
    loaders = make_data_loaders(config, common_data)
    net, output_shape, _bin_dtype, _layers, _net_name = create_network(bb, config)

    model_hw = tuple(int(v) for v in output_shape[-2:])
    if evaluation_hw[0] > model_hw[0] or evaluation_hw[1] > model_hw[1]:
        raise ValueError(
            "--eval-hw %dx%d is larger than the saved model output %dx%d. "
            "Use an evaluation size no larger than the model output."
            % (evaluation_hw[0], evaluation_hw[1], model_hw[0], model_hw[1])
        )

    checkpoint_dir = os.path.join(run_dir, "checkpoints")
    results = {
        "source_run_dir": run_dir,
        "re_evaluated_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "model_name": config.model_name,
        "architecture_variant": config.architecture_variant,
        "fpga_dw_pw_blocks": int(config.fpga_dw_pw_blocks),
        "evaluation_crop_hw": list(evaluation_hw),
        "model_output_hw": list(model_hw),
        "note": (
            "Only the central evaluation crop was changed. The saved checkpoints "
            "were loaded as-is; no DenseAffine/LUT training, distillation, RTL export, "
            "or original-log update was performed."
        ),
        "dense_test": None,
        "lut_test": None,
    }

    if not bb.load_networks(checkpoint_dir, net, name="best_dense", verbose=False):
        raise FileNotFoundError(
            "best_dense checkpoint is required for --retest-final: %s" % checkpoint_dir
        )
    net.send_command("switch_model dense")
    dense_test = evaluate_segmentation(
        bb,
        net,
        loaders["test"],
        output_shape=output_shape,
        model_kind=MODEL_KIND,
        title="dense final test center %dx%d" % evaluation_hw,
        evaluation_hw=evaluation_hw,
    )
    results["dense_test"] = {
        "loss": float(dense_test["loss"]),
        "scores": scores_to_jsonable(dense_test["scores"]),
    }

    if bb.load_networks(checkpoint_dir, net, name="best_lut", verbose=False):
        net.send_command("switch_model lut")
        lut_test = evaluate_segmentation(
            bb,
            net,
            loaders["test"],
            output_shape=output_shape,
            model_kind=MODEL_KIND,
            title="lut final test center %dx%d" % evaluation_hw,
            evaluation_hw=evaluation_hw,
        )
        results["lut_test"] = {
            "loss": float(lut_test["loss"]),
            "scores": scores_to_jsonable(lut_test["scores"]),
        }
    else:
        results["lut_checkpoint_warning"] = (
            "best_lut checkpoint was not found; only DenseAffine was evaluated."
        )

    output_path = _retest_output_path(run_dir, evaluation_hw)
    write_json(output_path, results)
    print("Re-test result saved: %s" % output_path)

def run_experiment(args):
    if args.retest_final:
        run_saved_final_test(args)
        return

    blocks = int(args.fpga_dw_pw_blocks)
    architecture_variant = "samepadding_replicate_dw_pw_%d_blocks" % blocks
    config = config_from_args(
        args,
        model_name=MODEL_KIND,
        architecture_variant=architecture_variant,
        fpga_dw_pw_blocks=blocks,
    )

    if args.dry_run:
        common_data = load_or_create_common_dataset(config, dry_run=True)
        arch = describe_without_binarybrain(config)
        print(json.dumps({
            "dry_run": True,
            "architecture": arch,
            "source_shape_train": list(common_data["source_imgs_train"].shape),
            "teacher_shape_train": list(common_data["teacher_imgs_train_full"].shape),
        }, indent=2, sort_keys=True))
        return

    if args.prepare_dataset_only:
        load_or_create_common_dataset(config, force=args.force_dataset)
        print("Common dataset is ready: %s" % config.common_data_dir)
        return

    resume_from_cnv = _validate_resume_request(args, config)
    is_resume = resume_from_cnv is not None

    bb = require_binarybrain()
    set_global_seed(config.run_seed)
    try:
        bb.set_device(0)
    except Exception:
        pass

    # Reuse exactly the same dataset split when continuing a partial run.
    common_data = load_or_create_common_dataset(config, force=args.force_dataset)
    if is_resume:
        existing_run_dir = os.path.join(config.run_root, config.run_id)
        _validate_resume_config(existing_run_dir, config)

    run_dir = ensure_new_run_dir(
        config,
        allow_existing=(args.allow_existing_run_dir or is_resume),
    )
    from experiment_common import configure_timing
    configure_timing(run_dir)
    loaders = make_data_loaders(config, common_data)
    net, output_shape, bin_dtype, layers, net_name = create_network(bb, config)

    architecture_summary = summarize_architecture(
        config.model_name,
        config.architecture_variant,
        fpga_magazine_stage_specs(config.fpga_dw_pw_blocks),
        output_shape,
    )

    # Keep original run metadata intact on resume; only resume_history.jsonl
    # and later training logs/checkpoints are appended.
    if not is_resume:
        write_json(os.path.join(run_dir, "architecture.json"), architecture_summary)
        save_run_config(run_dir, config, architecture_summary, bb=bb)
        net.send_command("switch_model lut")
        write_layer_info(run_dir, net, net_name)
        net.send_command("switch_model dense")

    checkpoint_dir = os.path.join(run_dir, "checkpoints")
    dense_best = _best_dense_from_logs(run_dir) if is_resume else None
    lut_best = None
    dense_test = None
    lut_test = None

    if not args.skip_training:
        start_cnv = 0

        if is_resume:
            start_cnv = resume_from_cnv
            checkpoint_name = _checkpoint_name_before_cnv(start_cnv)
            print(
                "Resuming LUT conversion from cnv%d using checkpoint '%s'."
                % (start_cnv, checkpoint_name)
            )
            if not bb.load_networks(checkpoint_dir, net, name=checkpoint_name, verbose=False):
                raise FileNotFoundError(
                    "Required checkpoint '%s' was not found or could not be loaded in %s"
                    % (checkpoint_name, checkpoint_dir)
                )

            # A checkpoint is saved after cnv(N-1) completed.  Reconstruct the
            # matching partial-LUT routing before starting cnvN.
            net.send_command("switch_model dense")
            net.send_command("parameter_lock true")
            for j in range(start_cnv):
                net.net_cnv[j].send_command("switch_model lut")

            rng_passes = _fast_forward_train_loader_rng(loaders, config, start_cnv)
            print("Restored train-loader shuffle state after %d completed passes." % rng_passes)
            _append_resume_history(run_dir, start_cnv, checkpoint_name)
        else:
            net.send_command("switch_model dense")
            net.send_command("parameter_lock false")
            dense_best = train_whole_model(
                bb,
                net,
                loaders,
                run_dir=run_dir,
                output_shape=output_shape,
                model_kind=MODEL_KIND,
                stage="dense",
                model_mode="dense",
                epochs=config.dense_epochs,
                learning_rate=config.learning_rate,
                best_checkpoint_name="best_dense",
                plot_enabled=config.plot_enabled,
                plot_every=config.plot_every,
                dense_class_iou_bar_every=config.dense_class_iou_bar_every,
                evaluation_hw=_evaluation_hw_from_config(config),
            )

            if not bb.load_networks(checkpoint_dir, net, name="best_dense", verbose=False):
                raise RuntimeError("best_dense checkpoint could not be loaded after DenseAffine training")
            net.send_command("switch_model dense")
            dense_test = evaluate_segmentation(
                bb,
                net,
                loaders["test"],
                output_shape=output_shape,
                model_kind=MODEL_KIND,
                title="dense final test",
                evaluation_hw=_evaluation_hw_from_config(config),
            )
            log_final_test(run_dir, "dense_final", "dense", dense_test)

            if not bb.load_networks(checkpoint_dir, net, name="best_dense", verbose=False):
                raise RuntimeError("best_dense checkpoint could not be reloaded before LUT conversion")

        for i in range(start_cnv, layers):
            layer_name = "cnv%d" % i
            print("----- %s -----" % layer_name)
            if config.distill_epochs > 0:
                distillation_layer(
                    bb,
                    net,
                    loaders["train"],
                    run_dir=run_dir,
                    index=i,
                    epochs=config.distill_epochs,
                    frame_modulation_size=config.frame_modulation_size,
                    bin_dtype=bin_dtype,
                    learning_rate=config.distill_learning_rate,
                )

            net.send_command("switch_model dense")
            net.send_command("parameter_lock true")
            for j in range(i + 1):
                net.net_cnv[j].send_command("switch_model lut")

            net.net_cnv[i].send_command("parameter_lock false")
            net.send_command("parameter_lock false")
            for j in range(i):
                net.net_cnv[j].send_command("parameter_lock true")

            if i == layers - 1:
                tracker = {"score": -float("inf")}
                train_tail_model(
                    bb,
                    net,
                    loaders,
                    run_dir=run_dir,
                    output_shape=output_shape,
                    model_kind=MODEL_KIND,
                    stage=layer_name,
                    model_mode="lut",
                    epochs=config.cnv_epochs,
                    learning_rate=config.layer_learning_rate,
                    start_index=i,
                    layers=layers,
                    best_tracker=tracker,
                    best_checkpoint_name="best_lut",
                    plot_enabled=config.plot_enabled,
                    plot_every=config.plot_every,
                    dense_class_iou_bar_every=config.dense_class_iou_bar_every,
                    evaluation_hw=_evaluation_hw_from_config(config),
                )
                lut_best = tracker
            else:
                train_tail_model(
                    bb,
                    net,
                    loaders,
                    run_dir=run_dir,
                    output_shape=output_shape,
                    model_kind=MODEL_KIND,
                    stage=layer_name,
                    model_mode="partial_lut",
                    epochs=config.cnv_epochs,
                    learning_rate=config.layer_learning_rate,
                    start_index=i,
                    layers=layers,
                    best_tracker={"score": -float("inf")},
                    best_checkpoint_name="best_%s" % layer_name,
                    plot_enabled=config.plot_enabled,
                    plot_every=config.plot_every,
                    dense_class_iou_bar_every=config.dense_class_iou_bar_every,
                    evaluation_hw=_evaluation_hw_from_config(config),
                )
                bb.save_networks(checkpoint_dir, net, name=layer_name, backups=-1)

        if not os.path.exists(os.path.join(checkpoint_dir, "best_lut")):
            bb.save_networks(checkpoint_dir, net, name="best_lut", backups=-1)
            lut_best = {
                "stage": "cnv%d" % (layers - 1),
                "epoch": config.cnv_epochs - 1,
                "score": None,
            }

    if dense_test is None:
        if not bb.load_networks(checkpoint_dir, net, name="best_dense", verbose=False):
            raise FileNotFoundError(
                "best_dense checkpoint is required to evaluate the final DenseAffine model"
            )
        net.send_command("switch_model dense")
        dense_test = evaluate_segmentation(
            bb,
            net,
            loaders["test"],
            output_shape=output_shape,
            model_kind=MODEL_KIND,
            title="dense final test",
            evaluation_hw=_evaluation_hw_from_config(config),
        )
        if not _has_logged_test(run_dir, "dense_final"):
            log_final_test(run_dir, "dense_final", "dense", dense_test)

    if bb.load_networks(checkpoint_dir, net, name="best_lut", verbose=False):
        net.send_command("switch_model lut")
        lut_test = evaluate_segmentation(
            bb,
            net,
            loaders["test"],
            output_shape=output_shape,
            model_kind=MODEL_KIND,
            title="lut final test",
            evaluation_hw=_evaluation_hw_from_config(config),
        )
        if not _has_logged_test(run_dir, "lut_final"):
            log_final_test(run_dir, "lut_final", "lut", lut_test)

    if dense_test is not None and lut_test is not None:
        final_results_path = os.path.join(run_dir, "final_results.json")
        had_final_results = os.path.exists(final_results_path)
        write_final_result_json(
            run_dir,
            dense_test,
            lut_test,
            dense_best,
            lut_best,
            evaluation_hw=_evaluation_hw_from_config(config),
        )
        # Avoid appending a duplicate completed run when --skip-training is
        # used after an already-finished experiment.
        if not had_final_results:
            append_summary(
                run_dir,
                config,
                architecture_summary,
                dense_test,
                lut_test,
                dense_best or {},
                lut_best or {},
            )

    if config.plot_enabled:
        maybe_plot_metrics(run_dir)

    if not args.no_rtl and lut_test is not None:
        export_rtl_artifacts(
            bb,
            net,
            common_data,
            run_dir=run_dir,
            module_name="%s_%s" % (net_name, config.run_id),
            copy_to_sim_path=config.rtl_sim_path if args.copy_rtl_sim else None,
        )

    print("Run directory: %s" % run_dir)

def log_final_test(run_dir, stage, model_mode, result):
    from experiment_common import log_scores

    log_scores(run_dir, stage=stage, epoch=0, phase="test", model_mode=model_mode, result=result)


def main():
    parser = build_arg_parser("FPGA Magazine LUT-Network comparison experiment", include_fpga_blocks=True)
    parser.add_argument(
        "--retest-final",
        action="store_true",
        help=(
            "Load the existing best_dense / best_lut checkpoints and re-run only the final test. "
            "Results are written under <run>/re_evaluations/ and original logs are preserved. "
            "Use with --run-id. For the full Same Padding output use --eval-hw 28; "
            "24 or 26 can be used for center-crop comparisons."
        ),
    )
    parser.add_argument(
        "--resume-from-cnv",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Resume LUT conversion at cnvN using checkpoints/cnv(N-1). "
            "Use N=0 to resume from best_dense. Requires --allow-existing-run-dir."
        ),
    )
    args = parser.parse_args()
    from experiment_common import run_smoke_test, run_camera_preview, run_loader_benchmark
    if args.smoke_test:
        run_smoke_test(args, model_kind=MODEL_KIND, create_network=create_network)
    elif args.camera_run_dir:
        run_camera_preview(args, model_kind=MODEL_KIND, create_network=create_network)
    elif args.benchmark_loader:
        run_loader_benchmark(args, model_kind=MODEL_KIND)
    else:
        run_experiment(args)


from experiment_common import _timed_phase
create_network = _timed_phase(create_network)

if __name__ == "__main__":
    main()
