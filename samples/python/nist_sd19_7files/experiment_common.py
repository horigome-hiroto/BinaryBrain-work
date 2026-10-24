#!/usr/bin/env python
# coding: utf-8

"""Shared run, evaluation, logging, and training helpers."""

import csv
import datetime
import json
import math
import os
import platform
import random
import shutil
import subprocess
import sys
import tempfile
import time

import numpy as np

from experiment_config import CLASS_NAMES, slugify
from experiment_metrics import (
    SegmentationMetrics,
    crop_prediction_for_common_eval,
    crop_teacher_for_common_eval,
    crop_teacher_for_model_output,
    scores_to_jsonable,
)


TRAINING_EVAL_NOTE = (
    "Training loss uses the teacher crop that matches each model output size. "
    "Validation/test mIoU and Class IoU use the configured central evaluation crop "
    "(full input extent by default; explicit --eval-hw for central evaluation)."
)


def set_global_seed(seed):
    random.seed(int(seed))
    np.random.seed(int(seed))
    try:
        import torch

        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed(int(seed))
            torch.cuda.manual_seed_all(int(seed))
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass


def require_binarybrain():
    try:
        import binarybrain as bb
    except ImportError as exc:
        raise RuntimeError("BinaryBrain extension unavailable. Build the repository binding first; this checkout currently has a CPU build error in Concatenate.h:264. No alternate network is substituted.") from exc
    if not getattr(bb.save_networks, "_sd19_timed", False):
        measured = _timed_phase(bb.save_networks)
        measured._sd19_timed = True
        bb.save_networks = measured
    return bb


def ensure_new_run_dir(config, *, allow_existing=False):
    run_dir = os.path.join(config.run_root, config.run_id)
    if os.path.exists(run_dir) and not allow_existing:
        raise FileExistsError(
            "run directory already exists: %s. Use a new --run-id or --allow-existing-run-dir." % run_dir
        )
    os.makedirs(run_dir, exist_ok=True)
    for subdir in ["checkpoints", "logs", "graphs", "rtl", "predictions"]:
        os.makedirs(os.path.join(run_dir, subdir), exist_ok=True)
    return run_dir


def _run_command(args):
    try:
        return subprocess.check_output(args, stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return None


def collect_environment_info(bb=None):
    info = {
        "python_version": sys.version.replace("\n", " "),
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "run_datetime": datetime.datetime.now().isoformat(timespec="seconds"),
        "git_commit_hash": _run_command(["git", "rev-parse", "HEAD"]),
    }
    try:
        import torch

        info["torch_version"] = torch.__version__
        info["cuda_available"] = bool(torch.cuda.is_available())
        info["gpu_name"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except Exception as e:
        info["torch_version"] = None
        info["cuda_available"] = None
        info["gpu_name"] = None
        info["torch_import_error"] = "%s: %s" % (type(e).__name__, e)
    if bb is not None:
        try:
            info["binarybrain_version"] = bb.get_version_string()
        except Exception as e:
            info["binarybrain_version"] = None
            info["binarybrain_version_error"] = "%s: %s" % (type(e).__name__, e)
        try:
            info["binarybrain_device_name"] = bb.get_device_name(0)
        except Exception as e:
            info["binarybrain_device_name"] = None
            info["binarybrain_device_error"] = "%s: %s" % (type(e).__name__, e)
    else:
        info["binarybrain_version"] = None
    return info


def write_json(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, sort_keys=True)


def save_run_config(run_dir, config, architecture_summary, bb=None):
    data = config.to_dict()
    data["training_and_evaluation_note"] = TRAINING_EVAL_NOTE
    data["evaluation_crop_hw"] = [int(config.eval_hw), int(config.eval_hw)]
    data["architecture_summary"] = architecture_summary
    data["environment"] = collect_environment_info(bb)
    write_json(os.path.join(run_dir, "config.json"), data)


def _safe_float(value):
    if value is None:
        return ""
    try:
        value = float(value)
    except (TypeError, ValueError):
        return ""
    if math.isnan(value) or math.isinf(value):
        return ""
    return value


def append_dict_row(path, row, fieldnames):
    """Append one CSV row, expanding an existing header when fields are added.

    Keeping this schema-migration step here prevents malformed CSV rows when a
    new experiment field (for example ``evaluation_crop_hw``) is introduced
    after summary.csv has already been created by an older code version.
    Existing values are preserved and new columns are left blank for old rows.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    requested_fields = list(fieldnames)
    actual_fields = requested_fields

    if os.path.exists(path):
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            existing_fields = list(reader.fieldnames or [])
            existing_rows = list(reader)

        # Preserve any existing columns and add newly requested ones at the
        # end.  This also makes future CSV schema additions backward-safe.
        actual_fields = list(existing_fields)
        for field in requested_fields:
            if field not in actual_fields:
                actual_fields.append(field)

        if actual_fields != existing_fields:
            temp_path = path + ".tmp"
            with open(temp_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=actual_fields, extrasaction="ignore")
                writer.writeheader()
                for existing_row in existing_rows:
                    writer.writerow(existing_row)
            os.replace(temp_path, path)

    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=actual_fields, extrasaction="ignore")
        if not os.path.exists(path) or os.path.getsize(path) == 0:
            writer.writeheader()
        writer.writerow(row)


METRIC_FIELDS = [
    "stage",
    "epoch",
    "phase",
    "model_mode",
    "train_update_loss",
    "loss",
    "all_miou",
    "foreground_miou",
    "pixel_acc",
    "valid_pixel_count",
]

CLASS_IOU_FIELDS = [
    "stage",
    "epoch",
    "phase",
    "model_mode",
    "class_id",
    "class_name",
    "iou",
]


# Wall-clock durations for update passes only.  Validation/test evaluation,
# plotting, checkpoint I/O, and RTL export are intentionally excluded so the
# values represent learning work rather than total script runtime.
TRAINING_TIME_FIELDS = [
    "stage",
    "epoch",
    "phase",
    "model_mode",
    "started_at",
    "finished_at",
    "duration_seconds",
]


def log_training_time(
    run_dir,
    *,
    stage,
    epoch,
    phase,
    model_mode,
    started_at,
    finished_at,
    duration_seconds,
):
    """Append a completed learning-pass duration to logs/training_time.csv."""
    append_dict_row(
        os.path.join(run_dir, "logs", "training_time.csv"),
        {
            "stage": stage,
            "epoch": epoch,
            "phase": phase,
            "model_mode": model_mode,
            "started_at": started_at,
            "finished_at": finished_at,
            "duration_seconds": _safe_float(duration_seconds),
        },
        TRAINING_TIME_FIELDS,
    )


def log_scores(run_dir, *, stage, epoch, phase, model_mode, result=None, train_update_loss=None):
    scores = result["scores"] if result is not None else {}
    row = {
        "stage": stage,
        "epoch": epoch,
        "phase": phase,
        "model_mode": model_mode,
        "train_update_loss": _safe_float(train_update_loss),
        "loss": _safe_float(result.get("loss") if result else None),
        "all_miou": _safe_float(scores.get("all_miou")),
        "foreground_miou": _safe_float(scores.get("foreground_miou")),
        "pixel_acc": _safe_float(scores.get("pixel_acc")),
        "valid_pixel_count": scores.get("valid_pixel_count", ""),
    }
    append_dict_row(os.path.join(run_dir, "logs", "metrics.csv"), row, METRIC_FIELDS)
    if result is None:
        return
    class_iou = np.asarray(scores["class_iou"], dtype=np.float64)
    for class_id, iou in enumerate(class_iou):
        append_dict_row(
            os.path.join(run_dir, "logs", "class_iou.csv"),
            {
                "stage": stage,
                "epoch": epoch,
                "phase": phase,
                "model_mode": model_mode,
                "class_id": class_id,
                "class_name": CLASS_NAMES[class_id] if class_id < len(CLASS_NAMES) else str(class_id),
                "iou": _safe_float(iou),
            },
            CLASS_IOU_FIELDS,
        )


def _format_float(value, digits=4):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "nan"
    if math.isnan(value):
        return "nan"
    if math.isinf(value):
        return "inf" if value > 0 else "-inf"
    return ("%." + str(digits) + "f") % value


def print_class_iou(title, scores):
    class_iou = np.asarray(scores["class_iou"], dtype=np.float64)
    parts = []
    for class_id, iou in enumerate(class_iou):
        name = CLASS_NAMES[class_id] if class_id < len(CLASS_NAMES) else str(class_id)
        parts.append("%s=%s" % (name, _format_float(iou)))
    print("[%s] class_iou: %s" % (title, " ".join(parts)))


def _progress(iterable, desc):
    try:
        from tqdm import tqdm

        return tqdm(iterable, desc=desc)
    except Exception:
        return iterable


def _progress_postfix(progress, **kwargs):
    if hasattr(progress, "set_postfix"):
        progress.set_postfix(**kwargs)


def _training_progress_update(progress, loss, seg_metrics):
    scores = seg_metrics.get_scores()
    _progress_postfix(
        progress,
        loss=_format_float(loss.get()),
        pixel_acc=_format_float(scores["pixel_acc"]),
        fg_mIoU=_format_float(scores["foreground_miou"]),
    )


def evaluate_segmentation(
    bb,
    net,
    loader,
    *,
    output_shape,
    model_kind,
    title,
    evaluation_hw=(26, 26),
):
    loss = bb.LossSoftmaxCrossEntropy()
    seg_metrics = SegmentationMetrics()
    loss.clear()
    seg_metrics.clear()
    for x_imgs, teacher_full in loader:
        x_np = np.asarray(x_imgs, dtype=np.float32)
        teacher_full_np = np.asarray(teacher_full)
        teacher_model_np = crop_teacher_for_model_output(teacher_full_np, output_shape).astype(np.float32)
        x_buf = bb.FrameBuffer.from_numpy(x_np)
        t_buf = bb.FrameBuffer.from_numpy(teacher_model_np)
        y_buf = net.forward(x_buf, train=False)
        loss.calculate(y_buf, t_buf)
        y_np = y_buf.numpy()
        pred_common = crop_prediction_for_common_eval(
            y_np,
            model_kind,
            output_shape=output_shape,
            eval_hw=evaluation_hw,
        )
        teacher_common = crop_teacher_for_common_eval(teacher_full_np, eval_hw=evaluation_hw)
        seg_metrics.calculate_numpy(pred_common, teacher_common)
    scores = seg_metrics.get_scores()
    print(
        "[%s] loss=%f all_mIoU=%f foreground_mIoU=%f pixel_acc=%f valid_pixels=%d"
        % (
            title,
            loss.get(),
            scores["all_miou"],
            scores["foreground_miou"],
            scores["pixel_acc"],
            scores["valid_pixel_count"],
        )
    )
    print_class_iou(title, scores)
    return {"loss": float(loss.get()), "scores": scores}


def _train_epoch_whole(
    bb,
    net,
    loader,
    *,
    output_shape,
    model_kind,
    optimizer,
    desc,
    evaluation_hw=(26, 26),
):
    loss = bb.LossSoftmaxCrossEntropy()
    seg_metrics = SegmentationMetrics()
    loss.clear()
    seg_metrics.clear()
    progress = _progress(loader, desc)
    for x_imgs, teacher_full in progress:
        x_np = np.asarray(x_imgs, dtype=np.float32)
        teacher_full_np = np.asarray(teacher_full)
        teacher_np = crop_teacher_for_model_output(teacher_full_np, output_shape).astype(np.float32)
        x_buf = bb.FrameBuffer.from_numpy(x_np)
        t_buf = bb.FrameBuffer.from_numpy(teacher_np)
        y_buf = net.forward(x_buf, train=True)
        dy_buf = loss.calculate(y_buf, t_buf)
        pred_common = crop_prediction_for_common_eval(
            y_buf.numpy(),
            model_kind,
            output_shape=output_shape,
            eval_hw=evaluation_hw,
        )
        teacher_common = crop_teacher_for_common_eval(teacher_full_np, eval_hw=evaluation_hw)
        seg_metrics.calculate_numpy(pred_common, teacher_common)
        _training_progress_update(progress, loss, seg_metrics)
        net.backward(dy_buf)
        optimizer.update()
    return float(loss.get())


def _train_epoch_tail(
    bb,
    net,
    loader,
    *,
    output_shape,
    model_kind,
    optimizer,
    start_index,
    layers,
    desc,
    evaluation_hw=(26, 26),
):
    loss = bb.LossSoftmaxCrossEntropy()
    seg_metrics = SegmentationMetrics()
    loss.clear()
    seg_metrics.clear()
    progress = _progress(loader, desc)
    for x_imgs, teacher_full in progress:
        x_np = np.asarray(x_imgs, dtype=np.float32)
        teacher_full_np = np.asarray(teacher_full)
        teacher_np = crop_teacher_for_model_output(teacher_full_np, output_shape).astype(np.float32)
        x_buf = bb.FrameBuffer.from_numpy(x_np)
        t_buf = bb.FrameBuffer.from_numpy(teacher_np)

        x_buf = net.real2bin.forward(x_buf, train=False)
        for i in range(layers):
            x_buf = net.net_cnv[i].forward(x_buf, train=(i >= start_index))
        y_buf = net.bin2real.forward(x_buf, train=True)

        dy_buf = loss.calculate(y_buf, t_buf)
        pred_common = crop_prediction_for_common_eval(
            y_buf.numpy(),
            model_kind,
            output_shape=output_shape,
            eval_hw=evaluation_hw,
        )
        teacher_common = crop_teacher_for_common_eval(teacher_full_np, eval_hw=evaluation_hw)
        seg_metrics.calculate_numpy(pred_common, teacher_common)
        _training_progress_update(progress, loss, seg_metrics)
        dy_buf = net.bin2real.backward(dy_buf)
        for i in reversed(range(start_index, layers)):
            dy_buf = net.net_cnv[i].backward(dy_buf)
        optimizer.update()
    return float(loss.get())


def train_whole_model(
    bb,
    net,
    loaders,
    *,
    run_dir,
    output_shape,
    model_kind,
    stage,
    model_mode,
    epochs,
    learning_rate,
    best_checkpoint_name,
    plot_enabled=True,
    plot_every=1,
    dense_class_iou_bar_every=10,
    evaluation_hw=(26, 26),
):
    optimizer = bb.OptimizerAdam(learning_rate=learning_rate)
    optimizer.set_variables(net.get_parameters(), net.get_gradients())
    best_score = -float("inf")
    best_epoch = None
    checkpoint_dir = os.path.join(run_dir, "checkpoints")
    for epoch in range(int(epochs)):
        started_at = datetime.datetime.now().isoformat(timespec="seconds")
        start_time = time.perf_counter()
        train_update_loss = _train_epoch_whole(
            bb,
            net,
            loaders["train"],
            output_shape=output_shape,
            model_kind=model_kind,
            optimizer=optimizer,
            desc="%s train %d/%d" % (stage, epoch + 1, int(epochs)),
            evaluation_hw=evaluation_hw,
        )
        duration_seconds = time.perf_counter() - start_time
        finished_at = datetime.datetime.now().isoformat(timespec="seconds")
        log_training_time(
            run_dir,
            stage=stage,
            epoch=epoch,
            phase="train_update",
            model_mode=model_mode,
            started_at=started_at,
            finished_at=finished_at,
            duration_seconds=duration_seconds,
        )
        log_scores(
            run_dir,
            stage=stage,
            epoch=epoch,
            phase="train_update",
            model_mode=model_mode,
            result=None,
            train_update_loss=train_update_loss,
        )
        train_eval = evaluate_segmentation(
            bb,
            net,
            loaders["train_eval"],
            output_shape=output_shape,
            model_kind=model_kind,
            title="%s train_eval %d" % (stage, epoch),
            evaluation_hw=evaluation_hw,
        )
        val_eval = evaluate_segmentation(
            bb,
            net,
            loaders["val"],
            output_shape=output_shape,
            model_kind=model_kind,
            title="%s validation %d" % (stage, epoch),
            evaluation_hw=evaluation_hw,
        )
        log_scores(run_dir, stage=stage, epoch=epoch, phase="train_eval", model_mode=model_mode, result=train_eval)
        log_scores(run_dir, stage=stage, epoch=epoch, phase="validation", model_mode=model_mode, result=val_eval)
        update_graphs_after_epoch(
            run_dir,
            stage=stage,
            epoch=epoch,
            total_epochs=int(epochs),
            plot_enabled=plot_enabled,
            plot_every=plot_every,
            dense_class_iou_bar_every=dense_class_iou_bar_every,
        )
        score = val_eval["scores"]["foreground_miou"]
        if score > best_score:
            best_score = score
            best_epoch = epoch
            bb.save_networks(checkpoint_dir, net, name=best_checkpoint_name, backups=-1)
    return {"best_epoch": best_epoch, "best_foreground_miou": best_score}


def train_tail_model(
    bb,
    net,
    loaders,
    *,
    run_dir,
    output_shape,
    model_kind,
    stage,
    model_mode,
    epochs,
    learning_rate,
    start_index,
    layers,
    best_tracker,
    best_checkpoint_name,
    plot_enabled=True,
    plot_every=1,
    dense_class_iou_bar_every=10,
    evaluation_hw=(26, 26),
):
    optimizer = bb.OptimizerAdam(learning_rate=learning_rate)
    optimizer.set_variables(net.get_parameters(), net.get_gradients())
    checkpoint_dir = os.path.join(run_dir, "checkpoints")
    local_best_epoch = None
    for epoch in range(int(epochs)):
        started_at = datetime.datetime.now().isoformat(timespec="seconds")
        start_time = time.perf_counter()
        train_update_loss = _train_epoch_tail(
            bb,
            net,
            loaders["train"],
            output_shape=output_shape,
            model_kind=model_kind,
            optimizer=optimizer,
            start_index=start_index,
            layers=layers,
            desc="%s train %d/%d" % (stage, epoch + 1, int(epochs)),
            evaluation_hw=evaluation_hw,
        )
        duration_seconds = time.perf_counter() - start_time
        finished_at = datetime.datetime.now().isoformat(timespec="seconds")
        log_training_time(
            run_dir,
            stage=stage,
            epoch=epoch,
            phase="train_update",
            model_mode=model_mode,
            started_at=started_at,
            finished_at=finished_at,
            duration_seconds=duration_seconds,
        )
        log_scores(
            run_dir,
            stage=stage,
            epoch=epoch,
            phase="train_update",
            model_mode=model_mode,
            result=None,
            train_update_loss=train_update_loss,
        )
        train_eval = evaluate_segmentation(
            bb,
            net,
            loaders["train_eval"],
            output_shape=output_shape,
            model_kind=model_kind,
            title="%s train_eval %d" % (stage, epoch),
            evaluation_hw=evaluation_hw,
        )
        val_eval = evaluate_segmentation(
            bb,
            net,
            loaders["val"],
            output_shape=output_shape,
            model_kind=model_kind,
            title="%s validation %d" % (stage, epoch),
            evaluation_hw=evaluation_hw,
        )
        log_scores(run_dir, stage=stage, epoch=epoch, phase="train_eval", model_mode=model_mode, result=train_eval)
        log_scores(run_dir, stage=stage, epoch=epoch, phase="validation", model_mode=model_mode, result=val_eval)
        update_graphs_after_epoch(
            run_dir,
            stage=stage,
            epoch=epoch,
            total_epochs=int(epochs),
            plot_enabled=plot_enabled,
            plot_every=plot_every,
            dense_class_iou_bar_every=dense_class_iou_bar_every,
        )
        score = val_eval["scores"]["foreground_miou"]
        if score > best_tracker.get("score", -float("inf")):
            best_tracker["score"] = score
            best_tracker["stage"] = stage
            best_tracker["epoch"] = epoch
            local_best_epoch = epoch
            bb.save_networks(checkpoint_dir, net, name=best_checkpoint_name, backups=-1)
    return {"best_epoch": local_best_epoch, "best_foreground_miou": best_tracker.get("score")}


def distillation_layer(bb, net, loader, *, run_dir, index, epochs, frame_modulation_size, bin_dtype, learning_rate=0.001):
    loss = bb.LossMeanSquaredError()
    optimizer = bb.OptimizerAdam(learning_rate=learning_rate)
    cnv_layer = net.net_cnv[index]
    bin2real0 = bb.BinaryToReal(frame_integration_size=frame_modulation_size, bin_dtype=bin_dtype)
    bin2real1 = bb.BinaryToReal(frame_integration_size=frame_modulation_size, bin_dtype=bin_dtype)

    cnv_layer.send_command("switch_model lut")
    cnv_layer.send_command("parameter_lock false")
    optimizer.set_variables(cnv_layer.get_parameters(), cnv_layer.get_gradients())
    losses = []
    for epoch in range(int(epochs)):
        loss.clear()
        started_at = datetime.datetime.now().isoformat(timespec="seconds")
        start_time = time.perf_counter()
        progress = _progress(loader, "cnv%d distill %d/%d" % (index, epoch + 1, int(epochs)))
        for x_imgs, _teacher_full in progress:
            net.send_command("switch_model lut")
            x_buf = bb.FrameBuffer.from_numpy(np.asarray(x_imgs, dtype=np.float32))
            x_buf = net.real2bin.forward(x_buf, train=False)
            for i in range(index):
                x_buf = net.net_cnv[i].forward(x_buf, train=False)

            cnv_layer.send_command("switch_model dense")
            t_buf = cnv_layer.forward(x_buf, train=False)
            t_buf = bin2real0.forward(t_buf, train=False)

            cnv_layer.send_command("switch_model lut")
            y_buf = cnv_layer.forward(x_buf, train=True)
            y_buf = bin2real1.forward(y_buf, train=True)
            dy_buf = loss.calculate(y_buf, t_buf)
            dy_buf = bin2real1.backward(dy_buf)
            cnv_layer.backward(dy_buf)
            optimizer.update()
            _progress_postfix(progress, loss=_format_float(loss.get()))
        duration_seconds = time.perf_counter() - start_time
        finished_at = datetime.datetime.now().isoformat(timespec="seconds")
        losses.append(float(loss.get()))
        log_training_time(
            run_dir,
            stage="cnv%d" % int(index),
            epoch=epoch,
            phase="distillation",
            model_mode="distillation",
            started_at=started_at,
            finished_at=finished_at,
            duration_seconds=duration_seconds,
        )
        append_dict_row(
            os.path.join(run_dir, "logs", "distillation.csv"),
            {
                "layer": index,
                "epoch": epoch,
                "loss": _safe_float(loss.get()),
                "started_at": started_at,
                "finished_at": finished_at,
                "duration_seconds": _safe_float(duration_seconds),
            },
            ["layer", "epoch", "loss", "started_at", "finished_at", "duration_seconds"],
        )
    return losses


SUMMARY_FIELDS = [
    "model_name",
    "architecture_variant",
    "evaluation_crop_hw",
    "model_output_hw",
    "fpga_dw_pw_blocks",
    "run_seed",
    "dataset_seed",
    "dense_fg_miou",
    "lut_fg_miou",
    "dense_all_miou",
    "lut_all_miou",
    "dense_pixel_acc",
    "lut_pixel_acc",
    "fg_miou_drop_dense_to_lut",
    "all_miou_drop_dense_to_lut",
    "pixel_acc_drop_dense_to_lut",
    "logical_lut_count",
    "dense_train_seconds",
    "lut_distillation_seconds",
    "lut_tail_train_seconds",
    "lut_conversion_seconds",
    "total_training_seconds",
    "best_dense_epoch",
    "best_lut_epoch",
    "vivado_lut",
    "vivado_ff",
    "vivado_bram",
    "vivado_dsp",
    "wns",
    "fmax_mhz",
    "latency",
    "run_dir",
]


def collect_training_time_totals(run_dir):
    """Return update-only training-time totals recorded for one run.

    The result intentionally excludes validation/test evaluation, plotting,
    checkpoint writes, RTL export, and process startup.  It is therefore
    appropriate for comparing the amount of learning work between networks.
    """
    path = os.path.join(run_dir, "logs", "training_time.csv")
    totals = {
        "dense_train_seconds": 0.0,
        "lut_distillation_seconds": 0.0,
        "lut_tail_train_seconds": 0.0,
    }
    recorded_rows = 0
    if os.path.exists(path):
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                try:
                    seconds = float(row.get("duration_seconds", ""))
                except (TypeError, ValueError):
                    continue
                if not math.isfinite(seconds) or seconds < 0.0:
                    continue
                stage = str(row.get("stage", ""))
                phase = str(row.get("phase", ""))
                if stage == "dense" and phase == "train_update":
                    totals["dense_train_seconds"] += seconds
                    recorded_rows += 1
                elif stage.startswith("cnv") and phase == "distillation":
                    totals["lut_distillation_seconds"] += seconds
                    recorded_rows += 1
                elif stage.startswith("cnv") and phase == "train_update":
                    totals["lut_tail_train_seconds"] += seconds
                    recorded_rows += 1

    if recorded_rows == 0:
        return {
            "dense_train_seconds": "",
            "lut_distillation_seconds": "",
            "lut_tail_train_seconds": "",
            "lut_conversion_seconds": "",
            "total_training_seconds": "",
        }

    lut_conversion = (
        totals["lut_distillation_seconds"] + totals["lut_tail_train_seconds"]
    )
    total = totals["dense_train_seconds"] + lut_conversion
    return {
        "dense_train_seconds": _safe_float(totals["dense_train_seconds"]),
        "lut_distillation_seconds": _safe_float(totals["lut_distillation_seconds"]),
        "lut_tail_train_seconds": _safe_float(totals["lut_tail_train_seconds"]),
        "lut_conversion_seconds": _safe_float(lut_conversion),
        "total_training_seconds": _safe_float(total),
    }


def write_final_result_json(
    run_dir,
    dense_result,
    lut_result,
    dense_best,
    lut_best,
    evaluation_hw=(26, 26),
):
    write_json(
        os.path.join(run_dir, "final_results.json"),
        {
            "evaluation_crop_hw": [int(evaluation_hw[0]), int(evaluation_hw[1])],
            "dense_test": scores_to_jsonable(dense_result["scores"]) if dense_result else None,
            "lut_test": scores_to_jsonable(lut_result["scores"]) if lut_result else None,
            "dense_loss": dense_result.get("loss") if dense_result else None,
            "lut_loss": lut_result.get("loss") if lut_result else None,
            "best_dense": dense_best,
            "best_lut": lut_best,
            "training_time_seconds": collect_training_time_totals(run_dir),
            "note": TRAINING_EVAL_NOTE,
        },
    )


def append_summary(run_dir, config, architecture_summary, dense_result, lut_result, dense_best, lut_best):
    dense_scores = dense_result["scores"]
    lut_scores = lut_result["scores"]
    training_time = collect_training_time_totals(run_dir)
    row = {
        "model_name": config.model_name,
        "architecture_variant": config.architecture_variant,
        "evaluation_crop_hw": "%dx%d" % (int(config.eval_hw), int(config.eval_hw)),
        "model_output_hw": "%dx%d" % tuple(
            int(v) for v in architecture_summary["output_spatial_hw"]
        ),
        "fpga_dw_pw_blocks": config.fpga_dw_pw_blocks if config.model_name == "fpga_magazine" else "",
        "run_seed": config.run_seed,
        "dataset_seed": config.dataset_seed,
        "dense_fg_miou": _safe_float(dense_scores["foreground_miou"]),
        "lut_fg_miou": _safe_float(lut_scores["foreground_miou"]),
        "dense_all_miou": _safe_float(dense_scores["all_miou"]),
        "lut_all_miou": _safe_float(lut_scores["all_miou"]),
        "dense_pixel_acc": _safe_float(dense_scores["pixel_acc"]),
        "lut_pixel_acc": _safe_float(lut_scores["pixel_acc"]),
        "fg_miou_drop_dense_to_lut": _safe_float(dense_scores["foreground_miou"] - lut_scores["foreground_miou"]),
        "all_miou_drop_dense_to_lut": _safe_float(dense_scores["all_miou"] - lut_scores["all_miou"]),
        "pixel_acc_drop_dense_to_lut": _safe_float(dense_scores["pixel_acc"] - lut_scores["pixel_acc"]),
        "logical_lut_count": architecture_summary["logical_lut_count"],
        "dense_train_seconds": training_time["dense_train_seconds"],
        "lut_distillation_seconds": training_time["lut_distillation_seconds"],
        "lut_tail_train_seconds": training_time["lut_tail_train_seconds"],
        "lut_conversion_seconds": training_time["lut_conversion_seconds"],
        "total_training_seconds": training_time["total_training_seconds"],
        "best_dense_epoch": dense_best.get("best_epoch") if dense_best else "",
        "best_lut_epoch": lut_best.get("epoch") if lut_best else "",
        "vivado_lut": "",
        "vivado_ff": "",
        "vivado_bram": "",
        "vivado_dsp": "",
        "wns": "",
        "fmax_mhz": "",
        "latency": "",
        "run_dir": run_dir,
    }
    append_dict_row(os.path.join(run_dir, "summary.csv"), row, SUMMARY_FIELDS)
    append_dict_row(os.path.join(config.run_root, "summary.csv"), row, SUMMARY_FIELDS)
    if getattr(config, "plot_enabled", True):
        plot_final_test_dense_vs_lut(run_dir, config=config)


def _read_csv_rows(path):
    if not os.path.exists(path):
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _to_float_or_nan(value):
    if value is None or value == "":
        return np.nan
    try:
        return float(value)
    except ValueError:
        return np.nan


def _to_int_or_none(value):
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _cnv_index(stage):
    stage = str(stage)
    if stage.startswith("cnv") and stage[3:].isdigit():
        return int(stage[3:])
    return None


def _is_cnv_stage(stage):
    return _cnv_index(stage) is not None


def _get_pyplot():
    try:
        mpl_config_dir = os.path.join(tempfile.gettempdir(), "codex_matplotlib")
        os.makedirs(mpl_config_dir, exist_ok=True)
        os.environ.setdefault("MPLCONFIGDIR", mpl_config_dir)
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt

        return plt
    except Exception:
        return None


def _plot_line_series(output_path, series, *, title, xlabel, ylabel, ylim=None, xticks=None, xticklabels=None):
    plt = _get_pyplot()
    if plt is None:
        return
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.figure(figsize=(10, 5))
    plotted = False
    for label, points in series:
        points = [(x, y) for x, y in points if x is not None and not np.isnan(y)]
        if not points:
            continue
        points.sort(key=lambda v: v[0])
        xs, ys = zip(*points)
        plt.plot(xs, ys, marker="o", label=label)
        plotted = True
    if plotted:
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        if ylim is not None:
            plt.ylim(*ylim)
        if xticks is not None:
            plt.xticks(xticks, xticklabels if xticklabels is not None else xticks, rotation=45)
        plt.grid(True)
        plt.legend(loc="best")
        plt.tight_layout()
        plt.savefig(output_path)
    plt.close()


def _metric_points(rows, *, stage, phase, value_key):
    points = []
    for row in rows:
        if row.get("stage") != stage or row.get("phase") != phase:
            continue
        epoch = _to_int_or_none(row.get("epoch"))
        value = _to_float_or_nan(row.get(value_key))
        points.append((epoch, value))
    return points


def _last_metric_rows_by_cnv(rows, *, phase):
    selected = {}
    for row in rows:
        if row.get("phase") != phase:
            continue
        index = _cnv_index(row.get("stage"))
        epoch = _to_int_or_none(row.get("epoch"))
        if index is None or epoch is None:
            continue
        if index not in selected or epoch >= selected[index][0]:
            selected[index] = (epoch, row)
    return {index: row for index, (_epoch, row) in selected.items()}


def _last_class_rows_by_cnv(rows, *, phase):
    selected = {}
    for row in rows:
        if row.get("phase") != phase:
            continue
        index = _cnv_index(row.get("stage"))
        epoch = _to_int_or_none(row.get("epoch"))
        class_id = _to_int_or_none(row.get("class_id"))
        if index is None or epoch is None or class_id is None:
            continue
        key = (index, class_id)
        if key not in selected or epoch >= selected[key][0]:
            selected[key] = (epoch, row)
    by_class = {class_id: [] for class_id in range(len(CLASS_NAMES))}
    for (index, class_id), (_epoch, row) in selected.items():
        by_class.setdefault(class_id, []).append((index, _to_float_or_nan(row.get("iou"))))
    return by_class


def plot_dense_graphs(run_dir):
    metrics_rows = _read_csv_rows(os.path.join(run_dir, "logs", "metrics.csv"))
    class_rows = _read_csv_rows(os.path.join(run_dir, "logs", "class_iou.csv"))
    out_dir = os.path.join(run_dir, "graphs", "dense")

    _plot_line_series(
        os.path.join(out_dir, "train_update_loss_curve_dense.png"),
        [("train_update", _metric_points(metrics_rows, stage="dense", phase="train_update", value_key="train_update_loss"))],
        title="DenseAffine Train Update Loss",
        xlabel="epoch",
        ylabel="loss",
    )

    metric_specs = [
        ("loss", "loss", None),
        ("pixel_acc", "pixel accuracy", (0.0, 1.0)),
        ("all_miou", "all mIoU", (0.0, 1.0)),
        ("foreground_miou", "foreground mIoU", (0.0, 1.0)),
    ]
    for key, ylabel, ylim in metric_specs:
        _plot_line_series(
            os.path.join(out_dir, "%s_curve_dense.png" % key),
            [
                ("train_eval", _metric_points(metrics_rows, stage="dense", phase="train_eval", value_key=key)),
                ("validation", _metric_points(metrics_rows, stage="dense", phase="validation", value_key=key)),
            ],
            title="DenseAffine %s" % ylabel,
            xlabel="epoch",
            ylabel=ylabel,
            ylim=ylim,
        )

    for phase in ["train_eval", "validation"]:
        series = []
        for class_id, class_name in enumerate(CLASS_NAMES):
            points = []
            for row in class_rows:
                if row.get("stage") == "dense" and row.get("phase") == phase and _to_int_or_none(row.get("class_id")) == class_id:
                    points.append((_to_int_or_none(row.get("epoch")), _to_float_or_nan(row.get("iou"))))
            series.append((class_name, points))
        _plot_line_series(
            os.path.join(out_dir, "class_iou_curve_dense_%s.png" % phase),
            series,
            title="DenseAffine Class IoU (%s)" % phase,
            xlabel="epoch",
            ylabel="IoU",
            ylim=(0.0, 1.0),
        )


def plot_cnv_summary_graphs(run_dir):
    metrics_rows = _read_csv_rows(os.path.join(run_dir, "logs", "metrics.csv"))
    class_rows = _read_csv_rows(os.path.join(run_dir, "logs", "class_iou.csv"))
    out_dir = os.path.join(run_dir, "graphs", "cnv_summary")

    train_update = _last_metric_rows_by_cnv(metrics_rows, phase="train_update")
    cnv_indices = sorted(train_update.keys())
    if cnv_indices:
        _plot_line_series(
            os.path.join(out_dir, "train_update_loss_curve_cnv_all.png"),
            [("train_update", [(i, _to_float_or_nan(train_update[i].get("train_update_loss"))) for i in cnv_indices])],
            title="CNV Train Update Loss",
            xlabel="cnv stage",
            ylabel="loss",
            xticks=cnv_indices,
            xticklabels=["cnv%d" % i for i in cnv_indices],
        )

    metric_specs = [
        ("loss", "loss", None),
        ("pixel_acc", "pixel accuracy", (0.0, 1.0)),
        ("all_miou", "all mIoU", (0.0, 1.0)),
        ("foreground_miou", "foreground mIoU", (0.0, 1.0)),
    ]
    for key, ylabel, ylim in metric_specs:
        series = []
        all_indices = set()
        for phase in ["train_eval", "validation"]:
            by_cnv = _last_metric_rows_by_cnv(metrics_rows, phase=phase)
            indices = sorted(by_cnv.keys())
            all_indices.update(indices)
            series.append((phase, [(i, _to_float_or_nan(by_cnv[i].get(key))) for i in indices]))
        all_indices = sorted(all_indices)
        _plot_line_series(
            os.path.join(out_dir, "%s_curve_cnv_all.png" % key),
            series,
            title="CNV %s" % ylabel,
            xlabel="cnv stage",
            ylabel=ylabel,
            ylim=ylim,
            xticks=all_indices,
            xticklabels=["cnv%d" % i for i in all_indices],
        )

    for phase in ["train_eval", "validation"]:
        by_class = _last_class_rows_by_cnv(class_rows, phase=phase)
        all_indices = sorted({index for points in by_class.values() for index, _value in points})
        series = [(CLASS_NAMES[class_id], by_class.get(class_id, [])) for class_id in range(len(CLASS_NAMES))]
        _plot_line_series(
            os.path.join(out_dir, "class_iou_curve_cnv_all_%s.png" % phase),
            series,
            title="CNV Class IoU (%s)" % phase,
            xlabel="cnv stage",
            ylabel="IoU",
            ylim=(0.0, 1.0),
            xticks=all_indices,
            xticklabels=["cnv%d" % i for i in all_indices],
        )


def plot_class_iou_bar(run_dir, *, stage, epoch, filename):
    plt = _get_pyplot()
    if plt is None:
        return
    rows = _read_csv_rows(os.path.join(run_dir, "logs", "class_iou.csv"))
    values = {"train_eval": {}, "validation": {}}
    for row in rows:
        if row.get("stage") != stage or _to_int_or_none(row.get("epoch")) != int(epoch):
            continue
        phase = row.get("phase")
        if phase not in values:
            continue
        class_id = _to_int_or_none(row.get("class_id"))
        if class_id is not None:
            values[phase][class_id] = _to_float_or_nan(row.get("iou"))
    if not values["train_eval"] and not values["validation"]:
        return

    out_dir = os.path.join(run_dir, "graphs", "class_iou_bar")
    os.makedirs(out_dir, exist_ok=True)
    x = np.arange(len(CLASS_NAMES))
    width = 0.36
    plt.figure(figsize=(11, 5))
    plt.bar(x - width / 2, [values["train_eval"].get(i, np.nan) for i in range(len(CLASS_NAMES))], width, label="train_eval")
    plt.bar(x + width / 2, [values["validation"].get(i, np.nan) for i in range(len(CLASS_NAMES))], width, label="validation")
    plt.title("Class IoU %s epoch %03d" % (stage, int(epoch)))
    plt.xlabel("class")
    plt.ylabel("IoU")
    plt.ylim(0.0, 1.0)
    plt.xticks(x, CLASS_NAMES, rotation=30)
    plt.grid(True, axis="y")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, filename))
    plt.close()


def update_graphs_after_epoch(run_dir, *, stage, epoch, total_epochs, plot_enabled=True, plot_every=1, dense_class_iou_bar_every=10):
    if not plot_enabled:
        return
    is_last_epoch = int(epoch) == int(total_epochs) - 1
    should_update = is_last_epoch or (int(plot_every) > 0 and int(epoch) % int(plot_every) == 0)
    if not should_update:
        return
    if stage == "dense":
        plot_dense_graphs(run_dir)
        if int(dense_class_iou_bar_every) > 0:
            if is_last_epoch or int(epoch) % int(dense_class_iou_bar_every) == 0:
                plot_class_iou_bar(
                    run_dir,
                    stage="dense",
                    epoch=epoch,
                    filename="class_iou_bar_dense_epoch%03d.png" % int(epoch),
                )
    elif _is_cnv_stage(stage):
        plot_cnv_summary_graphs(run_dir)
        if is_last_epoch:
            plot_class_iou_bar(
                run_dir,
                stage=stage,
                epoch=epoch,
                filename="class_iou_bar_%s_final.png" % stage,
            )


def plot_final_test_dense_vs_lut(run_dir, config=None):
    plt = _get_pyplot()
    if plt is None:
        return
    rows = _read_csv_rows(os.path.join(run_dir, "logs", "class_iou.csv"))
    values = {"dense_final": {}, "lut_final": {}}
    for row in rows:
        if row.get("phase") != "test":
            continue
        stage = row.get("stage")
        if stage not in values:
            continue
        class_id = _to_int_or_none(row.get("class_id"))
        if class_id is not None:
            values[stage][class_id] = _to_float_or_nan(row.get("iou"))
    if not values["dense_final"] or not values["lut_final"]:
        return

    metrics_rows = _read_csv_rows(os.path.join(run_dir, "logs", "metrics.csv"))

    def metric(stage, key):
        for row in metrics_rows:
            if row.get("stage") == stage and row.get("phase") == "test":
                return _to_float_or_nan(row.get(key))
        return np.nan

    x = np.arange(len(CLASS_NAMES))
    width = 0.36
    plt.figure(figsize=(11, 5.5))
    plt.bar(x - width / 2, [values["dense_final"].get(i, np.nan) for i in range(len(CLASS_NAMES))], width, label="Dense")
    plt.bar(x + width / 2, [values["lut_final"].get(i, np.nan) for i in range(len(CLASS_NAMES))], width, label="LUT")
    title_prefix = "Final Test Class IoU"
    if config is not None:
        title_prefix = "%s %s %s" % (config.model_name, config.architecture_variant, config.run_id)
    plt.title(
        "%s\nDense FG=%s LUT FG=%s Dense Acc=%s LUT Acc=%s"
        % (
            title_prefix,
            _format_float(metric("dense_final", "foreground_miou")),
            _format_float(metric("lut_final", "foreground_miou")),
            _format_float(metric("dense_final", "pixel_acc")),
            _format_float(metric("lut_final", "pixel_acc")),
        )
    )
    plt.xlabel("class")
    plt.ylabel("IoU")
    plt.ylim(0.0, 1.0)
    plt.xticks(x, CLASS_NAMES, rotation=30)
    plt.grid(True, axis="y")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "graphs", "final_test_class_iou_dense_vs_lut.png"))
    plt.close()


def maybe_plot_metrics(run_dir):
    plot_dense_graphs(run_dir)
    plot_cnv_summary_graphs(run_dir)
    plot_final_test_dense_vs_lut(run_dir)


def write_layer_info(run_dir, net, net_name):
    path = os.path.join(run_dir, "%s_layer_info.txt" % slugify(net_name))
    with open(path, "w") as f:
        f.write(net.get_info())
    return path


def export_rtl_artifacts(bb, net, common_data, *, run_dir, module_name, copy_to_sim_path=None):
    rtl_dir = os.path.join(run_dir, "rtl")
    os.makedirs(rtl_dir, exist_ok=True)
    verilog_path = os.path.join(rtl_dir, "%s.v" % slugify(module_name))
    with open(verilog_path, "w") as f:
        f.write("`timescale 1ns / 1ps\n\n")
        bb.dump_verilog_lut_cnv_layers(f, "MnistSemanticSegmentation", net.net_cnv)
    if copy_to_sim_path:
        os.makedirs(copy_to_sim_path, exist_ok=True)
        shutil.copyfile(verilog_path, os.path.join(copy_to_sim_path, "%s.v" % slugify(module_name)))

    sources = common_data["source_imgs_test"]
    tile_h, tile_w = sources.shape[-2:]

    def make_source_tile(rows, cols):
        img = np.zeros((1, rows * tile_h, cols * tile_w), dtype=np.float32)
        k = 0
        for r in range(rows):
            for c in range(cols):
                img[:, r * tile_h:(r + 1) * tile_h, c * tile_w:(c + 1) * tile_w] = sources[k % len(sources)]
                k += 1
        return (img * 255).astype(np.uint8)

    img = make_source_tile(720 // tile_h + 1, 1280 // tile_w + 1)
    for height, width in [(120, 160), (480, 640), (720, 1280)]:
        bb.write_ppm(os.path.join(rtl_dir, "sd19_test_%dx%d.ppm" % (width, height)), img[:, :height, :width])
    return verilog_path


# Added coarse phase timings: no extra forward passes or batch conversions.
from functools import wraps
_TIMING_DIR = None

def configure_timing(run_dir):
    global _TIMING_DIR
    _TIMING_DIR = run_dir

def _timed_phase(function):
    @wraps(function)
    def measured(*args, **kwargs):
        started = time.perf_counter()
        try:
            return function(*args, **kwargs)
        finally:
            if _TIMING_DIR:
                path = os.path.join(_TIMING_DIR, "logs", "phase_timings.jsonl")
                os.makedirs(os.path.dirname(path), exist_ok=True)
                with open(path, "a") as out:
                    out.write(json.dumps({"phase": function.__name__, "seconds": time.perf_counter()-started})+"\n")
    return measured

for _phase in ("evaluate_segmentation", "_train_epoch_whole", "_train_epoch_tail",
               "distillation_layer", "update_graphs_after_epoch", "export_rtl_artifacts"):
    globals()[_phase] = _timed_phase(globals()[_phase])


def run_smoke_test(args, *, model_kind, create_network):
    from pathlib import Path
    from experiment_config import config_from_args
    from experiment_dataset import load_or_create_common_dataset, make_data_loaders
    config=config_from_args(args, model_name=model_kind, architecture_variant='samepadding_smoke', fpga_dw_pw_blocks=getattr(args, 'fpga_dw_pw_blocks', 0))
    bb=require_binarybrain();set_global_seed(config.run_seed)
    started=time.perf_counter()
    net,shape,dtype,layers,name=create_network(bb,config)
    assert tuple(shape)==(11,args.size,args.size)
    report=dict(model=model_kind,size=args.size,layers=layers,build_seconds=time.perf_counter()-started, modes={})
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
    report['train_loss']=_train_epoch_whole(bb,net,loaders['train'],output_shape=shape,model_kind=model_kind,optimizer=optimizer,desc='smoke only',evaluation_hw=(config.eval_hw,)*2)
    report['train_seconds']=time.perf_counter()-started
    result=evaluate_segmentation(bb,net,loaders['val'],output_shape=shape,model_kind=model_kind,title='smoke',evaluation_hw=(config.eval_hw,)*2)
    from experiment_metrics import scores_to_jsonable
    report['validation']=dict(loss=result['loss'],scores=scores_to_jsonable(result['scores']))
    Path(args.smoke_report or ('smoke_%s_%d.json' % (model_kind, args.size))).write_text(json.dumps(report,indent=2))


def run_camera_preview(args, *, model_kind, create_network):
    from pathlib import Path
    from PIL import Image
    from experiment_config import ExperimentConfig
    from experiment_dataset import load_or_create_common_dataset
    run = Path(args.camera_run_dir)
    saved=json.loads((run/'config.json').read_text())
    cfg=ExperimentConfig(**{k:v for k,v in saved.items() if k in ExperimentConfig.__dataclass_fields__})
    if cfg.model_name != model_kind:
        raise ValueError('Use the network driver matching the saved camera run')
    bb=require_binarybrain()
    net,_,_,_,_=create_network(bb,cfg)
    if not bb.load_networks(str(run/'checkpoints'),net,name=args.camera_checkpoint,verbose=False):
        raise FileNotFoundError(args.camera_checkpoint)
    net.send_command('switch_model '+('dense' if args.camera_checkpoint=='best_dense' else 'lut'))
    data=load_or_create_common_dataset(cfg)
    canvas=np.zeros((132,640),np.float32)
    truth=np.full((132,640),10,np.uint8)
    rng=np.random.default_rng(args.run_seed);s=cfg.input_h
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
    out=Path(args.camera_output);out.mkdir(parents=True,exist_ok=True)
    Image.fromarray(np.uint8(canvas*255)).save(out/'input.png')
    np.save(out/'prediction.npy',pred);np.save(out/'teacher.npy',truth)
    plt = _get_pyplot()
    if plt is None:
        raise RuntimeError('Matplotlib is required for the camera preview')
    fig,ax=plt.subplots(3,1,figsize=(16,9))
    ax[0].imshow(canvas,cmap='gray');ax[1].imshow(truth,cmap='tab20',vmin=0,vmax=10);ax[2].imshow(pred,cmap='tab20',vmin=0,vmax=10)
    fig.tight_layout();fig.savefig(out/'camera.png');plt.close(fig)
    (out/'metadata.json').write_text(json.dumps(dict(run_dir=str(run),checkpoint=args.camera_checkpoint,seed=args.run_seed,
        inference='nonoverlapping square patches; replicate edge; context seams',canvas_hw=[132,640],patch_size=s),indent=2))


def benchmark_data_loader(config, repeats=3):
    import statistics
    from experiment_dataset import load_or_create_common_dataset, make_data_loaders
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


def run_loader_benchmark(args, *, model_kind):
    from pathlib import Path
    from experiment_config import config_from_args
    config = config_from_args(args, model_name=model_kind,
        architecture_variant="samepadding_benchmark",
        fpga_dw_pw_blocks=getattr(args, "fpga_dw_pw_blocks", 0))
    result = benchmark_data_loader(config)
    Path(args.benchmark_report).write_text(json.dumps(result, indent=2))
    print("Loader-only benchmark: " + args.benchmark_report)
