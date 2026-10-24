#!/usr/bin/env python
# coding: utf-8

"""Shared crop and segmentation metric utilities."""

import math
import numpy as np

from experiment_config import BACKGROUND_CLASS, CLASS_NAMES, NUM_CLASSES


# Kept as the public default for backward compatibility.  Callers may pass
# eval_hw=(24, 24) to evaluate only the central 24x24 region.
COMMON_EVAL_HW = (26, 26)


def _normalize_eval_hw(eval_hw=None):
    """Return a validated (height, width) evaluation crop size."""
    if eval_hw is None:
        return COMMON_EVAL_HW
    if isinstance(eval_hw, (int, np.integer)):
        eval_hw = (int(eval_hw), int(eval_hw))
    target_h, target_w = _as_hw(eval_hw)
    if target_h <= 0 or target_w <= 0:
        raise ValueError("eval_hw must be positive, got %s" % ((target_h, target_w),))
    return target_h, target_w


def _as_hw(output_shape):
    if output_shape is None:
        raise ValueError("output_shape is required")
    shape = tuple(int(v) for v in output_shape)
    if len(shape) == 2:
        return shape
    if len(shape) == 3:
        return shape[-2], shape[-1]
    if len(shape) == 4:
        return shape[-2], shape[-1]
    raise ValueError("unexpected output_shape: %r" % (output_shape,))


def _center_crop_4d(array, target_hw, *, name):
    arr = np.asarray(array)
    if arr.ndim != 4:
        raise ValueError("%s must be [N, C, H, W], got shape %r" % (name, arr.shape))
    target_h, target_w = _as_hw(target_hw)
    h, w = arr.shape[-2], arr.shape[-1]
    if target_h > h or target_w > w:
        raise ValueError("%s target %s is larger than source HW %s" % (name, (target_h, target_w), (h, w)))
    crop_top = (h - target_h) // 2
    crop_left = (w - target_w) // 2
    if crop_top * 2 + target_h != h or crop_left * 2 + target_w != w:
        raise ValueError("%s cannot be center-cropped evenly from %s to %s" % (name, (h, w), (target_h, target_w)))
    if crop_top == 0 and crop_left == 0:
        return arr
    return arr[:, :, crop_top:crop_top + target_h, crop_left:crop_left + target_w]


def crop_teacher_for_common_eval(teacher_full, eval_hw=COMMON_EVAL_HW):
    """Crop full84 teacher labels to the configured central evaluation region."""
    return _center_crop_4d(
        teacher_full,
        _normalize_eval_hw(eval_hw),
        name="teacher_full",
    )


def crop_teacher_for_model_output(teacher_full, output_shape):
    """Crop full84 teacher labels to the exact model output height and width."""
    output_hw = _as_hw(output_shape)
    return _center_crop_4d(teacher_full, output_hw, name="teacher_full")


def crop_prediction_for_common_eval(y_np, model_kind, output_shape=None, eval_hw=COMMON_EVAL_HW):
    """Crop predictions to the configured central evaluation region.

    The crop is derived from the actual output shape.  For the FPGA magazine
    variants this covers 30/28/26 outputs; for g_research the native output is
    26x26.  Passing eval_hw=(24, 24) removes one border pixel on every side
    from a 26x26 prediction without changing the model itself.
    """
    y = np.asarray(y_np)
    if y.ndim != 4:
        raise ValueError("prediction must be [N, C, H, W], got shape %r" % (y.shape,))
    if output_shape is not None and _as_hw(output_shape) != y.shape[-2:]:
        raise ValueError(
            "prediction HW %s does not match declared output_shape %s" %
            (y.shape[-2:], _as_hw(output_shape))
        )
    return _center_crop_4d(y, _normalize_eval_hw(eval_hw), name="prediction")


def _safe_nanmean(values):
    values = np.asarray(values, dtype=np.float64)
    if np.all(np.isnan(values)):
        return np.nan
    return float(np.nanmean(values))


class SegmentationMetrics:
    """mIoU, foreground mIoU, class IoU, pixel accuracy, and valid pixels."""

    def __init__(self, num_classes=NUM_CLASSES, background_index=BACKGROUND_CLASS, class_names=None):
        self.num_classes = int(num_classes)
        self.background_index = int(background_index)
        self.class_names = class_names if class_names is not None else CLASS_NAMES
        self.clear()

    def clear(self):
        self.confusion = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)

    def calculate(self, y_buf, t_buf):
        self.calculate_numpy(y_buf.numpy(), t_buf.numpy())

    def calculate_numpy(self, pred_np, teacher_np):
        pred_arr = np.asarray(pred_np)
        teacher_arr = np.asarray(teacher_np)
        if pred_arr.shape != teacher_arr.shape:
            raise ValueError("prediction/teacher shape mismatch: %r vs %r" % (pred_arr.shape, teacher_arr.shape))
        if pred_arr.ndim != 4:
            raise ValueError("prediction/teacher must be [N, C, H, W], got %r" % (pred_arr.shape,))
        if pred_arr.shape[1] != self.num_classes:
            raise ValueError("channel count %d does not match num_classes %d" % (pred_arr.shape[1], self.num_classes))

        valid_teacher = np.any(teacher_arr > 0, axis=1).reshape(-1)
        pred = np.argmax(pred_arr, axis=1).reshape(-1)
        true = np.argmax(teacher_arr, axis=1).reshape(-1)
        valid = (
            valid_teacher
            & (true >= 0) & (true < self.num_classes)
            & (pred >= 0) & (pred < self.num_classes)
        )
        true = true[valid]
        pred = pred[valid]
        if true.size > 0:
            np.add.at(self.confusion, (true, pred), 1)

    def get_scores(self):
        hist = self.confusion.astype(np.float64)
        tp = np.diag(hist)
        gt_area = hist.sum(axis=1)
        pred_area = hist.sum(axis=0)
        union = gt_area + pred_area - tp
        class_iou = np.divide(
            tp,
            union,
            out=np.full(self.num_classes, np.nan, dtype=np.float64),
            where=union > 0,
        )
        foreground_indices = [i for i in range(self.num_classes) if i != self.background_index]
        valid_pixel_count = int(hist.sum())
        pixel_acc = np.nan
        if valid_pixel_count > 0:
            pixel_acc = float(tp.sum() / valid_pixel_count)
        all_miou = _safe_nanmean(class_iou)
        foreground_miou = _safe_nanmean(class_iou[foreground_indices])
        return {
            "all_miou": all_miou,
            "miou": all_miou,
            "foreground_miou": foreground_miou,
            "pixel_acc": pixel_acc,
            "valid_pixel_count": valid_pixel_count,
            "class_iou": class_iou,
        }

    def get(self):
        return self.get_scores()["foreground_miou"]


def scores_to_jsonable(scores):
    out = {}
    for key, value in scores.items():
        if key == "class_iou":
            out[key] = [
                None if (isinstance(v, float) and math.isnan(v)) else float(v)
                for v in np.asarray(value, dtype=np.float64)
            ]
        elif isinstance(value, (np.integer, int)):
            out[key] = int(value)
        elif value is None:
            out[key] = None
        else:
            value = float(value)
            out[key] = None if math.isnan(value) else value
    return out
