#!/usr/bin/env python
# coding: utf-8

"""Architecture descriptions and logical LUT-count estimates."""

from experiment_config import (
    FPGA_MAGAZINE_DW_PW_BLOCKS,
    FPGA_MAGAZINE_FINAL_POINTWISE_LAYERS,
    G_RESEARCH_FINAL_POINTWISE_LAYERS,
    G_RESEARCH_SPATIAL_LAYERS,
    INPUT_H,
    INPUT_W,
)


def lut_conv1_count(output_ch):
    return int(output_ch)


def lut_conv2_count(output_ch):
    return int(output_ch) * 7


def lut_conv3_count(output_ch):
    return int(output_ch) * (6 * 6 + 6 + 1)


def expected_fpga_output_hw(dw_pw_blocks):
    return INPUT_H, INPUT_W


def expected_g_research_output_hw(
    spatial_layers=G_RESEARCH_SPATIAL_LAYERS
):
    return INPUT_H, INPUT_W

def fpga_magazine_stage_specs(dw_pw_blocks=FPGA_MAGAZINE_DW_PW_BLOCKS):
    dw_pw_blocks = int(dw_pw_blocks)
    if dw_pw_blocks <= 0:
        raise ValueError("fpga_dw_pw_blocks must be positive")

    specs = []
    for i in range(dw_pw_blocks):
        if i == 0:
            input_ch = 1
            output_ch = 32
            depthwise_lut_ch = 32
            role = "initial_1_to_32_dw_pw"
        elif i < dw_pw_blocks - 1:
            input_ch = 32
            output_ch = 32
            depthwise_lut_ch = 256
            role = "middle_32_to_32_dw_pw"
        else:
            input_ch = 32
            output_ch = 64
            depthwise_lut_ch = 256
            role = "final_32_to_64_dw_pw"
        pointwise_luts = lut_conv2_count(output_ch)
        specs.append({
            "stage": "cnv%d" % i,
            "role": role,
            "spatial_3x3": 1,
            "filter_size": "3x3 + 1x1",
            "connection": "random_depthwise_candidates + serial_pointwise",
            "input_channels": input_ch,
            "hidden_channels": depthwise_lut_ch,
            "output_channels": output_ch,
            "logical_lut_count": depthwise_lut_ch + pointwise_luts,
            "logical_lut_breakdown": {
                "depthwise_random_3x3_luts": depthwise_lut_ch,
                "pointwise_serial_1x1_luts": pointwise_luts,
            },
        })

    final_layers = [
        (dw_pw_blocks, "final_pointwise_to_36", 32 if dw_pw_blocks == 1 else 64, None, 36, lut_conv2_count(36)),
        (dw_pw_blocks + 1, "final_pointwise_36_to_11", 36, None, 11, lut_conv2_count(11)),
    ]
    for index, role, input_ch, hidden_ch, output_ch, luts in final_layers:
        specs.append({
            "stage": "cnv%d" % index,
            "role": role,
            "spatial_3x3": 0,
            "filter_size": "1x1",
            "connection": "serial_pointwise",
            "input_channels": input_ch,
            "hidden_channels": hidden_ch,
            "output_channels": output_ch,
            "logical_lut_count": luts,
            "logical_lut_breakdown": {"pointwise_serial_1x1_luts": luts},
        })
    return specs


def g_research_stage_specs():
    specs = []
    for i in range(G_RESEARCH_SPATIAL_LAYERS + G_RESEARCH_FINAL_POINTWISE_LAYERS):
        if i < G_RESEARCH_SPATIAL_LAYERS:
            if i == 0:
                role = "initial_random_3x3"
                input_ch = 1
                hidden_ch = None
                output_ch = 36
                luts = lut_conv1_count(output_ch)
                breakdown = {"random_3x3_luts": luts}
                connection = "random"
                filter_size = "3x3"
            elif i < 4:
                role = "pw_then_depthwise_36"
                input_ch = 36
                hidden_ch = 36
                output_ch = 36
                luts = lut_conv2_count(output_ch) + lut_conv1_count(output_ch)
                breakdown = {
                    "pointwise_serial_1x1_luts": lut_conv2_count(output_ch),
                    "depthwise_3x3_luts": lut_conv1_count(output_ch),
                }
                connection = "serial_pointwise + depthwise"
                filter_size = "1x1 + 3x3"
            elif i < 10:
                role = "mobilenet_hidden_36"
                input_ch = 36
                hidden_ch = 36
                output_ch = 36
                connection = "serial_pointwise + depthwise + serial_pointwise"
                filter_size = "1x1 + 3x3 + 1x1"
                luts = lut_conv2_count(hidden_ch) + lut_conv1_count(hidden_ch) + lut_conv2_count(output_ch)
                breakdown = {
                    "pw0_serial_1x1_luts": lut_conv2_count(hidden_ch),
                    "depthwise_3x3_luts": lut_conv1_count(hidden_ch),
                    "pw1_serial_1x1_luts": lut_conv2_count(output_ch),
                }
            elif i < 19:
                role = "mobilenet_hidden_72"
                input_ch = 36
                hidden_ch = 72
                output_ch = 36
                connection = "serial_pointwise + depthwise + serial_pointwise"
                filter_size = "1x1 + 3x3 + 1x1"
                luts = lut_conv2_count(hidden_ch) + lut_conv1_count(hidden_ch) + lut_conv2_count(output_ch)
                breakdown = {
                    "pw0_serial_1x1_luts": lut_conv2_count(hidden_ch),
                    "depthwise_3x3_luts": lut_conv1_count(hidden_ch),
                    "pw1_serial_1x1_luts": lut_conv2_count(output_ch),
                }
            elif i < 26:
                role = "mobilenet_hidden_144"
                input_ch = 36
                hidden_ch = 144
                output_ch = 36
                connection = "serial_pointwise + depthwise + serial_pointwise"
                filter_size = "1x1 + 3x3 + 1x1"
                luts = lut_conv2_count(hidden_ch) + lut_conv1_count(hidden_ch) + lut_conv2_count(output_ch)
                breakdown = {
                    "pw0_serial_1x1_luts": lut_conv2_count(hidden_ch),
                    "depthwise_3x3_luts": lut_conv1_count(hidden_ch),
                    "pw1_serial_1x1_luts": lut_conv2_count(output_ch),
                }
            else:
                role = "mobilenet_hidden_288"
                input_ch = 36
                hidden_ch = 288
                output_ch = 36
                connection = "serial_pointwise + depthwise + serial_pointwise"
                filter_size = "1x1 + 3x3 + 1x1"
                luts = lut_conv2_count(hidden_ch) + lut_conv1_count(hidden_ch) + lut_conv2_count(output_ch)
                breakdown = {
                    "pw0_serial_1x1_luts": lut_conv2_count(hidden_ch),
                    "depthwise_3x3_luts": lut_conv1_count(hidden_ch),
                    "pw1_serial_1x1_luts": lut_conv2_count(output_ch),
                }
            specs.append({
                "stage": "cnv%d" % i,
                "role": role,
                "spatial_3x3": 1,
                "filter_size": filter_size,
                "connection": connection,
                "input_channels": input_ch,
                "hidden_channels": hidden_ch,
                "output_channels": output_ch,
                "logical_lut_count": luts,
                "logical_lut_breakdown": breakdown,
            })
        elif i == G_RESEARCH_SPATIAL_LAYERS:
            luts = lut_conv2_count(72)
            specs.append({
                "stage": "cnv%d" % i,
                "role": "final_pointwise_36_to_72",
                "spatial_3x3": 0,
                "filter_size": "1x1",
                "connection": "serial_pointwise",
                "input_channels": 36,
                "hidden_channels": None,
                "output_channels": 72,
                "logical_lut_count": luts,
                "logical_lut_breakdown": {"pointwise_serial_1x1_luts": luts},
            })
        else:
            luts = lut_conv3_count(11)
            specs.append({
                "stage": "cnv%d" % i,
                "role": "final_pointwise_72_to_11",
                "spatial_3x3": 0,
                "filter_size": "1x1",
                "connection": "serial_pointwise_3_levels",
                "input_channels": 72,
                "hidden_channels": None,
                "output_channels": 11,
                "logical_lut_count": luts,
                "logical_lut_breakdown": {"pointwise_serial_1x1_3_level_luts": luts},
            })
    return specs


def summarize_architecture(model_name, architecture_variant, stage_specs, output_shape):
    total_luts = int(sum(stage["logical_lut_count"] for stage in stage_specs))
    return {
        "model_name": model_name,
        "architecture_variant": architecture_variant,
        "stage_count": len(stage_specs),
        "spatial_3x3_count": int(sum(stage["spatial_3x3"] for stage in stage_specs)),
        "logical_lut_count": total_luts,
        "logical_lut_note": "Logical LUT count is a design-level estimate, not Vivado post-synthesis LUT utilization.",
        "output_shape": list(output_shape),
        "output_spatial_hw": list(output_shape[-2:]),
        "stages": stage_specs,
        "vivado_resources": {
            "vivado_lut": None,
            "vivado_ff": None,
            "vivado_bram": None,
            "vivado_dsp": None,
            "wns": None,
            "fmax_mhz": None,
            "latency": None,
        },
    }
