#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""慢充片段电芯异常波动检测：排列熵离群初筛 + 电压残差幅度确认。

诊断逻辑（以一次完整慢充片段为处理单位）：
0. （可选预处理）任一电芯电压为 null 的整帧丢弃，保证参考曲线各帧口径一致。
1. 沿用 `voltage_fault_diagnosis` 的电流方向筛选与完整充电片段分割。
2. 对每个片段、每颗电芯计算【加权排列熵】（方差加权，对 LFP 平台期的量化
   台阶/并列鲁棒；use_weighted_entropy=False 时退回普通归一化排列熵）；
   在同一片段内，以熵值有限电芯的中位数为基准，用 MAD 判定离群（初筛）：
       deviation = (pe - median) 或 abs(pe - median)
       E_i = [deviation > max(pe_mad_k * 1.4826 * MAD, pe_min_deviation)]
3. 对每个时刻，在所有有效电芯之间取电压中位数得到参考曲线 V_ref(t)；
   对通过初筛的电芯（或按配置对全部电芯）计算残差
       r_i(t) = V_i(t) - V_ref(t)
   统一换算到 mV 后，在整段有效残差点上取两个幅度量：
       A_i = Q_0.95(r_i) - Q_0.05(r_i)       （偏移/尖峰幅度）
       F_i = std(逐点差分 Δr_i)              （波动强度）
4. 最终判定 is_anomaly = E_i 且 (A_i > amplitude_threshold_mv 或
   F_i > fluctuation_threshold_mv)。fluctuation_threshold_mv 缺省为 None
   时退化为只按 A_i 确认，保持原行为。

该版本是给现有排列熵告警增加幅度确认，以减少小幅噪声误报；它仍可能漏掉
短时尖峰、低熵的规则振荡和全包共同异常，上线前需对比原算法的告警保留率、
误报和漏检情况。幅度/波动阈值需要正常/异常历史数据校准，本模块不提供
“已验证”的默认值。

示例：
    python voltage_amplitude_diagnosis.py --amplitude-threshold-mv 20
    python voltage_amplitude_diagnosis.py --input data/my_pack.csv \
        --voltage-unit V --amplitude-threshold-mv 20

输入 CSV 格式与 `voltage_fault_diagnosis.py` 一致（宽表，一文件一车辆/包）。
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from voltage_fault_diagnosis import (
    Config,
    _permutation_entropy_details,
    _validated_timestamps,
    _weighted_permutation_entropy_details,
    extract_charging_segments,
    load_voltage_csv,
)

# 可选分组的列名，用于从宽表中提取 vehicle_id。
GROUP_COLUMNS = ("vehicle_id", "asset_id", "pack_id", "cluster_id")

AMPLITUDE_COLUMNS = [
    "vehicle_id",
    "segment_id",
    "cell_id",
    "start_time",
    "end_time",
    "valid_point_count",
    "valid_time_ratio",
    "permutation_entropy",
    "median_permutation_entropy",
    "raw_mad_permutation_entropy",
    "pe_deviation",
    "pe_effective_threshold",
    "pe_candidate",
    "residual_q05_mv",
    "residual_q95_mv",
    "residual_amplitude_mv",
    "residual_fluctuation_mv",
    "residual_bias_mv",
    "amplitude_threshold_mv",
    "status",
    "is_anomaly",
]


@dataclass(frozen=True)
class AmplitudeConfig:
    """慢充片段排列熵离群初筛 + 残差幅度确认的配置。

    前两个字段无默认值，必须在构造时显式给出；`amplitude_threshold_mv`
    需要正常/异常历史数据校准，本模块不提供“已验证”的默认值。
    """

    # 新增，需校准
    amplitude_threshold_mv: float
    voltage_unit: str  # "V" 或 "mV"
    # 可选波动强度门槛（mV）；None 时退化为只按幅度确认，保持原行为
    fluctuation_threshold_mv: float | None = None
    # 用加权排列熵（方差加权）替代普通排列熵；对 LFP 平台期的量化台阶/并列鲁棒
    use_weighted_entropy: bool = True
    # 任一电芯电压为 null 的整帧丢弃（预处理），保证参考曲线各帧口径一致
    drop_incomplete_frames: bool = True

    # 排列熵离群初筛（pe_* 沿用现有排列熵的数值）
    pe_order: int = 3
    pe_delay: int = 1
    pe_direction: str = "two_sided"  # "high" 或 "two_sided"
    pe_mad_k: float = 3.0
    pe_min_deviation: float = 0.1
    tie_method: str = "stable"
    minimum_patterns: int = 30
    max_tie_fraction: float = 0.2

    # 充电片段提取（沿用现有值）
    charge_direction: str = "negative"
    min_charge_current_a: float = 0.0
    min_charge_samples: int = 32
    min_charge_fraction: float = 0.9
    max_interruption_samples: int = 0
    sample_interval_seconds: float | None = None
    gap_factor: float = 1.5

    # 数据质量（沿用现有值并补充幅度/参考曲线门槛）
    min_valid_cells: int = 2
    min_valid_cell_fraction: float = 0.8
    min_valid_points: int = 30
    min_valid_cell_ratio: float = 0.8
    min_valid_time_ratio: float = 0.8

    # 调试开关
    compute_amplitude_for_all_cells: bool = False

    def __post_init__(self) -> None:
        if not math.isfinite(self.amplitude_threshold_mv) or self.amplitude_threshold_mv <= 0:
            raise ValueError("amplitude_threshold_mv 必须是正有限数（mV）")
        if self.voltage_unit not in {"V", "mV"}:
            raise ValueError("voltage_unit 必须是 'V' 或 'mV'")
        if self.fluctuation_threshold_mv is not None and (
            not math.isfinite(self.fluctuation_threshold_mv)
            or self.fluctuation_threshold_mv <= 0
        ):
            raise ValueError("fluctuation_threshold_mv 必须是正有限数（mV）或 None")

        for name in ("use_weighted_entropy", "drop_incomplete_frames"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} 必须是布尔值")

        for name, lower in (
            ("pe_order", 2),
            ("pe_delay", 1),
            ("minimum_patterns", 1),
            ("min_valid_cells", 2),
            ("min_valid_points", 2),
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
                raise ValueError(f"{name} 必须是整数")
            if value < lower:
                raise ValueError(f"{name} 必须 >= {lower}")

        if self.pe_direction not in {"high", "two_sided"}:
            raise ValueError("pe_direction 必须是 'high' 或 'two_sided'")
        if self.tie_method not in {"stable", "skip"}:
            raise ValueError("tie_method 必须是 'stable' 或 'skip'")
        if (isinstance(self.max_interruption_samples, bool)
                or not isinstance(self.max_interruption_samples, (int, np.integer))
                or self.max_interruption_samples < 0):
            raise ValueError("max_interruption_samples 必须是非负整数")

        if not math.isfinite(self.pe_mad_k) or self.pe_mad_k < 0:
            raise ValueError("pe_mad_k 必须是非负有限数")
        if not math.isfinite(self.pe_min_deviation) or not 0 <= self.pe_min_deviation <= 1:
            raise ValueError("pe_min_deviation 必须满足 0 <= value <= 1")

        for name in (
            "min_charge_current_a",
            "max_tie_fraction",
            "gap_factor",
            "min_valid_cell_fraction",
            "min_charge_fraction",
            "min_valid_cell_ratio",
            "min_valid_time_ratio",
        ):
            value = getattr(self, name)
            if not math.isfinite(value):
                raise ValueError(f"{name} 必须是有限数")
        if self.min_charge_current_a < 0:
            raise ValueError("min_charge_current_a 必须 >= 0")
        for name in ("max_tie_fraction", "min_valid_cell_fraction",
                     "min_charge_fraction", "min_valid_cell_ratio",
                     "min_valid_time_ratio"):
            value = getattr(self, name)
            if not 0 <= value <= 1:
                raise ValueError(f"{name} 必须满足 0 <= value <= 1")
        for name in ("min_valid_cell_fraction", "min_charge_fraction",
                     "min_valid_cell_ratio", "min_valid_time_ratio"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} 必须 > 0")
        if self.gap_factor <= 1:
            raise ValueError("gap_factor 必须 > 1")
        if self.sample_interval_seconds is not None:
            if (not math.isfinite(self.sample_interval_seconds)
                    or self.sample_interval_seconds <= 0):
                raise ValueError("sample_interval_seconds 必须为正有限数或 None")

        required_samples = (
            (self.pe_order - 1) * self.pe_delay + self.minimum_patterns
        )
        if self.min_charge_samples < required_samples:
            raise ValueError(
                "min_charge_samples 太小：当前排列熵参数至少需要 "
                f"{required_samples} 个样本"
            )

    def legacy_config(self) -> Config:
        """构造等价的旧版 Config，仅供复用分割/排列熵原语使用。"""

        return Config(
            charge_direction=self.charge_direction,
            min_charge_current_a=self.min_charge_current_a,
            min_charge_samples=self.min_charge_samples,
            min_charge_fraction=self.min_charge_fraction,
            max_interruption_samples=self.max_interruption_samples,
            permutation_order=self.pe_order,
            permutation_delay=self.pe_delay,
            minimum_patterns=self.minimum_patterns,
            entropy_threshold=1.0,  # 本模块不调用旧的固定阈值判异，仅满足 Config 校验
            tie_method=self.tie_method,
            max_tie_fraction=self.max_tie_fraction,
            sample_interval_seconds=self.sample_interval_seconds,
            gap_factor=self.gap_factor,
            min_valid_cells=self.min_valid_cells,
            min_valid_cell_fraction=self.min_valid_cell_fraction,
        )


def permutation_entropy_mad_outliers(
    entropy: Sequence[float],
    reference_mask: Sequence[bool] | None,
    pe_mad_k: float,
    pe_min_deviation: float,
    pe_direction: str,
) -> tuple[float, float, float, np.ndarray, np.ndarray]:
    """按 MAD 判定排列熵离群，返回
    ``(median, raw_mad, effective_threshold, deviation, flags)``。

    参考集由 ``reference_mask`` 限定；所有具有有限熵值的候选电芯仍与参考
    中位数比较。有效阈值 ``max(pe_mad_k * 1.4826 * raw_mad, pe_min_deviation)``
    不除以 MAD，因此 ``raw_mad == 0`` 时不会除零，退化为 ``pe_min_deviation``
    地板。``deviation`` 在 ``high`` 方向为有符号差值，``two_sided`` 为绝对值。
    """

    values = np.asarray(entropy, dtype=float)
    if values.ndim != 1:
        raise ValueError("entropy 必须是一维序列")
    finite = np.isfinite(values)
    if reference_mask is None:
        reference = finite
    else:
        supplied = np.asarray(reference_mask, dtype=bool)
        if supplied.shape != values.shape:
            raise ValueError("reference_mask 必须与 entropy 形状一致")
        reference = finite & supplied

    deviation = np.full(values.shape, np.nan)
    flags = np.zeros(values.shape, dtype=bool)
    if not reference.any():
        return math.nan, math.nan, math.nan, deviation, flags

    median = float(np.median(values[reference]))
    raw_mad = float(np.median(np.abs(values[reference] - median)))
    effective_threshold = max(pe_mad_k * 1.4826 * raw_mad, pe_min_deviation)

    if pe_direction == "high":
        diff = values - median
    else:  # two_sided
        diff = np.abs(values - median)
    deviation[finite] = diff[finite]
    flags[finite] = diff[finite] > effective_threshold
    return median, raw_mad, effective_threshold, deviation, flags


def residual_amplitude_mv(
    residual_mv: Sequence[float],
    min_valid_points: int,
) -> tuple[float | None, float | None, float | None, float | None]:
    """对已换算到 mV 的带正负号残差计算 ``(q05, q95, amplitude, bias)``。

    直接对带符号残差取分位数，不先取绝对值；``amplitude = q95 - q05``，
    ``bias = median``（仅用于解释稳定偏高/偏低，不参与判定）。有效残差点
    不足 ``min_valid_points`` 时四个值均为 ``None``。
    """

    values = np.asarray(residual_mv, dtype=float)
    if values.ndim != 1:
        raise ValueError("residual_mv 必须是一维序列")
    finite = values[np.isfinite(values)]
    if finite.size < min_valid_points:
        return None, None, None, None
    q05, q95 = np.quantile(finite, [0.05, 0.95], method="linear")
    return (
        float(q05),
        float(q95),
        float(q95 - q05),
        float(np.median(finite)),
    )


def residual_fluctuation_mv(
    residual_mv: Sequence[float],
    min_valid_points: int,
) -> float | None:
    """对已换算到 mV 的残差计算逐点差分的标准差，作为“波动强度”。

    波动强度度量叠加在慢充斜坡上的高频抖动：先取差分 ``Δr[t] = r[t+1] - r[t]``，
    再对相邻有效的差分取标准差。它平移不变（恒定偏置不影响）、对平滑趋势不敏感
    （线性斜坡的差分近似常数），专门刻画采样噪声/接触不良造成的曲线波动。
    单个大尖峰对 std 的贡献被 1/sqrt(n) 摊薄，主要由幅度指标 A_i 负责捕捉。

    输入可以是含 NaN 的整段残差：只有相邻两点都有效时的差分才参与计算，
    断档两侧不会被误当作相邻样本。有效残差点不足 ``min_valid_points`` 时返回 None。
    """

    values = np.asarray(residual_mv, dtype=float)
    if values.ndim != 1:
        raise ValueError("residual_mv 必须是一维序列")
    valid = np.isfinite(values)
    if int(valid.sum()) < min_valid_points:
        return None
    diffs = np.diff(values)
    diff_valid = valid[:-1] & valid[1:]
    diffs = diffs[diff_valid]
    if diffs.size < 1:
        return None
    return float(np.std(diffs))


def _reference_curve(
    segment_voltages: np.ndarray,
    min_cells_for_reference: int,
) -> tuple[np.ndarray, np.ndarray]:
    """对每个时刻在所有有效电芯之间取电压中位数，返回 ``(ref, ref_valid)``。

    参考集合使用全部有效（有限）电芯，不剔除排列熵离群电芯。某时刻有效电芯
    不足 ``min_cells_for_reference`` 时，该时刻参考电压置为 NaN（无效）。
    """

    valid = np.isfinite(segment_voltages)
    valid_count = valid.sum(axis=1)
    ref_valid = valid_count >= min_cells_for_reference
    ref = np.full(segment_voltages.shape[0], np.nan)
    if ref_valid.any():
        ref[ref_valid] = np.nanmedian(segment_voltages[ref_valid], axis=1)
    return ref, ref_valid


def _make_record(
    vehicle_id: str,
    segment_id: int,
    cell_id: str,
    start_time: str,
    end_time: str,
    amplitude_threshold_mv: float,
    status: str,
    *,
    valid_point_count=math.nan,
    valid_time_ratio=math.nan,
    permutation_entropy=math.nan,
    median_permutation_entropy=math.nan,
    raw_mad_permutation_entropy=math.nan,
    pe_deviation=math.nan,
    pe_effective_threshold=math.nan,
    pe_candidate=None,
    residual_q05_mv=None,
    residual_q95_mv=None,
    residual_amplitude_mv=None,
    residual_fluctuation_mv=None,
    residual_bias_mv=None,
    is_anomaly=None,
) -> dict[str, object]:
    """按输出字段顺序构造一条“片段 × 电芯”记录；未计算字段留空。"""

    return {
        "vehicle_id": vehicle_id,
        "segment_id": segment_id,
        "cell_id": cell_id,
        "start_time": start_time,
        "end_time": end_time,
        "valid_point_count": valid_point_count,
        "valid_time_ratio": valid_time_ratio,
        "permutation_entropy": permutation_entropy,
        "median_permutation_entropy": median_permutation_entropy,
        "raw_mad_permutation_entropy": raw_mad_permutation_entropy,
        "pe_deviation": pe_deviation,
        "pe_effective_threshold": pe_effective_threshold,
        "pe_candidate": pe_candidate,
        "residual_q05_mv": residual_q05_mv,
        "residual_q95_mv": residual_q95_mv,
        "residual_amplitude_mv": residual_amplitude_mv,
        "residual_fluctuation_mv": residual_fluctuation_mv,
        "residual_bias_mv": residual_bias_mv,
        "amplitude_threshold_mv": amplitude_threshold_mv,
        "status": status,
        "is_anomaly": is_anomaly,
    }


@dataclass
class AmplitudeResult:
    """逐“片段 × 电芯”的诊断结果长表与元数据。"""

    vehicle_id: str
    cell_names: list[str]
    records: pd.DataFrame
    config: AmplitudeConfig
    inferred_interval_seconds: float
    source_rows: int
    dropped_frames: int = 0


def diagnose_voltage_amplitude(
    voltages: np.ndarray,
    currents: Sequence[float],
    timestamps: Sequence,
    config: AmplitudeConfig,
    cell_names: Sequence[str] | None = None,
    vehicle_id: str = "",
) -> AmplitudeResult:
    """执行“慢充片段 → 排列熵 MAD 离群初筛 → 残差幅度确认”诊断。

    返回 ``AmplitudeResult``，其中 ``records`` 为每个充电片段、每颗电芯一条
    记录的长表。电压单位由 ``config.voltage_unit`` 指定，幅度字段统一为 mV。
    """

    cfg = config
    values = np.asarray(voltages, dtype=float)
    current_values = np.asarray(currents, dtype=float)
    if values.ndim != 2 or values.shape[0] < 1 or values.shape[1] < 2:
        raise ValueError("voltages 必须为形状 (时间点, 电芯) 的二维数组，且至少两芯")
    rows, cells = values.shape
    if current_values.ndim != 1 or current_values.size != rows:
        raise ValueError("currents 必须是一维，且长度与电压行数一致")
    times = _validated_timestamps(timestamps, rows)
    names = (
        list(cell_names)
        if cell_names is not None
        else [f"cell_{index + 1:03d}" for index in range(cells)]
    )
    if len(names) != cells or len(set(names)) != cells:
        raise ValueError("cell_names 必须与电芯数一致且不重复")
    if cfg.min_valid_cells > cells:
        raise ValueError("min_valid_cells 不能大于电芯数量")

    dropped_frames = 0
    if cfg.drop_incomplete_frames:
        frame_valid = np.isfinite(values).all(axis=1)
        dropped_frames = int((~frame_valid).sum())
        if dropped_frames > 0:
            values = values[frame_valid]
            current_values = current_values[frame_valid]
            times = times[frame_valid]

    segments, interval = extract_charging_segments(
        current_values, times, cfg.legacy_config()
    )
    scale = 1000.0 if cfg.voltage_unit == "V" else 1.0
    min_cells_for_reference = max(
        cfg.min_valid_cells, math.ceil(cells * cfg.min_valid_cell_ratio)
    )
    required_reference_cells = max(
        cfg.min_valid_cells, math.ceil(cells * cfg.min_valid_cell_fraction)
    )

    records: list[dict[str, object]] = []
    for segment in segments:
        start, stop = segment.start_index, segment.stop_index
        n_samples = segment.sample_count
        start_time = times[start].isoformat()
        end_time = times[stop - 1].isoformat()
        charge_fraction = segment.qualified_charge_samples / n_samples
        segment_data_valid = (
            n_samples >= cfg.min_charge_samples
            and segment.qualified_charge_samples >= cfg.min_charge_samples
            and charge_fraction >= cfg.min_charge_fraction
        )

        if not segment_data_valid:
            for cell in range(cells):
                records.append(_make_record(
                    vehicle_id, segment.segment_id, names[cell],
                    start_time, end_time, cfg.amplitude_threshold_mv,
                    "insufficient_data",
                ))
            continue

        segment_voltages = values[start:stop]
        entropy = np.full(cells, np.nan)
        tie_fraction = np.full(cells, np.nan)
        for cell in range(cells):
            details = (
                _weighted_permutation_entropy_details
                if cfg.use_weighted_entropy
                else _permutation_entropy_details
            )
            value, tied, _ = details(
                segment_voltages[:, cell],
                cfg.pe_order,
                cfg.pe_delay,
                cfg.tie_method,
                cfg.minimum_patterns,
            )
            entropy[cell] = value
            tie_fraction[cell] = tied

        finite = np.isfinite(entropy)
        if cfg.use_weighted_entropy:
            # 加权熵对量化台阶/并列鲁棒，无需并列率质量门；恒定信号熵为 NaN 自然被排除。
            reference_eligible = finite
        else:
            tie_acceptable = np.isfinite(tie_fraction) & (tie_fraction <= cfg.max_tie_fraction)
            reference_eligible = finite & tie_acceptable
        reference_formed = reference_eligible.sum() >= required_reference_cells
        if reference_formed:
            median_pe, raw_mad, effective_threshold, deviation, pe_flags = (
                permutation_entropy_mad_outliers(
                    entropy,
                    reference_eligible,
                    cfg.pe_mad_k,
                    cfg.pe_min_deviation,
                    cfg.pe_direction,
                )
            )
        else:
            median_pe = raw_mad = effective_threshold = math.nan
            deviation = np.full(cells, np.nan)
            pe_flags = np.zeros(cells, dtype=bool)

        ref_curve, ref_valid = _reference_curve(
            segment_voltages, min_cells_for_reference
        )

        for cell in range(cells):
            cell_entropy = float(entropy[cell])
            if not math.isfinite(cell_entropy) or not reference_formed:
                records.append(_make_record(
                    vehicle_id, segment.segment_id, names[cell],
                    start_time, end_time, cfg.amplitude_threshold_mv,
                    "insufficient_data",
                    permutation_entropy=cell_entropy,
                    median_permutation_entropy=median_pe,
                    raw_mad_permutation_entropy=raw_mad,
                    pe_deviation=deviation[cell],
                    pe_effective_threshold=effective_threshold,
                    pe_candidate=False if math.isfinite(cell_entropy) else None,
                ))
                continue

            pe_candidate = bool(pe_flags[cell])
            cell_valid = ref_valid & np.isfinite(segment_voltages[:, cell])
            valid_points = int(cell_valid.sum())
            valid_time_ratio = valid_points / n_samples
            residual_data_valid = (
                valid_points >= cfg.min_valid_points
                and valid_time_ratio >= cfg.min_valid_time_ratio
            )
            q05 = q95 = amplitude = fluctuation = bias = None
            if (pe_candidate or cfg.compute_amplitude_for_all_cells) and residual_data_valid:
                residual_mv = (segment_voltages[:, cell] - ref_curve) * scale
                q05, q95, amplitude, bias = residual_amplitude_mv(
                    residual_mv, cfg.min_valid_points
                )
                fluctuation = residual_fluctuation_mv(
                    residual_mv, cfg.min_valid_points
                )

            amplitude_confirmed = (
                amplitude is not None and amplitude > cfg.amplitude_threshold_mv
            )
            fluctuation_confirmed = (
                cfg.fluctuation_threshold_mv is not None
                and fluctuation is not None
                and fluctuation > cfg.fluctuation_threshold_mv
            )
            if not pe_candidate:
                status = "not_flagged"
                is_anomaly = False
            elif not residual_data_valid:
                status = "insufficient_data"
                is_anomaly = None
            elif amplitude_confirmed or fluctuation_confirmed:
                status = "suspected_fluctuation"
                is_anomaly = True
            else:
                status = "pe_only"
                is_anomaly = False

            records.append(_make_record(
                vehicle_id, segment.segment_id, names[cell],
                start_time, end_time, cfg.amplitude_threshold_mv,
                status,
                valid_point_count=valid_points,
                valid_time_ratio=valid_time_ratio,
                permutation_entropy=cell_entropy,
                median_permutation_entropy=median_pe,
                raw_mad_permutation_entropy=raw_mad,
                pe_deviation=deviation[cell],
                pe_effective_threshold=effective_threshold,
                pe_candidate=pe_candidate,
                residual_q05_mv=q05,
                residual_q95_mv=q95,
                residual_amplitude_mv=amplitude,
                residual_fluctuation_mv=fluctuation,
                residual_bias_mv=bias,
                is_anomaly=is_anomaly,
            ))

    frame = pd.DataFrame(records, columns=AMPLITUDE_COLUMNS)
    return AmplitudeResult(
        vehicle_id=vehicle_id,
        cell_names=names,
        records=frame,
        config=cfg,
        inferred_interval_seconds=interval,
        source_rows=rows,
        dropped_frames=dropped_frames,
    )


def _read_group_id(path: Path) -> str | None:
    """从宽表中读取唯一分组值；无分组列时返回 None。"""

    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        header = next(csv.reader(stream), [])
    for column in GROUP_COLUMNS:
        if column not in header:
            continue
        frame = pd.read_csv(path, encoding="utf-8-sig", usecols=[column])
        unique = frame[column].dropna().astype(str).unique()
        if len(unique) == 0:
            return None
        if len(unique) != 1:
            raise ValueError(f"{path.name}: {column} 含多个组，请先拆分为独立 CSV")
        return str(unique[0])
    return None


def save_amplitude_results(
    result: AmplitudeResult,
    output_dir: Path,
    input_path: Path,
) -> dict[str, object]:
    """保存逐“片段 × 电芯”记录长表与摘要 JSON。"""

    output_dir.mkdir(parents=True, exist_ok=True)
    result.records.to_csv(
        output_dir / "records.csv",
        index=False,
        float_format="%.10g",
        encoding="utf-8-sig",
    )

    if result.records.empty:
        status_counts: dict[str, int] = {}
        anomaly_records = 0
        suspected_cells: list[str] = []
    else:
        status_counts = {
            str(key): int(value)
            for key, value in result.records["status"].value_counts().items()
        }
        anomaly_mask = result.records["is_anomaly"].fillna(False).astype(bool)
        anomaly_records = int(anomaly_mask.sum())
        suspected_cells = sorted(
            str(name)
            for name in result.records.loc[anomaly_mask, "cell_id"].unique()
        )

    warnings: list[str] = []
    if len(result.records) == 0:
        warnings.append("未提取到充电片段；请检查电流列、充电电流符号和电流阈值。")
    elif anomaly_records == 0 and status_counts.get("insufficient_data", 0) == 0:
        warnings.append("没有电芯同时满足排列熵离群与幅度条件；本算法未标记异常。")

    confirm_rule = (
        f"amplitude (Q95-Q05 of residual) > {result.config.amplitude_threshold_mv:g} mV"
    )
    if result.config.fluctuation_threshold_mv is not None:
        confirm_rule += (
            " OR fluctuation (std of residual successive diffs) > "
            f"{result.config.fluctuation_threshold_mv:g} mV"
        )

    entropy_metric = (
        "weighted permutation entropy (variance-weighted)"
        if result.config.use_weighted_entropy
        else "normalized permutation entropy"
    )
    summary: dict[str, object] = {
        "method": (
            "slow-charge segment permutation-entropy MAD outlier pre-screen "
            "+ cross-cell voltage residual amplitude confirmation"
        ),
        "entropy_metric": entropy_metric,
        "decision_rule": (
            "PE outlier E_i: deviation_i > max(pe_mad_k * 1.4826 * MAD, "
            f"pe_min_deviation); confirm F_i: {confirm_rule}; "
            "is_anomaly = E_i AND F_i"
        ),
        "voltage_unit": result.config.voltage_unit,
        "amplitude_threshold_mv": result.config.amplitude_threshold_mv,
        "config": asdict(result.config),
        "input_file": input_path.name,
        "input_sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
        "rows": result.source_rows,
        "dropped_incomplete_frames": result.dropped_frames,
        "cells": len(result.cell_names),
        "charging_segments": (
            0 if result.records.empty
            else int(result.records["segment_id"].nunique())
        ),
        "records": len(result.records),
        "anomaly_records": anomaly_records,
        "suspected_cells": suspected_cells,
        "status_counts": status_counts,
        "inferred_or_configured_sample_interval_seconds": (
            result.inferred_interval_seconds
            if math.isfinite(result.inferred_interval_seconds) else None
        ),
        "limitations": [
            "相对检测依赖多数电芯正常；全包共同异常会被中位数抵消。",
            "幅度门槛可能降低召回率；上线前需对比原算法的告警保留率、误报和漏检。",
            "整段只产生一次诊断，不能定位片段内部异常起始时刻。",
            "输出是统计异常提示，不等同于故障根因或安全结论。",
        ],
        "warnings": warnings,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    base = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--input", type=Path, default=base / "data",
        help="一个宽表 CSV，或包含 CSV 的目录",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=base / "amplitude_results"
    )
    parser.add_argument("--current-column", default="current_a")
    parser.add_argument("--voltage-unit", choices=["V", "mV"], default="V")
    parser.add_argument(
        "--amplitude-threshold-mv", type=float, required=True,
        help="残差幅度门槛（mV）；需用正常/异常历史数据校准，无默认值",
    )
    parser.add_argument(
        "--fluctuation-threshold-mv", type=float, default=None,
        help="残差波动强度门槛（mV，逐点差分 std）；缺省时只按幅度确认",
    )
    parser.add_argument(
        "--use-weighted-entropy", action=argparse.BooleanOptionalAction, default=True,
        help="用加权排列熵（方差加权）替代普通排列熵；对 LFP 平台期鲁棒",
    )
    parser.add_argument(
        "--drop-incomplete-frames", action=argparse.BooleanOptionalAction, default=True,
        help="任一电芯电压为 null 的整帧丢弃",
    )
    parser.add_argument("--pe-order", type=int, default=3)
    parser.add_argument("--pe-delay", type=int, default=1)
    parser.add_argument(
        "--pe-direction", choices=["high", "two_sided"], default="two_sided"
    )
    parser.add_argument("--pe-mad-k", type=float, default=3.0)
    parser.add_argument("--pe-min-deviation", type=float, default=0.1)
    parser.add_argument("--tie-method", choices=["stable", "skip"], default="stable")
    parser.add_argument("--minimum-patterns", type=int, default=30)
    parser.add_argument("--max-tie-fraction", type=float, default=0.2)
    parser.add_argument(
        "--charge-direction", choices=["negative", "positive"], default="negative"
    )
    parser.add_argument("--min-charge-current-a", type=float, default=0.0)
    parser.add_argument("--min-charge-samples", type=int, default=32)
    parser.add_argument("--min-charge-fraction", type=float, default=0.9)
    parser.add_argument("--max-interruption-samples", type=int, default=0)
    parser.add_argument("--sample-interval-seconds", type=float, default=None)
    parser.add_argument("--gap-factor", type=float, default=1.5)
    parser.add_argument("--min-valid-cells", type=int, default=2)
    parser.add_argument("--min-valid-cell-fraction", type=float, default=0.8)
    parser.add_argument("--min-valid-points", type=int, default=30)
    parser.add_argument("--min-valid-cell-ratio", type=float, default=0.8)
    parser.add_argument("--min-valid-time-ratio", type=float, default=0.8)
    parser.add_argument(
        "--compute-amplitude-for-all-cells", action="store_true",
        help="对全部有效电芯计算残差幅度（仅调试，不影响判定逻辑）",
    )
    args = parser.parse_args(argv)

    try:
        config = AmplitudeConfig(
            **{name: getattr(args, name)
               for name in AmplitudeConfig.__dataclass_fields__.keys()}
        )
        paths = sorted(args.input.glob("*.csv")) if args.input.is_dir() else [args.input]
        if not paths or not all(path.is_file() for path in paths):
            raise ValueError("没有找到输入 CSV")

        for input_path in paths:
            voltages, currents, timestamps, names = load_voltage_csv(
                input_path, args.current_column
            )
            vehicle_id = _read_group_id(input_path) or input_path.stem
            result = diagnose_voltage_amplitude(
                voltages, currents, timestamps, config, names, vehicle_id
            )
            destination = args.output_dir / input_path.stem
            summary = save_amplitude_results(result, destination, input_path)
            print(
                f"{input_path.name}: segments={summary['charging_segments']}, "
                f"records={summary['records']}, "
                f"anomaly_records={summary['anomaly_records']}, "
                f"suspected_cells={summary['suspected_cells']}"
            )
            print(f"  output: {destination}")
            for warning in summary["warnings"]:
                print(f"  warning: {warning}", file=sys.stderr)
        return 0
    except (ValueError, OSError, TypeError, OverflowError) as exc:
        parser.exit(2, f"error: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
