#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""慢充稳定窗口内的电芯电压逐帧差分异常检测（delta / 相对差分法）。

诊断逻辑（以一次完整宽表 CSV 为处理单位，逐“稳定慢充窗口 × 电芯”输出）：

1. 按 ``charge_status`` 字段筛选慢充帧（区别于快充/行驶/静置），并在时间断档
   处强制切段。慢充帧还要求电流值有限，否则视为中断。
2. 在每个慢充段内滑动固定长度局部窗口（默认 10 分钟窗口、1 分钟步长，按采样
   间隔换算成帧数），仅对“电流稳定、且不在充电末端陡升区”的窗口继续分析。
   - 电流稳定同时要求：
       Q_95(I) - Q_5(I) < current_span_threshold_a
       max|ΔI|             < current_jump_threshold_a
     两个电流阈值按数据源标定，不因“慢充”标签就默认稳定。
   - 高 SOC 悬崖门控（电压/dVdt）：以跨芯电压中位数曲线（共同模式）为参考，跳过
     共同模式上升速率 > max_rise_rate_mv_per_min、或参考电压中位数 >
     max_cell_voltage_v 的窗口。陡升区会把电芯间正常的 SOC 差异放大成大的差分
     差异——既抬高 MAD 门槛造成漏检，又把提前进入陡升区的单芯误判成单向异常，
     故直接挡住。
   - 高 SOC 门控（SOC 字段，可选）：cvol_socmax 是“最高单体电压对应的 SOC”
     （百分数 0–100），能直接反映是否已有单芯进入充电末端陡升区——这是电压
     中位数门控抓不到的“单芯提前”情形。当窗口内 cvol_socmax 的 90 分位代表值
     > max_cell_soc_pct、或芯间 SOC 极差 > max_soc_spread_pct 时跳过。SOC 列
     缺失、阈值 <=0 时该门控自动关闭，退回电压/dVdt 门控。
3. 对每个窗口、每颗电芯计算相邻帧电压差分（统一换算到 mV）：
       d_i(t) = V_i(t) - V_i(t-1)
   不跨缺测/断档求差分，不预先平滑电压。
4. 减去同一帧的差分中位数，得到“该电芯比大多数电芯额外变化了多少”：
       m(t) = median_j d_j(t)
       e_i(t) = d_i(t) - m(t)
5. 用“绝对幅度 + 同帧 MAD”判定异常变化帧：
       MAD(t) = median_j |d_j(t) - m(t)|
       T(t)   = max(absolute_threshold_mv, mad_k * 1.4826 * MAD(t))
       hit_i(t) = 1 当 |e_i(t)| > T(t)，否则 0
   absolute_threshold_mv 保底排除量化台阶与正常小幅噪声，也避免 MAD 为零时除零。
6. 在窗口内统计该电芯的异常变化帧数 n_i 与占比 p_i = n_i / N_valid_i，判定候选：
       n_i >= min_anomaly_frames 且 p_i >= min_anomaly_ratio
   使用局部占比而非整次充电累计，避免充电时间越长越容易判异常的偏置。
7. 记录正、负方向异常帧数，用于区分波动形态（不作为硬门槛）：
       - 正、负超限各 >= min_bidirectional_each_side：反复双向波动候选
       - 达到候选但主要单方向：反复异常变化，需复核形态

该方法覆盖“反复出现、相邻采样点变化足够大”的局部异常；缓慢大幅摆动、极稀疏
毛刺与全组共同异常可能漏检。检出的结果应称为“电压波动候选”，不能仅凭它确认
采样硬件故障或电芯故障。全部阈值均为联调起点，需用正常/异常历史样本标定。

输入 CSV 为宽表（一文件一车辆/包），至少需要 ``timestamp``、``current_a``、
``charge_status`` 与两个 ``cell_数字`` 电压列。``charge_status`` 中代表慢充的
取值用 ``--slow-charge-status-values`` 指定（默认单个数值码 1）。

示例：
    python delta.py --input data/my_pack.csv
    python delta.py --input data/my_pack.csv --charge-status-column charge_status \
        --slow-charge-status-values 1 --voltage-unit V
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from voltage_fault_diagnosis import _time_gap_mask, _validated_timestamps

# 可选分组的列名，用于从宽表中提取 vehicle_id。
GROUP_COLUMNS = ("vehicle_id", "asset_id", "pack_id", "cluster_id")

WINDOW_COLUMNS = [
    "vehicle_id",
    "window_id",
    "start_time",
    "end_time",
    "n_frames",
    "n_diff_frames",
    "current_span_a",
    "current_max_jump_a",
    "current_stable",
    "ref_median_voltage_v",
    "rise_rate_mv_per_min",
    "soc_max_pct",
    "soc_spread_pct",
    "candidate_cell_count",
    "bidirectional_cell_count",
    "status",
]

RECORD_COLUMNS = [
    "vehicle_id",
    "window_id",
    "cell_id",
    "window_start_time",
    "window_end_time",
    "valid_diff_frames",
    "anomaly_frames",
    "anomaly_ratio",
    "positive_frames",
    "negative_frames",
    "max_abs_relative_diff_mv",
    "absolute_threshold_mv",
    "mad_k",
    "status",
    "is_anomaly",
]


@dataclass(frozen=True)
class DeltaConfig:
    """逐帧差分异常检测的配置。所有阈值均为联调起点，需按数据源标定。"""

    # 输入列与慢充取值
    charge_status_column: str = "charge_status"
    slow_charge_status_values: tuple = (1,)
    voltage_unit: str = "V"  # "V" 或 "mV"

    # 窗口（秒，按采样间隔换算成帧数）
    window_seconds: float = 600.0
    step_seconds: float = 60.0
    sample_interval_seconds: float | None = None
    gap_factor: float = 1.5

    # 电流稳定性门槛（A）
    current_span_threshold_a: float = 5.0
    current_jump_threshold_a: float = 3.0

    # 高 SOC 悬崖门控（电压/dVdt，挡住充电末端陡升区）
    max_rise_rate_mv_per_min: float | None = 5.0  # 共同模式上升速率上限；<=0 关闭
    max_cell_voltage_v: float | None = 3.45       # 参考电压中位数上限（V）；<=0 关闭

    # 高 SOC 门控（SOC 字段，抓“单芯提前进悬崖”；列缺失时自动退回电压门控）
    soc_max_column: str | None = "cvol_socmax"    # 最高单体电压对应 SOC 列名
    soc_min_column: str | None = "cvol_socmin"    # 最低单体电压对应 SOC 列名
    max_cell_soc_pct: float | None = 97.0         # 窗口内 socmax 90 分位上限；<=0 关闭
    max_soc_spread_pct: float | None = None       # 芯间 SOC 极差 90 分位上限；None/<=0 关闭

    # 差分异常判定
    absolute_threshold_mv: float = 5.0  # A_0，绝对差分地板
    mad_k: float = 5.0

    # 窗口内计数门槛
    min_valid_diff_frames: int = 30
    min_anomaly_frames: int = 4
    min_anomaly_ratio: float = 0.05
    min_bidirectional_each_side: int = 2

    # 数据质量
    min_valid_cells: int = 2
    min_valid_cell_fraction: float = 0.5
    drop_incomplete_frames: bool = True

    def __post_init__(self) -> None:
        if self.voltage_unit not in {"V", "mV"}:
            raise ValueError("voltage_unit 必须是 'V' 或 'mV'")

        values = tuple(self.slow_charge_status_values)
        object.__setattr__(self, "slow_charge_status_values", values)
        if len(values) == 0:
            raise ValueError("slow_charge_status_values 不能为空")
        for value in values:
            if isinstance(value, bool):
                raise ValueError("slow_charge_status_values 不能包含布尔值")
            if not isinstance(value, (str, int, float, np.integer, np.floating)):
                raise ValueError("slow_charge_status_values 只能是字符串或数字")

        for name in (
            "window_seconds",
            "step_seconds",
            "current_span_threshold_a",
            "current_jump_threshold_a",
            "absolute_threshold_mv",
            "mad_k",
            "gap_factor",
            "min_anomaly_ratio",
            "min_valid_cell_fraction",
        ):
            value = getattr(self, name)
            if not math.isfinite(value):
                raise ValueError(f"{name} 必须是有限数")
        for name in (
            "max_rise_rate_mv_per_min",
            "max_cell_voltage_v",
            "max_cell_soc_pct",
            "max_soc_spread_pct",
        ):
            value = getattr(self, name)
            if value is not None and not math.isfinite(value):
                raise ValueError(f"{name} 必须是有限数或 None")
        for name in ("soc_max_column", "soc_min_column"):
            value = getattr(self, name)
            if value is not None and not isinstance(value, str):
                raise ValueError(f"{name} 必须是字符串或 None")
            if value == "":
                object.__setattr__(self, name, None)
        if self.window_seconds <= 0:
            raise ValueError("window_seconds 必须 > 0")
        if self.step_seconds <= 0:
            raise ValueError("step_seconds 必须 > 0")
        if self.current_span_threshold_a < 0:
            raise ValueError("current_span_threshold_a 必须 >= 0")
        if self.current_jump_threshold_a < 0:
            raise ValueError("current_jump_threshold_a 必须 >= 0")
        if self.absolute_threshold_mv <= 0:
            raise ValueError("absolute_threshold_mv 必须 > 0")
        if self.mad_k < 0:
            raise ValueError("mad_k 必须 >= 0")
        if not 0 < self.min_anomaly_ratio <= 1:
            raise ValueError("min_anomaly_ratio 必须满足 0 < value <= 1")
        if not 0 < self.min_valid_cell_fraction <= 1:
            raise ValueError("min_valid_cell_fraction 必须满足 0 < value <= 1")
        if self.gap_factor <= 1:
            raise ValueError("gap_factor 必须 > 1")
        if self.sample_interval_seconds is not None:
            if (not math.isfinite(self.sample_interval_seconds)
                    or self.sample_interval_seconds <= 0):
                raise ValueError("sample_interval_seconds 必须为正有限数或 None")

        for name, lower in (
            ("min_valid_diff_frames", 1),
            ("min_anomaly_frames", 1),
            ("min_bidirectional_each_side", 1),
            ("min_valid_cells", 2),
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
                raise ValueError(f"{name} 必须是整数")
            if value < lower:
                raise ValueError(f"{name} 必须 >= {lower}")

        if not isinstance(self.drop_incomplete_frames, bool):
            raise ValueError("drop_incomplete_frames 必须是布尔值")


def _slow_charge_mask(status: Sequence, slow_values: tuple) -> np.ndarray:
    """返回 status 中取值命中 slow_values 的布尔掩码（慢充帧）。

    数字慢充码通过 ``pd.to_numeric`` 强转后比较，字符串码按文本比较；
    NaN/空值均不视为慢充。返回形状与 status 一致的一维布尔数组。
    """

    arr = np.asarray(status)
    flat = arr.ravel()
    mask = np.zeros(flat.shape, dtype=bool)
    text = flat.astype(str)
    for value in slow_values:
        if isinstance(value, str):
            mask |= text == value
        else:
            numeric = pd.to_numeric(
                pd.Series(flat), errors="coerce"
            ).to_numpy(dtype=float)
            mask |= np.isfinite(numeric) & np.isclose(numeric, float(value))
    return mask.reshape(arr.shape)


def _contiguous_runs(mask: np.ndarray, hard_gaps: np.ndarray) -> list[tuple[int, int]]:
    """把布尔掩码按时间断档切分成连续区间，返回 ``(start, stop_exclusive)``。"""

    runs: list[tuple[int, int]] = []
    start: int | None = None
    size = mask.size
    for index in range(size + 1):
        active = index < size and bool(mask[index])
        gap_break = index < size and bool(hard_gaps[index])
        if start is not None and (not active or gap_break):
            runs.append((start, index))
            start = None
        if active and start is None:
            start = index
    return runs


def _current_stability(window_current: np.ndarray) -> tuple[float, float, bool]:
    """返回 ``(Q95-Q5, max|ΔI|, 是否可计算)``；不足两有效点返回 nan。"""

    values = np.asarray(window_current, dtype=float)
    finite = np.isfinite(values)
    if int(finite.sum()) < 2:
        return math.nan, math.nan, False
    q05, q95 = np.quantile(values[finite], [0.05, 0.95], method="linear")
    span = float(q95 - q05)
    diffs = np.diff(values)
    diff_finite = finite[:-1] & finite[1:]
    max_jump = float(np.max(np.abs(diffs[diff_finite]))) if diff_finite.any() else 0.0
    return span, max_jump, True


def _window_cliff_metrics(
    window_voltage: np.ndarray,
    scale: float,
    interval: float,
) -> tuple[float, float]:
    """返回窗口的 ``(参考电压中位数 V, 共同模式上升速率 mV/min)``。

    以每帧跨芯电压中位数曲线（共同模式）为参考，与逐芯抖动无关：抖动电芯的
    瞬时差分很大但净斜率接近零，因此用中位数斜率判断“陡升区”不会误伤抖动本身。
    有效帧不足两帧时返回 ``(nan, nan)``。
    """

    ref = np.nanmedian(window_voltage, axis=1)  # 跨芯中位数曲线（输入单位）
    finite = np.isfinite(ref)
    if int(finite.sum()) < 2:
        return math.nan, math.nan
    ref_mv = ref * scale  # 统一换算到 mV
    ref_median_v = float(np.nanmedian(ref_mv)) / 1000.0
    diffs = np.diff(ref_mv)
    diff_valid = finite[:-1] & finite[1:]
    per_frame = float(np.median(diffs[diff_valid])) if diff_valid.any() else 0.0
    rise_rate = (
        per_frame * (60.0 / interval)
        if math.isfinite(interval) and interval > 0
        else per_frame
    )
    return ref_median_v, rise_rate


def _is_cliff_gated(
    ref_median_v: float,
    rise_rate: float,
    cfg: DeltaConfig,
) -> bool:
    """按上升速率 / 电压上限判断窗口是否处于充电末端陡升区，返回是否跳过。"""

    if cfg.max_rise_rate_mv_per_min is not None and cfg.max_rise_rate_mv_per_min > 0:
        if math.isfinite(rise_rate) and rise_rate > cfg.max_rise_rate_mv_per_min:
            return True
    if cfg.max_cell_voltage_v is not None and cfg.max_cell_voltage_v > 0:
        if math.isfinite(ref_median_v) and ref_median_v > cfg.max_cell_voltage_v:
            return True
    return False


def _window_soc_metrics(
    window_soc_max: np.ndarray,
    window_soc_min: np.ndarray | None,
) -> tuple[float, float]:
    """返回窗口的 ``(最高电压芯 SOC 代表值, 芯间 SOC 极差代表值)``（百分数）。

    ``cvol_socmax`` 是“最高单体电压对应的 SOC”。代表值取 90 分位而非中位数：
    既能抓到“窗口末端才进入陡升区”的边界窗口，又对单帧 OCV 校正跳变稳健。
    ``window_soc_min`` 为 None 时极差返回 nan。
    """

    soc_max = np.asarray(window_soc_max, dtype=float)
    finite_max = np.isfinite(soc_max)
    if finite_max.any():
        soc_max_ref = float(np.nanpercentile(soc_max[finite_max], 90.0))
    else:
        soc_max_ref = math.nan
    if window_soc_min is None:
        return soc_max_ref, math.nan
    soc_min = np.asarray(window_soc_min, dtype=float)
    spread = soc_max - soc_min
    finite_spread = np.isfinite(spread)
    if finite_spread.any():
        spread_ref = float(np.nanpercentile(spread[finite_spread], 90.0))
    else:
        spread_ref = math.nan
    return soc_max_ref, spread_ref


def _is_soc_gated(
    soc_max_ref: float,
    soc_spread_ref: float,
    cfg: DeltaConfig,
) -> bool:
    """按 cvol_socmax / 芯间 SOC 极差判断是否进入高 SOC 陡升区，返回是否跳过。"""

    if cfg.max_cell_soc_pct is not None and cfg.max_cell_soc_pct > 0:
        if math.isfinite(soc_max_ref) and soc_max_ref > cfg.max_cell_soc_pct:
            return True
    if cfg.max_soc_spread_pct is not None and cfg.max_soc_spread_pct > 0:
        if math.isfinite(soc_spread_ref) and soc_spread_ref > cfg.max_soc_spread_pct:
            return True
    return False


@dataclass
class DeltaResult:
    """逐“稳定慢充窗口 × 电芯”的诊断结果与元数据。"""

    vehicle_id: str
    cell_names: list[str]
    windows: pd.DataFrame
    records: pd.DataFrame
    config: DeltaConfig
    inferred_interval_seconds: float
    source_rows: int
    dropped_frames: int = 0
    slow_charge_rows: int = 0


def diagnose_voltage_delta(
    voltages: np.ndarray,
    currents: Sequence[float],
    status: Sequence,
    timestamps: Sequence,
    config: DeltaConfig,
    cell_names: Sequence[str] | None = None,
    vehicle_id: str = "",
    soc_max: Sequence[float] | None = None,
    soc_min: Sequence[float] | None = None,
) -> DeltaResult:
    """执行“慢充稳定窗口 → 逐帧差分 → 同帧 MAD + 绝对门槛 → 窗口计数”诊断。

    ``status`` 为与行数等长的 ``charge_status`` 取值序列（可为数字或字符串）。
    ``soc_max`` / ``soc_min`` 为可选的逐帧 SOC 列（百分数 0–100，如 cvol_socmax /
    cvol_socmin），长度与行数一致；任一为 None 或全无效时，SOC 门控自动关闭。
    返回 ``DeltaResult``，其中 ``windows`` 为每窗口一行，``records`` 为每个
    通过稳定性筛选的窗口内每颗电芯一行。
    """

    cfg = config
    values = np.asarray(voltages, dtype=float)
    current_values = np.asarray(currents, dtype=float)
    status_values = np.asarray(status)
    soc_max_values = np.asarray(soc_max, dtype=float) if soc_max is not None else None
    soc_min_values = np.asarray(soc_min, dtype=float) if soc_min is not None else None
    if values.ndim != 2 or values.shape[0] < 1 or values.shape[1] < 2:
        raise ValueError("voltages 必须为形状 (时间点, 电芯) 的二维数组，且至少两芯")
    rows, cells = values.shape
    if current_values.ndim != 1 or current_values.size != rows:
        raise ValueError("currents 必须是一维，且长度与电压行数一致")
    if status_values.ndim != 1 or status_values.size != rows:
        raise ValueError("status 必须是一维，且长度与电压行数一致")
    if soc_max_values is not None and (
        soc_max_values.ndim != 1 or soc_max_values.size != rows
    ):
        raise ValueError("soc_max 必须是一维，且长度与电压行数一致")
    if soc_min_values is not None and (
        soc_min_values.ndim != 1 or soc_min_values.size != rows
    ):
        raise ValueError("soc_min 必须是一维，且长度与电压行数一致")
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
            status_values = status_values[frame_valid]
            times = times[frame_valid]
            if soc_max_values is not None:
                soc_max_values = soc_max_values[frame_valid]
            if soc_min_values is not None:
                soc_min_values = soc_min_values[frame_valid]
            rows = values.shape[0]

    hard_gaps, interval = _time_gap_mask(
        times, cfg.sample_interval_seconds, cfg.gap_factor
    )
    slow_mask = _slow_charge_mask(status_values, cfg.slow_charge_status_values)
    slow_mask = slow_mask & np.isfinite(current_values)
    slow_charge_rows = int(slow_mask.sum())

    scale = 1000.0 if cfg.voltage_unit == "V" else 1.0
    min_ref_cells = max(
        cfg.min_valid_cells, math.ceil(cells * cfg.min_valid_cell_fraction)
    )

    window_frames = 0
    step_frames = 0
    if math.isfinite(interval):
        window_frames = max(2, int(round(cfg.window_seconds / interval)))
        step_frames = max(1, int(round(cfg.step_seconds / interval)))
    min_window_frames = cfg.min_valid_diff_frames + 1

    window_records: list[dict[str, object]] = []
    cell_records: list[dict[str, object]] = []
    window_id = 0

    for run_start, run_stop in _contiguous_runs(slow_mask, hard_gaps):
        if window_frames < 2:
            break
        start = run_start
        while True:
            stop = min(start + window_frames, run_stop)
            window_id += 1
            width = stop - start
            start_time = times[start].isoformat()
            end_time = times[stop - 1].isoformat()
            n_diff_frames = max(0, width - 1)

            window_current = current_values[start:stop]
            window_voltage = values[start:stop]
            span, max_jump, current_ok = _current_stability(window_current)
            current_stable = (
                current_ok
                and span < cfg.current_span_threshold_a
                and max_jump < cfg.current_jump_threshold_a
            )
            ref_median_v, rise_rate = _window_cliff_metrics(
                window_voltage, scale, interval
            )
            cliff_gated = _is_cliff_gated(ref_median_v, rise_rate, cfg)
            if soc_max_values is not None:
                soc_max_ref, soc_spread_ref = _window_soc_metrics(
                    soc_max_values[start:stop],
                    soc_min_values[start:stop]
                    if soc_min_values is not None
                    else None,
                )
                soc_gated = _is_soc_gated(soc_max_ref, soc_spread_ref, cfg)
            else:
                soc_max_ref, soc_spread_ref = math.nan, math.nan
                soc_gated = False

            window_row: dict[str, object] = {
                "vehicle_id": vehicle_id,
                "window_id": window_id,
                "start_time": start_time,
                "end_time": end_time,
                "n_frames": width,
                "n_diff_frames": n_diff_frames,
                "current_span_a": span,
                "current_max_jump_a": max_jump,
                "current_stable": current_stable,
                "ref_median_voltage_v": ref_median_v,
                "rise_rate_mv_per_min": rise_rate,
                "soc_max_pct": soc_max_ref,
                "soc_spread_pct": soc_spread_ref,
                "candidate_cell_count": 0,
                "bidirectional_cell_count": 0,
                "status": "",
            }

            if not current_stable:
                window_row["status"] = "current_unstable"
                window_records.append(window_row)
            elif width < min_window_frames:
                window_row["status"] = "insufficient_frames"
                window_records.append(window_row)
            elif cliff_gated:
                window_row["status"] = "voltage_rise_gated"
                window_records.append(window_row)
            elif soc_gated:
                window_row["status"] = "soc_gated"
                window_records.append(window_row)
            else:
                candidate, bidirectional, metrics = _score_window(
                    window_voltage,
                    scale,
                    min_ref_cells,
                    cfg,
                )
                window_row["status"] = "scored"
                window_row["candidate_cell_count"] = int(candidate.sum())
                window_row["bidirectional_cell_count"] = int(bidirectional.sum())
                window_records.append(window_row)

                for cell in range(cells):
                    valid_count = int(metrics["valid_diff_frames"][cell])
                    anomaly_count = int(metrics["anomaly_frames"][cell])
                    positive_count = int(metrics["positive_frames"][cell])
                    negative_count = int(metrics["negative_frames"][cell])
                    ratio = metrics["anomaly_ratio"][cell]
                    max_abs = metrics["max_abs_relative_diff_mv"][cell]

                    if valid_count < cfg.min_valid_diff_frames:
                        status_label = "insufficient_data"
                        is_anomaly: bool | None = None
                    elif bidirectional[cell]:
                        status_label = "bidirectional_fluctuation"
                        is_anomaly = True
                    elif candidate[cell]:
                        status_label = "directional_change"
                        is_anomaly = True
                    else:
                        status_label = "not_flagged"
                        is_anomaly = False

                    cell_records.append({
                        "vehicle_id": vehicle_id,
                        "window_id": window_id,
                        "cell_id": names[cell],
                        "window_start_time": start_time,
                        "window_end_time": end_time,
                        "valid_diff_frames": valid_count,
                        "anomaly_frames": anomaly_count,
                        "anomaly_ratio": ratio,
                        "positive_frames": positive_count,
                        "negative_frames": negative_count,
                        "max_abs_relative_diff_mv": max_abs,
                        "absolute_threshold_mv": cfg.absolute_threshold_mv,
                        "mad_k": cfg.mad_k,
                        "status": status_label,
                        "is_anomaly": is_anomaly,
                    })

            if stop >= run_stop:
                break
            start += step_frames

    windows = pd.DataFrame(window_records, columns=WINDOW_COLUMNS)
    records = pd.DataFrame(cell_records, columns=RECORD_COLUMNS)
    return DeltaResult(
        vehicle_id=vehicle_id,
        cell_names=names,
        windows=windows,
        records=records,
        config=cfg,
        inferred_interval_seconds=interval,
        source_rows=rows,
        dropped_frames=dropped_frames,
        slow_charge_rows=slow_charge_rows,
    )


def _score_window(
    window_voltage: np.ndarray,
    scale: float,
    min_ref_cells: int,
    cfg: DeltaConfig,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """对单个窗口电压矩阵计算逐帧差分异常，返回 ``(candidate, bidirectional, metrics)``。

    ``window_voltage`` 形状为 ``(帧数, 电芯数)``，单位由 ``scale`` 统一换算到 mV。
    """

    width, cells = window_voltage.shape
    diff = np.diff(window_voltage, axis=0) * scale  # (width-1, cells) mV
    diff_finite = np.isfinite(diff)
    ref_formed = diff_finite.sum(axis=1) >= min_ref_cells  # (width-1,)

    frame_median = np.full(width - 1, np.nan)
    frame_threshold = np.full(width - 1, np.nan)
    for t in range(width - 1):
        if not ref_formed[t]:
            continue
        row = diff[t]
        finite = diff_finite[t]
        med = float(np.median(row[finite]))
        frame_median[t] = med
        mad = float(np.median(np.abs(row[finite] - med)))
        frame_threshold[t] = max(
            cfg.absolute_threshold_mv, cfg.mad_k * 1.4826 * mad
        )

    relative = diff - frame_median[:, None]  # (width-1, cells)
    valid = diff_finite & ref_formed[:, None]
    hit = valid & (np.abs(relative) > frame_threshold[:, None])

    valid_diff_frames = valid.sum(axis=0).astype(int)
    anomaly_frames = hit.sum(axis=0).astype(int)
    positive_frames = (hit & (relative > 0)).sum(axis=0).astype(int)
    negative_frames = (hit & (relative < 0)).sum(axis=0).astype(int)
    anomaly_ratio = np.divide(
        anomaly_frames,
        valid_diff_frames,
        out=np.full(cells, np.nan),
        where=valid_diff_frames > 0,
    )

    with np.errstate(invalid="ignore"):
        max_abs = np.nanmax(np.where(valid, np.abs(relative), np.nan), axis=0)
    max_abs = np.where(valid_diff_frames > 0, max_abs, np.nan)

    candidate = (
        (valid_diff_frames >= cfg.min_valid_diff_frames)
        & (anomaly_frames >= cfg.min_anomaly_frames)
        & (anomaly_ratio >= cfg.min_anomaly_ratio)
    )
    bidirectional = (
        candidate
        & (positive_frames >= cfg.min_bidirectional_each_side)
        & (negative_frames >= cfg.min_bidirectional_each_side)
    )

    metrics = {
        "valid_diff_frames": valid_diff_frames,
        "anomaly_frames": anomaly_frames,
        "anomaly_ratio": anomaly_ratio,
        "positive_frames": positive_frames,
        "negative_frames": negative_frames,
        "max_abs_relative_diff_mv": max_abs,
    }
    return candidate, bidirectional, metrics


def _read_soc_column(frame: pd.DataFrame, column: str | None) -> np.ndarray | None:
    """读取可选的 SOC 列（百分数 0–100）；列缺失返回 None，越界值按无效置 NaN。"""

    if not column or column not in frame.columns:
        return None
    values = pd.to_numeric(frame[column], errors="coerce").to_numpy(float)
    values[(values < 0.0) | (values > 100.0)] = np.nan
    return values


def load_delta_csv(
    path: Path,
    current_column: str = "current_a",
    charge_status_column: str = "charge_status",
    soc_max_column: str | None = None,
    soc_min_column: str | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DatetimeIndex, list[str],
           np.ndarray | None, np.ndarray | None]:
    """读取单包宽表，返回电压、电流、charge_status、时间戳、电芯列名及可选 SOC 列。

    ``soc_max_column`` / ``soc_min_column``（如 cvol_socmax / cvol_socmin）为可选的
    逐帧 SOC 百分数（0–100）列；列缺失时对应返回 None。
    """

    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        header = next(csv.reader(stream), [])
    if not header:
        raise ValueError(f"{path.name}: CSV 为空")
    if len(header) != len(set(header)):
        raise ValueError(f"{path.name}: CSV 含重复列名")

    frame = pd.read_csv(path, encoding="utf-8-sig")
    required = {"timestamp", current_column, charge_status_column}
    missing = required.difference(frame.columns)
    if missing or frame.empty:
        detail = ", ".join(sorted(missing)) or "至少一行数据"
        raise ValueError(f"{path.name}: 缺少 {detail}")

    names = [name for name in frame.columns if re.fullmatch(r"cell_\d+", name)]
    names.sort(key=lambda name: int(name.split("_")[1]))
    numeric_ids = [int(name.split("_")[1]) for name in names]
    if len(names) < 2 or len(set(numeric_ids)) != len(numeric_ids):
        raise ValueError("至少需要两个编号不重复的电芯列，例如 cell_001、cell_002")

    for group_column in GROUP_COLUMNS:
        if (group_column in frame
                and frame[group_column].nunique(dropna=False) != 1):
            raise ValueError(f"{group_column} 含多个组，请先拆分为独立 CSV")

    try:
        voltages = frame[names].apply(pd.to_numeric, errors="raise").to_numpy(float)
        currents = pd.to_numeric(frame[current_column], errors="raise").to_numpy(float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"电压列和 {current_column} 必须为数值：{exc}") from exc

    status = frame[charge_status_column].to_numpy()
    times = _validated_timestamps(frame["timestamp"], len(frame))
    soc_max = _read_soc_column(frame, soc_max_column)
    soc_min = _read_soc_column(frame, soc_min_column)
    return voltages, currents, status, times, names, soc_max, soc_min


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


def save_delta_results(
    result: DeltaResult,
    output_dir: Path,
    input_path: Path,
) -> dict[str, object]:
    """保存逐窗口汇总、逐“窗口 × 电芯”记录长表与摘要 JSON。"""

    output_dir.mkdir(parents=True, exist_ok=True)
    result.windows.to_csv(
        output_dir / "windows.csv",
        index=False,
        float_format="%.10g",
        encoding="utf-8-sig",
    )
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

    if result.windows.empty:
        scored_windows = 0
        unstable_windows = 0
        insufficient_windows = 0
        voltage_gated_windows = 0
        soc_gated_windows = 0
    else:
        scored_windows = int((result.windows["status"] == "scored").sum())
        unstable_windows = int((result.windows["status"] == "current_unstable").sum())
        insufficient_windows = int(
            (result.windows["status"] == "insufficient_frames").sum()
        )
        voltage_gated_windows = int(
            (result.windows["status"] == "voltage_rise_gated").sum()
        )
        soc_gated_windows = int((result.windows["status"] == "soc_gated").sum())
    total_gated_windows = voltage_gated_windows + soc_gated_windows

    soc_available = False
    if not result.windows.empty and "soc_max_pct" in result.windows.columns:
        soc_available = bool(result.windows["soc_max_pct"].notna().any())

    warnings: list[str] = []
    if len(result.windows) == 0:
        warnings.append(
            "未提取到慢充窗口；请检查 charge_status 列、慢充取值和电流列。"
        )
    elif scored_windows == 0 and result.slow_charge_rows > 0:
        warnings.append(
            "慢充帧存在但没有任何窗口进入评分；请检查电流稳定性门槛与高 SOC 门控"
            "（max_rise_rate_mv_per_min / max_cell_voltage_v / max_cell_soc_pct）"
            "是否过严。"
        )
    if result.config.soc_max_column and not soc_available:
        warnings.append(
            f"SOC 门控未生效：列 {result.config.soc_max_column} 缺失或全为无效值，"
            "已退回电压/dVdt 门控。"
        )
    if total_gated_windows > 0 and total_gated_windows == len(result.windows):
        warnings.append(
            f"全部 {total_gated_windows} 个窗口都被高 SOC 门控挡掉；若数据不在充电"
            "末端，请检查 max_rise_rate_mv_per_min / max_cell_voltage_v / "
            "max_cell_soc_pct 是否过低。"
        )
    if anomaly_records == 0 and scored_windows > 0:
        warnings.append(
            "没有电芯达到异常帧数与占比门槛；本算法未标记候选。"
        )

    summary: dict[str, object] = {
        "method": (
            "slow-charge stable-window frame-difference + cross-cell median "
            "removal + same-frame MAD/absolute threshold + per-window count"
        ),
        "decision_rule": (
            "hit_i(t) = |d_i(t) - median_j d_j(t)| > "
            "max(absolute_threshold_mv, mad_k * 1.4826 * MAD(t)); "
            "candidate_i = anomaly_frames_i >= min_anomaly_frames "
            "and anomaly_ratio_i >= min_anomaly_ratio"
        ),
        "config": asdict(result.config),
        "input_file": input_path.name,
        "input_sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
        "rows": result.source_rows,
        "dropped_incomplete_frames": result.dropped_frames,
        "slow_charge_rows": result.slow_charge_rows,
        "cells": len(result.cell_names),
        "windows": len(result.windows),
        "scored_windows": scored_windows,
        "current_unstable_windows": unstable_windows,
        "insufficient_frames_windows": insufficient_windows,
        "voltage_rise_gated_windows": voltage_gated_windows,
        "soc_gated_windows": soc_gated_windows,
        "records": len(result.records),
        "anomaly_records": anomaly_records,
        "suspected_cells": suspected_cells,
        "status_counts": status_counts,
        "inferred_or_configured_sample_interval_seconds": (
            result.inferred_interval_seconds
            if math.isfinite(result.inferred_interval_seconds) else None
        ),
        "limitations": [
            "相对检测依赖多数电芯正常；全组共同异常会被中位数抵消。",
            "只覆盖相邻采样点变化足够大的反复局部异常；缓慢大幅摆动、极稀疏毛刺可能漏检。",
            "输出是统计异常提示（电压波动候选），不等同于采样硬件故障或电芯故障结论。",
            "所有阈值均为联调起点，需用正常/异常历史数据标定。",
        ],
        "warnings": warnings,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return summary


def _parse_status_values(text: str) -> tuple:
    """把逗号分隔的取值字符串解析成 ``(int/float/str, ...)`` 元组。"""

    result = []
    for token in text.split(","):
        token = token.strip()
        if not token:
            continue
        try:
            result.append(int(token))
        except ValueError:
            try:
                result.append(float(token))
            except ValueError:
                result.append(token)
    return tuple(result)


def main(argv: Sequence[str] | None = None) -> int:
    base = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--input", type=Path, default=base / "data",
        help="一个宽表 CSV，或包含 CSV 的目录",
    )
    parser.add_argument("--output-dir", type=Path, default=base / "delta_results")
    parser.add_argument("--current-column", default="current_a")
    parser.add_argument("--charge-status-column", default="charge_status")
    parser.add_argument(
        "--slow-charge-status-values", default="1",
        help="逗号分隔的慢充取值（数字或文本），默认 1",
    )
    parser.add_argument("--voltage-unit", choices=["V", "mV"], default="V")
    parser.add_argument("--window-seconds", type=float, default=600.0)
    parser.add_argument("--step-seconds", type=float, default=60.0)
    parser.add_argument("--sample-interval-seconds", type=float, default=None)
    parser.add_argument("--gap-factor", type=float, default=1.5)
    parser.add_argument("--current-span-threshold-a", type=float, default=5.0)
    parser.add_argument("--current-jump-threshold-a", type=float, default=3.0)
    parser.add_argument(
        "--max-rise-rate-mv-per-min", type=float, default=5.0,
        help="共同模式电压上升速率上限（mV/min），超过判为充电末端陡升区并跳过；<=0 关闭",
    )
    parser.add_argument(
        "--max-cell-voltage-v", type=float, default=3.45,
        help="参考电压中位数上限（V），超过则跳过；LFP 平台顶约 3.4V，非 LFP 请调整或设 0 关闭",
    )
    parser.add_argument(
        "--soc-max-column", default="cvol_socmax",
        help="最高单体电压对应 SOC 列名（百分数 0–100）；列缺失自动退回电压门控",
    )
    parser.add_argument(
        "--soc-min-column", default="cvol_socmin",
        help="最低单体电压对应 SOC 列名（百分数 0–100），用于计算芯间极差",
    )
    parser.add_argument(
        "--max-cell-soc-pct", type=float, default=97.0,
        help="窗口内 cvol_socmax 90 分位上限（百分数），超过判为高 SOC 陡升区并跳过；<=0 关闭",
    )
    parser.add_argument(
        "--max-soc-spread-pct", type=float, default=None,
        help="芯间 SOC 极差（socmax-socmin）90 分位上限（百分数）；None 或 <=0 关闭",
    )
    parser.add_argument("--absolute-threshold-mv", type=float, default=5.0)
    parser.add_argument("--mad-k", type=float, default=5.0)
    parser.add_argument("--min-valid-diff-frames", type=int, default=30)
    parser.add_argument("--min-anomaly-frames", type=int, default=4)
    parser.add_argument("--min-anomaly-ratio", type=float, default=0.05)
    parser.add_argument("--min-bidirectional-each-side", type=int, default=2)
    parser.add_argument("--min-valid-cells", type=int, default=2)
    parser.add_argument("--min-valid-cell-fraction", type=float, default=0.5)
    parser.add_argument(
        "--drop-incomplete-frames", action=argparse.BooleanOptionalAction, default=True,
        help="任一电芯电压为 null 的整帧丢弃",
    )
    args = parser.parse_args(argv)

    try:
        args.slow_charge_status_values = _parse_status_values(
            args.slow_charge_status_values
        )
        config = DeltaConfig(
            **{name: getattr(args, name) for name in DeltaConfig.__dataclass_fields__.keys()}
        )
        paths = sorted(args.input.glob("*.csv")) if args.input.is_dir() else [args.input]
        if not paths or not all(path.is_file() for path in paths):
            raise ValueError("没有找到输入 CSV")

        for input_path in paths:
            voltages, currents, status, timestamps, names, soc_max, soc_min = (
                load_delta_csv(
                    input_path,
                    args.current_column,
                    args.charge_status_column,
                    config.soc_max_column,
                    config.soc_min_column,
                )
            )
            vehicle_id = _read_group_id(input_path) or input_path.stem
            result = diagnose_voltage_delta(
                voltages, currents, status, timestamps, config, names, vehicle_id,
                soc_max=soc_max, soc_min=soc_min,
            )
            destination = args.output_dir / input_path.stem
            summary = save_delta_results(result, destination, input_path)
            print(
                f"{input_path.name}: windows={summary['windows']}, "
                f"scored={summary['scored_windows']}, "
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
