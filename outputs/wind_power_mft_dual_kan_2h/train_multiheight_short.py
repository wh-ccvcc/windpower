"""Use 15-minute history and direct short-lead power targets on native observations."""

import argparse
from dataclasses import replace
from datetime import datetime
from hashlib import sha256
from pathlib import Path
from statistics import mean
import sys

import numpy as np
import pandas as pd
import torch

import io_utils
from mft_dual import DualMFTForecaster
from training import TrainConfig, fit_curve


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parents[1] / "windpower_cwt_server_20261113" / "full_model"))
from baselines.models.transformer import Transformer


NATIVE_MINUTES = 5
SAMPLE_MINUTES = 15
HISTORY_STEPS = 96
LEADS = (20,)
SCALES = (1, 2, 4, 8)
FEATURES = ("power_mw", *(f"wind_speed_{height}m_ms" for height in (90, 70, 30, 10)))
TRANSFORMER_CHANNELS = (0, 2)
NATIVE_PER_BIN = SAMPLE_MINUTES // NATIVE_MINUTES
HISTORY_ROWS = HISTORY_STEPS * NATIVE_PER_BIN


def load_windows(reference, leads=LEADS):
    path = ROOT / "data" / reference["source"]
    if sha256(path.read_bytes()).hexdigest() != reference["source_sha256"]:
        raise ValueError("The dataset does not match the saved checksum")
    data = pd.read_csv(path, parse_dates=["time"], usecols=["time", *FEATURES, "row_observed", "row_smi_filled"])
    end = data["time"].to_numpy(dtype="datetime64[ns]") + np.timedelta64(NATIVE_MINUTES, "m")
    np.testing.assert_array_equal(np.diff(end), np.timedelta64(NATIVE_MINUTES, "m"))
    values = data[list(FEATURES)].to_numpy(dtype=np.float64)
    available = data["row_observed"].to_numpy() & ~data["row_smi_filled"].to_numpy()
    valid_past = available & np.isfinite(values).all(axis=1) & (values[:, 1:] >= 0).all(axis=1)
    valid_power = available & np.isfinite(values[:, 0])
    ends_min = end.astype("datetime64[m]").astype(np.int64)
    lead_offsets = np.asarray(leads) // NATIVE_MINUTES - 1
    windows = {}
    for split in ("train", "validation"):
        first, stop = reference["split_row_boundaries"][split]
        history_valid = np.lib.stride_tricks.sliding_window_view(valid_past[first:stop], HISTORY_ROWS)
        starts = np.flatnonzero(history_valid.all(axis=1)) + first
        issuance = starts + HISTORY_ROWS
        issuance = issuance[(issuance + lead_offsets[-1] < stop) & (ends_min[issuance - 1] % SAMPLE_MINUTES == 0)]
        target_rows = issuance[:, None] + lead_offsets
        keep = valid_power[target_rows].all(axis=1)
        issuance, target_rows = issuance[keep], target_rows[keep]
        if not len(issuance):
            raise ValueError(f"No aligned windows in {split}")
        past = issuance[:, None] + np.arange(-HISTORY_ROWS, 0)
        binned = values[past].reshape(len(past), HISTORY_STEPS, NATIVE_PER_BIN, len(FEATURES)).mean(axis=2)
        origin = end[issuance - 1]
        target_time = end[target_rows]
        np.testing.assert_array_equal(target_time, origin[:, None] + np.asarray(leads) * np.timedelta64(1, "m"))
        windows[split] = dict(history=binned, origin=origin, target_time=target_time, truth=values[target_rows, 0],
                              last_native_power=values[issuance - 1, 0])
    first, stop = reference["split_row_boundaries"]["train"]
    bins = np.flatnonzero((ends_min[first:stop] % SAMPLE_MINUTES == 0)) + first
    bins = bins[bins - (NATIVE_PER_BIN - 1) >= first, None] + np.arange(1 - NATIVE_PER_BIN, 1)
    training = values[bins[valid_past[bins].all(axis=1)]].mean(axis=1)
    center, scale = training.mean(axis=0), training.std(axis=0)
    denominator = float(np.ptp(training[:, 0]))
    if not np.isfinite(scale).all() or (scale <= 0).any() or denominator <= 0:
        raise ValueError("Expected positive training ranges and variances")
    return windows, center, scale, denominator


def skill(mse, baseline, truth):
    baseline_mse = np.mean((baseline - truth) ** 2)
    return 100 * (1 - mse / baseline_mse) if baseline_mse else float("nan")


def scores(predicted, truth, persistence, native, divisor):
    with np.errstate(divide="ignore", invalid="ignore"):
        result = io_utils.score(predicted, truth)
    if not np.isfinite(result["r2"]):
        result["r2"] = float("nan")
    mse = np.mean((predicted - truth) ** 2)
    result.update(nrmse_pct=100 * result["rmse_mw"] / divisor, nmae_pct=100 * result["mae_mw"] / divisor,
                  persistence_skill_pct=skill(mse, persistence, truth), native_persistence_skill_pct=skill(mse, native, truth))
    return result


def summarize(rows, keys):
    groups = {}
    for row in rows:
        groups.setdefault(tuple(row[key] for key in keys), []).append(row)
    metrics = ("rmse_mw", "mae_mw", "r2", "nrmse_pct", "nmae_pct", "persistence_skill_pct", "native_persistence_skill_pct")
    return [{**dict(zip(keys, key)), "runs": len(group), "windows": group[0]["windows"],
             **{name: mean(row[name] for row in group) for name in metrics}} for key, group in groups.items()]


def write_validation_reports(output, val, predictions, divisor, leads=LEADS, *, include_persistence=True):
    truth = val["truth"]
    persistence = np.broadcast_to(val["history"][:, -1, :1], truth.shape)
    native = np.broadcast_to(val["last_native_power"][:, None], truth.shape)
    curves = predictions
    if include_persistence:
        for name, curve in (("persistence", persistence), ("persistence_native", native)):
            np.savez_compressed(output / f"{name}_predictions.npz", origin=val["origin"], target_time=val["target_time"],
                                target_mw=truth, prediction_mw=curve)
        curves = {("persistence", ""): persistence, ("persistence_native", ""): native, **predictions}
    gaps = np.flatnonzero(np.diff(val["origin"]) != np.timedelta64(SAMPLE_MINUTES, "m")) + 1
    blocks = np.split(np.arange(len(truth)), gaps)
    rows, block_rows = [], []
    for (name, seed), predicted in curves.items():
        if predicted.shape != truth.shape or not np.isfinite(predicted).all():
            raise ValueError(f"Invalid forecasts for {name} seed={seed}")
        for i, lead in enumerate(leads):
            rows.append(dict(model=name, seed=seed, lead_minutes=lead, windows=len(truth),
                             **scores(predicted[:, i], truth[:, i], persistence[:, i], native[:, i], divisor)))
            for index, block in enumerate(blocks, 1):
                block_rows.append(dict(model=name, seed=seed, lead_minutes=lead, block=index,
                                       start=str(val["origin"][block[0]]), end=str(val["origin"][block[-1]]), windows=len(block),
                                       **scores(predicted[block, i], truth[block, i], persistence[block, i], native[block, i], divisor)))
    io_utils.write_csv(output / "validation_metrics.csv", rows)
    io_utils.write_csv(output / "validation_summary.csv", summarize(rows, ("model", "lead_minutes")))
    io_utils.write_csv(output / "validation_stability.csv", summarize(block_rows, ("model", "lead_minutes", "block", "start", "end")))
    return [len(block) for block in blocks]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--leads", type=int, nargs="+", default=LEADS, help="Forecast offsets in minutes")
    parser.add_argument("--width", type=int, default=32, help="Hidden channel width for both branches")
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout rate inside MFT/Strip/CAFR")
    parser.add_argument("--mft-features", choices=("fan", "tcn"), default="tcn",
                        help="Feature module between MHCB and Pre-LN residual TAN")
    parser.add_argument("--input-layout", choices=("wind_cwt", "power_cwt"), default="power_cwt",
                        help="Use wind CWT, or use power CWT with all inputs in the temporal branch")
    parser.add_argument("--temporal-rnn", choices=("none", "lstm", "bilstm", "bigru"), default="none",
                        help="Optional residual RNN after MFT (default: none)")
    parser.add_argument("--cwt-encoder", choices=("strip", "original"), default="strip",
                        help="Scale-time encoder for CWT maps (AAAI 2026 StripModule or raw CWT)")
    parser.add_argument("--cwt-full-support", action="store_true",
                        help="Mask CWT boundary steps (default: off to preserve recent dynamics)")
    parser.add_argument("--fusion", choices=("cafr", "gated"), default="cafr",
                        help="Dual-branch fusion (AAAI 2026 CAFR1D or gated concat)")
    parser.add_argument("--output-mode", choices=("anchor", "revin"), default="anchor",
                        help="Output formulation: anchor residual or RevIN direct forecast")
    parser.add_argument("--native-power-anchor", action=argparse.BooleanOptionalAction, default=True,
                        help="Predict changes from the latest observed five-minute power")
    parser.add_argument("--wind-selection", action=argparse.BooleanOptionalAction, default=True,
                        help="Learn time-dependent weights for the four wind heights")
    comparison_mode = parser.add_mutually_exclusive_group()
    comparison_mode.add_argument("--comparison-run", type=Path, help="Reuse saved Ours and Transformer predictions")
    comparison_mode.add_argument("--ours-only", action="store_true", help="Train only the current Ours model")
    parser.add_argument("--reference-only", action="store_true", help="Compare only with the saved Ours model")
    args = parser.parse_args()

    if args.reference_only and args.comparison_run is None:
        parser.error("--reference-only requires --comparison-run")
    leads = tuple(sorted(set(args.leads)))
    if any(lead < NATIVE_MINUTES or lead % NATIVE_MINUTES for lead in leads):
        parser.error("Every lead must be a positive multiple of five minutes")

    reference = io_utils.read_json(ROOT / "data" / "protocol.json")
    windows, center, scale, divisor = load_windows(reference, leads)
    cfg = replace(TrainConfig(**reference["optimizer"]), history=HISTORY_STEPS,
                  device="cuda" if torch.cuda.is_available() else "cpu", seeds=tuple(reference["seeds"]))
    history = {name: ((window["history"] - center) / scale).astype(np.float32) for name, window in windows.items()}
    targets = {name: ((window["truth"] - center[0]) / scale[0]).astype(np.float32) for name, window in windows.items()}

    lead_tag = "_".join(f"{lead}min" for lead in leads)
    run_type = f"{args.input_layout}_{args.mft_features}_w{args.width}"
    suffix = f"{lead_tag}_{args.cwt_encoder}_{args.fusion}_rnn-{args.temporal_rnn}_{args.output_mode}"
    if args.native_power_anchor and args.output_mode == "anchor":
        suffix += "_native"
    if args.wind_selection:
        suffix += "_windgate"
    output = ROOT / "runs" / f"{datetime.now():%Y%m%d_%H%M%S_%f}_{run_type}_{suffix}"
    output.mkdir(parents=True)
    print(f"RESULT DIRECTORY {output}", flush=True)
    print(f"WINDOWS {dict((name, len(window['origin'])) for name, window in windows.items())}", flush=True)

    use_native_anchor = args.native_power_anchor and args.output_mode == "anchor"
    model_config = dict(
        history=HISTORY_STEPS, horizon=len(leads), variables=len(FEATURES),
        width=args.width, fan="tcn" if args.mft_features == "tcn" else "dual-attn",
        cwt_scales=SCALES, cwt_encoder=args.cwt_encoder, cwt_full_support=args.cwt_full_support,
        temporal_rnn=args.temporal_rnn, fusion=args.fusion, output_mode=args.output_mode,
        native_power_anchor=use_native_anchor, wind_selection=args.wind_selection,
        input_layout=args.input_layout, dropout=args.dropout,
    )
    if args.input_layout == "wind_cwt":
        model_config["cwt_wind_indices"] = (1, 2, 3, 4)

    ours_history = history
    if use_native_anchor:
        ours_history = {}
        for name, features in history.items():
            anchor = np.zeros_like(features[..., :1])
            anchor[:, -1, 0] = (windows[name]["last_native_power"] - center[0]) / scale[0]
            ours_history[name] = np.concatenate((features, anchor), axis=-1)

    model_inputs = {"ours": ours_history}
    val = windows["validation"]
    truth = val["truth"]
    predictions = {}
    if args.comparison_run is None:
        if not args.ours_only:
            model_inputs["transformer"] = {name: x[..., TRANSFORMER_CHANNELS] for name, x in history.items()}
    else:
        comparison = io_utils.read_json(args.comparison_run / "protocol.json")
        if (comparison["sample_minutes"], comparison["history_steps"], comparison["lead_minutes"], comparison["features"]) != (
            SAMPLE_MINUTES, HISTORY_STEPS, list(leads), list(FEATURES),
        ):
            parser.error("Comparison must use the same input sampling, history, features and forecast horizons")
        references = {"ours": "ours_reference"}
        if not args.reference_only:
            references["transformer"] = "transformer"
        for name, label in references.items():
            for seed in cfg.seeds:
                archive = args.comparison_run / name / f"seed{seed}" / "validation_predictions.npz"
                with np.load(archive, allow_pickle=False) as record:
                    for key in ("origin", "target_time"):
                        np.testing.assert_array_equal(record[key], val[key])
                    np.testing.assert_array_equal(record["target_mw"], truth)
                    predictions[label, seed] = record["prediction_mw"]

    for name in model_inputs:
        (output / name).mkdir()
        for seed in cfg.seeds:
            torch.manual_seed(seed)
            model = DualMFTForecaster(**model_config) if name == "ours" else Transformer(2, len(leads), cfg)
            print(f"TRAIN {name} seed={seed}", flush=True)
            directory = output / name / f"seed{seed}"
            result = fit_curve(model, model_inputs[name], targets, replace(cfg, seeds=(seed,)), directory)
            forecast = result["validation"].astype(np.float64) * scale[0] + center[0]
            predictions[name, seed] = forecast
            np.savez_compressed(directory / "validation_predictions.npz", origin=val["origin"], target_time=val["target_time"],
                                target_mw=truth, prediction_mw=forecast)

    blocks = write_validation_reports(output, val, predictions, divisor, leads,
                                      include_persistence=not args.reference_only)
    io_utils.write_json(output / "protocol.json", {
        "sample_minutes": SAMPLE_MINUTES, "issue_minutes": SAMPLE_MINUTES, "history_steps": HISTORY_STEPS,
        "native_target_minutes": NATIVE_MINUTES, "lead_minutes": leads, "features": FEATURES,
        "transformer_features": [FEATURES[i] for i in TRANSFORMER_CHANNELS], "cwt_model": model_config,
        "scaler_mean": center.tolist(), "scaler_std": scale.tolist(), "normalizer_mw": divisor,
        "persistence": "Last observed 15-minute average power at issue time",
        "native_persistence": "Last observed native five-minute power at issue time", "training": vars(cfg),
        "comparison_run": str(args.comparison_run.resolve()) if args.comparison_run else None,
        "windows": {name: len(window["origin"]) for name, window in windows.items()},
        "continuous_validation_blocks": blocks,
    })
    print(f"COMPLETED {output}", flush=True)


if __name__ == "__main__":
    main()
