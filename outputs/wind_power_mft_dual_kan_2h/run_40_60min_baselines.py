"""Compare additional baselines on a saved multi-height validation run."""

import argparse
import csv
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
import json
import os
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parent
DEFAULT_REFERENCE = ROOT / "runs" / "20261120_193957_771852_multiheight_15min_40min_60min"
SAMPLE_MINUTES = 15
NEURAL = ("transformer", "lstm", "bilstm", "gru", "itransformer", "informer", "tft")
sys.path.insert(0, str(ROOT.parent))


def prepare(reference, output, leads):
    from train_multiheight_short import TRANSFORMER_CHANNELS, load_windows
    import io_utils

    data = io_utils.read_json(ROOT / "data" / "protocol.json")
    windows, center, scale, divisor = load_windows(data, leads)
    validation = windows["validation"]
    with np.load(reference / "persistence_predictions.npz", allow_pickle=False) as expected:
        for key in ("origin", "target_time"):
            np.testing.assert_array_equal(validation[key], expected[key])
        np.testing.assert_array_equal(validation["truth"], expected["target_mw"])
    output.mkdir(exist_ok=True)
    archive = {}
    for split, span in windows.items():
        history = ((span["history"] - center) / scale).astype(np.float32)
        archive[f"{split}_history"] = history[..., TRANSFORMER_CHANNELS]
        archive[f"{split}_target"] = ((span["truth"] - center[0]) / scale[0]).astype(np.float32)
    np.savez_compressed(output / "aligned_inputs.npz", **archive, origin=validation["origin"],
                        target_time=validation["target_time"], target_mw=validation["truth"],
                        scaler=np.array([center[0], scale[0]]), divisor=divisor)
    print(f"Prepared {len(windows['train']['origin'])} train and {len(validation['origin'])} validation windows", flush=True)


def save(folder, prediction, data):
    target = data["target_mw"]
    if prediction.shape != target.shape or not np.isfinite(prediction).all():
        raise ValueError(f"Invalid validation predictions in {folder}")
    np.savez_compressed(folder / "validation_predictions.npz", prediction_mw=prediction, target_mw=target,
                        origin=data["origin"], target_time=data["target_time"])


def neural_factory(name, config, leads):
    import torch
    from torch import nn
    from baselines.models.rnn import LSTM, GRU
    from validation_forecast_benchmark import neural_factory as prior_factory

    horizon = len(leads)
    if name == "transformer":
        from baselines.models.transformer import Transformer

        return Transformer(2, horizon, config)
    if name in ("lstm", "gru"):
        constructor = LSTM if name == "lstm" else GRU
        return constructor(2, horizon, config)
    if name == "itransformer":
        return prior_factory(name, config.history, horizon, config)
    if name == "tft":
        result = prior_factory(name, config.history, horizon, config)
        minutes = torch.tensor([*range(-(config.history - 1) * SAMPLE_MINUTES, 1, SAMPLE_MINUTES), *leads],
                               dtype=torch.float32)
        result.positions.copy_((2 * (minutes - minutes[0]) / (minutes[-1] - minutes[0]) - 1).reshape(1, -1, 1))
        return result
    if name == "bilstm":
        class BidirectionalLSTM(nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = config.rnn_layers
                self.lstm = nn.LSTM(2, config.rnn_hidden, num_layers=self.layers, batch_first=True,
                                    bidirectional=True, dropout=config.dropout if self.layers > 1 else 0)
                self.head = nn.Linear(2 * config.rnn_hidden, horizon)

            def forward(self, history):
                _, (hidden, _) = self.lstm(history)
                last_layer = hidden.reshape(self.layers, 2, len(history), -1)[-1]
                return self.head(last_layer.transpose(0, 1).flatten(1))

        return BidirectionalLSTM()
    if name == "informer":
        vendor = ROOT.parent / "original_15min_experiment" / "20261118_103844_637385" / "snapshot" / "Informer2020"
        sys.path.insert(0, str(vendor))
        try:
            from models.model import Informer
        finally:
            sys.path.pop(0)

        class PowerInformer(nn.Module):
            def __init__(self):
                super().__init__()
                self.label_steps = config.history // 2
                self.model = Informer(2, 2, 1, config.history, self.label_steps, horizon,
                                      d_model=config.d_model, n_heads=config.n_heads, e_layers=config.n_layers, d_layers=1,
                                      d_ff=config.d_ff, dropout=config.dropout, attn="prob", freq="h", embed="timeF", distil=True)

            def forecast(self, history):
                decoder = torch.cat((history[:, -self.label_steps:], history.new_zeros((len(history), horizon, 2))), dim=1)
                marks_enc = history.new_zeros((*history.shape[:2], 4))
                marks_dec = history.new_zeros((*decoder.shape[:2], 4))
                return self.model(history, marks_enc, decoder, marks_dec).squeeze(-1)

            def forward(self, history):
                if self.training:
                    return self.forecast(history)
                devices = [history.device.index] if history.is_cuda else []
                with torch.random.fork_rng(devices=devices):
                    torch.manual_seed(42)
                    return self.forecast(history)

        return PowerInformer()
    raise ValueError(f"Unknown benchmark {name}")


def run_neural(names, reference, output, leads):
    import torch
    from train_multiheight_short import HISTORY_STEPS
    from training import TrainConfig, fit, fit_curve
    import io_utils

    torch.set_num_threads(2)
    protocol = io_utils.read_json(reference / "protocol.json")
    cfg = replace(TrainConfig(**protocol["training"]), history=HISTORY_STEPS,
                  device="cuda" if torch.cuda.is_available() else "cpu", seeds=tuple(protocol["training"]["seeds"]))
    if "tft" in names and cfg.device == "cpu":
        raise RuntimeError("The NVIDIA TFT implementation requires CUDA")
    with np.load(output / "aligned_inputs.npz", allow_pickle=False) as data:
        x = {split: data[f"{split}_history"] for split in ("train", "validation")}
        y = {split: data[f"{split}_target"] for split in x}
        offset, scale = data["scaler"]
        for name in names:
            parent = output / name
            parent.mkdir(exist_ok=True)
            for seed in cfg.seeds:
                folder = parent / f"seed{seed}"
                print(f"TRAIN {name} seed={seed}", flush=True)
                torch.manual_seed(seed)
                model = neural_factory(name, cfg, leads).to(cfg.device)
                seed_cfg = replace(cfg, seeds=(seed,))
                if name == "tft":
                    # Initialize lazy vendor parameters and use its no-grad evaluation path.
                    with torch.no_grad():
                        model.eval()(torch.as_tensor(x["train"][:2], device=cfg.device))
                    fitted = fit(model, x, y, seed_cfg, folder)
                    with torch.no_grad():
                        inputs = torch.as_tensor(x["validation"], device=cfg.device)
                        prediction = torch.cat([fitted(part) for part in inputs.split(cfg.batch_size)]).cpu().numpy()
                else:
                    prediction = fit_curve(model, x, y, seed_cfg, folder)["validation"]
                save(folder, prediction.astype(np.float64) * scale + offset, data)
                print(f"COMPLETED {name} seed={seed}", flush=True)


def run_arima(output, leads):
    from validation_forecast_benchmark import fit_arima

    folder = output / "arima"
    folder.mkdir(exist_ok=True)
    with np.load(output / "aligned_inputs.npz", allow_pickle=False) as data:
        histories = data["validation_history"][..., 0].astype(np.float64)
        offset, scale = data["scaler"]
        steps = (max(leads) + SAMPLE_MINUTES - 1) // SAMPLE_MINUTES
        values = []
        with ProcessPoolExecutor(max_workers=min(6, os.cpu_count() or 1)) as pool:
            for index, row in enumerate(pool.map(fit_arima, ((row, steps) for row in histories), chunksize=8), 1):
                values.append(row)
                if index % 150 == 0:
                    print(f"ARIMA {index}/{len(histories)}", flush=True)
        predictions, converged = zip(*values)
        grid = np.column_stack((histories[:, -1], np.asarray(predictions)))
        minute_grid = np.arange(steps + 1) * SAMPLE_MINUTES
        power = np.stack([np.interp(leads, minute_grid, row) for row in grid])
        save(folder, power * scale + offset, data)
    metadata = dict(order=[2, 1, 2], fit="Refit each window using its 96 historical power observations",
                    windows=len(values), optimizer_converged=sum(converged),
                    target_mapping="Linear interpolation between 15-minute forecast steps and the last observation at minute zero")
    (folder / "protocol.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"COMPLETED ARIMA, optimizer converged on {sum(converged)}/{len(values)} windows", flush=True)


def report(reference, output, leads):
    from train_multiheight_short import load_windows, write_validation_reports
    import io_utils

    source = io_utils.read_json(ROOT / "data" / "protocol.json")
    data, _, _, divisor = load_windows(source, leads)
    validation = data["validation"]
    def read_predictions(folder):
        with np.load(folder / "validation_predictions.npz", allow_pickle=False) as record:
            for key in ("origin", "target_time"):
                np.testing.assert_array_equal(record[key], validation[key])
            np.testing.assert_array_equal(record["target_mw"], validation["truth"])
            return record["prediction_mw"]

    predictions = {("arima", ""): read_predictions(output / "arima")}
    protocol = io_utils.read_json(reference / "protocol.json")
    for model in ("ours", *NEURAL):
        parent = (reference if model == "ours" else output) / model
        for seed in protocol["training"]["seeds"]:
            folder = parent / f"seed{seed}"
            if model == "transformer" and not (folder / "validation_predictions.npz").is_file():
                folder = reference / model / f"seed{seed}"
            predictions[model, seed] = read_predictions(folder)
    blocks = write_validation_reports(output, validation, predictions, divisor, leads)
    io_utils.write_json(output / "protocol.json", {
        "reference_run": str(reference), "history_steps": 96, "sample_minutes": SAMPLE_MINUTES, "lead_minutes": leads,
        "neural_baseline_inputs": ["historical power", "historical 70 m wind"], "arima_input": "historical power",
        "validation_windows": len(validation["origin"]), "validation_blocks": blocks,
        "persistence_inputs": ["last 15-minute mean power", "latest observed five-minute power"],
        "neural_seeds": protocol["training"]["seeds"], "intransformer_interpretation": "Informer",
    })
    with (output / "validation_summary.csv").open(encoding="utf-8-sig", newline="") as file:
        for row in csv.DictReader(file):
            print(row["model"], row["lead_minutes"], row["r2"], row["rmse_mw"], flush=True)


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("stage", choices=("prepare", "neural", "arima", "report"))
    cli.add_argument("--models", nargs="+", choices=NEURAL)
    cli.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    args = cli.parse_args()
    reference = args.reference.resolve()
    protocol = json.loads((reference / "protocol.json").read_text(encoding="utf-8-sig"))
    if protocol["sample_minutes"] != SAMPLE_MINUTES or protocol["history_steps"] != 96:
        cli.error("Expected 96 historical points sampled every 15 minutes")
    leads = tuple(protocol["lead_minutes"])
    output = reference / "additional_baselines"
    if args.stage == "neural":
        models = args.models
        if models is None:
            saved_transformer = all((reference / "transformer" / f"seed{seed}" / "validation_predictions.npz").is_file()
                                    for seed in protocol["training"]["seeds"])
            models = tuple(name for name in NEURAL if name != "transformer" or not saved_transformer)
        run_neural(models, reference, output, leads)
    elif args.stage == "arima":
        run_arima(output, leads)
    else:
        {"prepare": prepare, "report": report}[args.stage](reference, output, leads)


if __name__ == "__main__":
    main()
