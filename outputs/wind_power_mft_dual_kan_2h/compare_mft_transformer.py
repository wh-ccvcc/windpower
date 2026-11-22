"""Compare single-branch MFT and Transformer on native five-minute forecasts."""

import argparse
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch import nn

import io_utils
from temporal import VANFANTANEncoder
from train_multiheight_short import FEATURES, HISTORY_ROWS, NATIVE_MINUTES, ROOT, Transformer, load_windows
from training import TrainConfig, fit, short_predict


LEADS = tuple(range(5, 31, 5))


class MFTForecast(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.encoder = VANFANTANEncoder(cfg.history, len(FEATURES), cfg.d_model, fan="tcn", dropout=cfg.dropout)
        self.head = nn.Linear(cfg.d_model, len(LEADS))

    def forward(self, history):
        return self.head(self.encoder(history)[:, -1])


def score_rows(model, split, prediction, truth, persistence):
    for lead, column in (("all", slice(None)), *((lead, i) for i, lead in enumerate(LEADS))):
        measured = truth[:, column]
        result = io_utils.score(prediction[:, column], measured)
        baseline_rmse = io_utils.score(persistence[:, column], measured)["rmse_mw"]
        skill = 100 * (1 - (result["rmse_mw"] / baseline_rmse) ** 2) if baseline_rmse else None
        yield dict(model=model, split=split, lead_minutes=lead, windows=len(truth),
                   **result, persistence_skill_pct=skill)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable")
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    torch.set_num_threads(args.threads)

    reference = io_utils.read_json(ROOT / "data" / "protocol.json")
    windows, center, scale, _ = load_windows(reference, LEADS, sample_minutes=NATIVE_MINUTES, include_test=True)
    cfg = replace(TrainConfig(**reference["optimizer"]), device=device, history=HISTORY_ROWS, seeds=(42,),
                  d_model=32, d_ff=64, dropout=0.0)
    inputs = {split: ((data["history"] - center) / scale).astype(np.float32) for split, data in windows.items()}
    targets = {split: ((data["truth"] - data["last_native_power"][:, None]) / scale[0]).astype(np.float32)
               for split, data in windows.items()}
    output = args.output or ROOT / "runs" / f"{datetime.now():%Y%m%d_%H%M%S}_mft_transformer_5min"
    output.mkdir(parents=True)
    models = {
        "mft": lambda: MFTForecast(cfg),
        "transformer": lambda: Transformer(len(FEATURES), len(LEADS), cfg, readout="last"),
    }
    metadata = {
        "source": reference["source"], "source_sha256": reference["source_sha256"],
        "split_row_boundaries": reference["split_row_boundaries"],
        "sample_minutes": NATIVE_MINUTES, "issue_minutes": NATIVE_MINUTES,
        "history_steps": cfg.history, "lead_minutes": LEADS, "features": FEATURES,
        "windows": {split: len(data["truth"]) for split, data in windows.items()},
        "scaler_mean": center.tolist(), "scaler_std": scale.tolist(), "training": asdict(cfg),
        "torch_version": str(torch.__version__), "cpu_threads": args.threads,
        "readout": "Last history step with a linear head predicting six residuals from the last observed power.",
        "models": {"mft": "MHCB + two residual TCN blocks + one Pre-LN TAN block",
                   "transformer": "Two standard Transformer encoder layers with sinusoidal positions"},
        "trainable_parameters": {},
    }
    io_utils.write_json(output / "protocol.json", metadata)
    print(f"RESULT DIRECTORY {output}", flush=True)
    print(f"WINDOWS {metadata['windows']}", flush=True)
    rows = []
    for split in ("validation", "test"):
        data = windows[split]
        persistence = np.broadcast_to(data["last_native_power"][:, None], data["truth"].shape)
        rows.extend(score_rows("persistence", split, persistence, data["truth"], persistence))
    for name, build_model in models.items():
        torch.manual_seed(cfg.seeds[0])
        model = build_model()
        metadata["trainable_parameters"][name] = sum(p.numel() for p in model.parameters() if p.requires_grad)
        io_utils.write_json(output / "protocol.json", metadata)
        folder = output / name
        print(f"TRAIN {name} seed={cfg.seeds[0]}", flush=True)
        fit(model, inputs, targets, cfg, folder)
        checkpoint = torch.load(folder / "best.pt", map_location="cpu", weights_only=True)
        model.load_state_dict(checkpoint["model_state"])
        for split in ("validation", "test"):
            data = windows[split]
            persistence = np.broadcast_to(data["last_native_power"][:, None], data["truth"].shape)
            prediction = short_predict(model, inputs[split], cfg).astype(np.float64) * scale[0] + persistence
            np.savez_compressed(folder / f"{split}_predictions.npz", origin=data["origin"],
                                target_time=data["target_time"], prediction_mw=prediction, target_mw=data["truth"])
            rows.extend(score_rows(name, split, prediction, data["truth"], persistence))
        io_utils.write_csv(output / "metrics.csv", rows)
    print(f"COMPLETED {output}", flush=True)


if __name__ == "__main__":
    main()
