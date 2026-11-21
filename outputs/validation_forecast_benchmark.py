"""Independent validation-only benchmark for the 10-minute, one-hour forecast."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
import json
from pathlib import Path
import os
import sys
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parent / "validation_forecast_benchmark_10min_1h"
REFERENCE = ROOT.parent / "forecast_wind_fusion_10min_1h_trial" / "20261119_223124_466918"
NAMES = ("arima", "cnn", "lstm", "itransformer", "autoformer", "tft")


def load_data():
    from forecast_wind_fusion_trial import baseline, binned_supplied_wind_windows

    protocol = baseline.read_json(REFERENCE / "protocol.json")
    full_data, full_windows, _, center, scale = binned_supplied_wind_windows(protocol, weather_steps=0)
    data, targets, windows = {}, {}, {}
    for split in ("train", "validation"):
        span = full_windows[split]
        with np.load(REFERENCE / f"{split}_wind_predictions.npz", allow_pickle=False) as record:
            ids = np.searchsorted(span["input_time"][:, -1], record["origin"])
            windows[split] = {name: values[ids] for name, values in span.items()}
            np.testing.assert_array_equal(windows[split]["input_time"][:, -1], record["origin"])
            np.testing.assert_array_equal(windows[split]["target_time"], record["target_time"])
        data[split] = full_data[split]["inputs"][ids]
        targets[split] = full_data[split]["power_target"][ids]
    return protocol, data, targets, windows, (float(center[0]), float(scale[0]))


def fit_arima(task):
    # Spawned workers need their own import path on Windows.
    temp_dir = os.environ.get("TEMP")
    if temp_dir:
        sys.path.insert(0, str(Path(temp_dir) / "windpower_benchmark_arima_packages"))
    import warnings
    from statsmodels.tsa.arima.model import ARIMA

    history, horizon = task
    if np.ptp(history) == 0:
        return np.repeat(history[-1], horizon), True
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        model = ARIMA(history, order=(2, 1, 2))
        result = model.fit(method_kwargs={"maxiter": 80, "warn_convergence": False})
        if not result.mle_retvals["converged"]:
            retry = model.fit(start_params=result.params, method_kwargs={"method": "powell", "maxiter": 80,
                                                                          "warn_convergence": False})
            if retry.mle_retvals["converged"] or retry.llf > result.llf:
                result = retry
    return result.forecast(steps=horizon), bool(result.mle_retvals["converged"])


def save_predictions(folder, prediction, windows):
    truth = windows["y_mw"][..., 0]
    if prediction.shape != truth.shape or not np.isfinite(prediction).all():
        raise ValueError(f"Invalid validation predictions for {folder.name}")
    np.savez_compressed(folder / "validation_predictions.npz", prediction_mw=prediction, target_mw=truth,
                        origin=windows["input_time"][:, -1], target_time=windows["target_time"])


def run_arima(workers):
    with np.load(ROOT / "arima_inputs.npz", allow_pickle=False) as archive:
        history = archive["history"]
        center, scale = archive["scaler"]
        windows = {"y_mw": archive["target_mw"][..., None],
                   "input_time": archive["origin"][:, None], "target_time": archive["target_time"]}
    horizon = windows["target_time"].shape[1]
    folder = ROOT / "arima"
    folder.mkdir(exist_ok=True)
    tasks = ((curve, horizon) for curve in history)
    results = []
    print(f"TRAIN ARIMA with {workers} workers", flush=True)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for count, result in enumerate(pool.map(fit_arima, tasks, chunksize=8), 1):
            results.append(result)
            if count % 200 == 0:
                print(f"ARIMA {count}/{len(history)}", flush=True)
    predicted, converged = zip(*results)
    save_predictions(folder, np.asarray(predicted) * scale + center, windows)
    info = dict(order=(2, 1, 2), windows=len(results), converged=sum(converged))
    (folder / "fit.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(f"BEST arima converged={sum(converged)}/{len(results)}", flush=True)


def prepare_arima():
    _, data, _, windows, scaler = load_data()
    span = windows["validation"]
    np.savez_compressed(ROOT / "arima_inputs.npz", history=data["validation"][..., 0].astype(np.float64),
                        target_mw=span["y_mw"][..., 0], origin=span["input_time"][:, -1],
                        target_time=span["target_time"], scaler=scaler)


def neural_factory(name, history, horizon, cfg):
    import torch
    from torch import nn
    from baselines.models.rnn import LSTM
    from baselines.models.itransformer import ITransformer

    if name == "lstm":
        return LSTM(2, horizon, cfg)
    if name == "itransformer":
        return ITransformer(2, horizon, cfg)
    if name == "cnn":
        class CNN(nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = nn.Sequential(nn.Conv1d(2, 32, 5, padding=2), nn.GELU(),
                                             nn.Conv1d(32, 32, 5, padding=2), nn.GELU(),
                                             nn.Conv1d(32, 32, 3, padding=1), nn.GELU())
                self.readout = nn.Linear(32, horizon)

            def forward(self, x):
                return self.readout(self.encoder(x.transpose(1, 2))[:, :, -1])

        return CNN()
    if name == "autoformer":
        from importlib.util import module_from_spec, spec_from_file_location

        path = ROOT / "vendor" / "autoformer"
        sys.path.insert(0, str(path))
        try:
            spec = spec_from_file_location("autoformer_benchmark_vendor", path / "models" / "Autoformer.py")
            vendor = module_from_spec(spec)
            spec.loader.exec_module(vendor)
        finally:
            sys.path.pop(0)

        class Autoformer(nn.Module):
            def __init__(self):
                super().__init__()
                self.label_len = horizon * 2
                self.model = vendor.Model(SimpleNamespace(seq_len=history, label_len=self.label_len, pred_len=horizon,
                    output_attention=False, moving_avg=13, enc_in=2, dec_in=2, c_out=2, d_model=32, n_heads=4,
                    d_ff=64, dropout=cfg.dropout, embed="timeF", freq="t", factor=2, activation="gelu", e_layers=2, d_layers=1))

            def forward(self, x):
                length = self.label_len + horizon
                decoder = torch.cat((x[:, -self.label_len:], x.new_zeros((len(x), horizon, 2))), dim=1)
                y = self.model(x, x.new_zeros((len(x), history, 5)), decoder, x.new_zeros((len(x), length, 5)))
                return y[..., 0]

        return Autoformer()
    if name == "tft":
        from importlib.util import module_from_spec, spec_from_file_location

        spec = spec_from_file_location("tft_benchmark_vendor", ROOT / "vendor" / "tft" / "modeling.py")
        vendor = module_from_spec(spec)
        sys.modules[spec.name] = vendor
        spec.loader.exec_module(vendor)

        class TFT(nn.Module):
            def __init__(self):
                super().__init__()
                self.model = vendor.TemporalFusionTransformer(SimpleNamespace(
                    static_categorical_inp_lens=[1], temporal_known_categorical_inp_lens=[],
                    temporal_observed_categorical_inp_lens=[], static_continuous_inp_size=0,
                    temporal_known_continuous_inp_size=1, temporal_observed_continuous_inp_size=1,
                    temporal_target_size=1, hidden_size=24, dropout=cfg.dropout, num_static_vars=1,
                    n_head=4, attn_dropout=cfg.dropout, example_length=history + horizon, encoder_length=history,
                    num_historic_vars=3, num_future_vars=1, quantiles=[0.5]))
                self.register_buffer("positions", torch.linspace(-1, 1, history + horizon).reshape(1, -1, 1))

            def forward(self, history_input):
                batch = len(history_input)
                inputs = {"s_cat": history_input.new_zeros((batch, 1, 1), dtype=torch.long),
                          "k_cont": self.positions.expand(batch, -1, -1),
                          "o_cont": history_input[..., 1:2], "target": history_input[..., :1]}
                return self.model(inputs).squeeze(-1)

        return TFT()
    raise ValueError(f"Unknown model {name}")


def run_neural(models):
    import torch
    from forecast_wind_fusion_trial import baseline, fit_curve, source

    torch.set_num_threads(2)
    protocol, data, targets, windows, (center, scale) = load_data()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if "tft" in models and device != "cuda":
        raise RuntimeError("The NVIDIA TFT implementation requires CUDA")
    cfg = replace(source.TrainConfig(**protocol["optimizer"]), device=device,
                  seeds=tuple(protocol["optimizer"]["seeds"]))
    for name in models:
        folder = ROOT / name
        folder.mkdir(exist_ok=True)
        ranking = []
        for seed in cfg.seeds:
            print(f"TRAIN {name} seed={seed}", flush=True)
            torch.manual_seed(seed)
            model = neural_factory(name, protocol["history_steps"], protocol["horizon_steps"], cfg).to(device)
            seed_dir = folder / f"seed{seed}"
            seed_cfg = replace(cfg, seeds=(seed,))
            if name == "tft":
                with torch.no_grad():
                    model.eval()(torch.as_tensor(data["train"][:2], dtype=torch.float32, device=device))
                if not (seed_dir / "best.pt").is_file():
                    source.fit(model, data, targets, seed_cfg, seed_dir)
                checkpoint = torch.load(seed_dir / "best.pt", map_location="cpu", weights_only=True)
                model.load_state_dict(checkpoint["model_state"], strict=True)
                # The upstream scripted TFT embedding does not support inference-mode tensors.
                with torch.no_grad():
                    x = torch.as_tensor(data["validation"], device=device)
                    normal = torch.cat([model.eval()(batch) for batch in x.split(cfg.batch_size)]).cpu().numpy()
            else:
                normal = fit_curve(model, data, targets, seed_cfg, seed_dir)["validation"]
            power = normal.astype(np.float64) * scale + center
            score = baseline.score(power, windows["validation"]["y_mw"][..., 0])
            ranking.append((score["rmse_mw"], seed, power))
            print(f"VERIFIED {name} seed={seed} validation RMSE={score['rmse_mw']:.6f}", flush=True)
        best, seed, power = min(ranking, key=lambda row: (row[0], row[1]))
        save_predictions(folder, power, windows["validation"])
        baseline.write_json(folder / "fit.json", dict(seed=seed, validation_rmse_mw=best, seeds=list(cfg.seeds)))
        print(f"BEST {name} seed={seed} validation RMSE={best:.6f}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("models", nargs="+", choices=(*NAMES, "prepare-arima"))
    parser.add_argument("--workers", type=int, default=5)
    args = parser.parse_args()
    if args.models == ["prepare-arima"]:
        prepare_arima()
    elif args.models == ["arima"]:
        run_arima(args.workers)
    elif "arima" in args.models or "prepare-arima" in args.models:
        parser.error("Run ARIMA separately from the CUDA models")
    else:
        run_neural(args.models)


if __name__ == "__main__":
    main()
