from dataclasses import asdict, dataclass
from io_utils import write_csv
import numpy as np
import torch
from torch import nn


@dataclass
class TrainConfig:
    """训练超参。服务器 GPU 上可增大 epochs / batch_size。"""
    epochs: int = 60
    batch_size: int = 128
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 10                 # 验证集早停耐心（0 = 关闭）
    gradient_clip_norm: float = 1.0
    seeds: tuple[int, ...] = (42,)
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers: int = 0
    # 轻量 smoke test：设小值只跑通流程；服务器上置 None 跑全量
    max_train_samples: int | None = None
    max_val_samples: int | None = None
    max_test_samples: int | None = None

    # 统一模型容量（各模型共享，便于公平对比）
    d_model: int = 64
    n_heads: int = 4
    d_ff: int = 128
    n_layers: int = 2
    dropout: float = 0.1
    history: int = 48           # 历史步长（由数据决定，加载后自动覆盖）
    feature_mode: str = "basic"  # basic / weather_dynamic
    rnn_hidden: int = 64
    rnn_layers: int = 2

    # ARIMA 固定阶数（避免服务器上做阶数搜索耗时）
    arima_order: tuple[int, int, int] = (2, 1, 2)

def _batches(X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool, seed: int):
    n = len(X)
    idx = np.arange(n)
    if shuffle:
        np.random.default_rng(seed).shuffle(idx)
    for s in range(0, n, batch_size):
        b = idx[s:s + batch_size]
        yield X[b], y[b]

def _tensor(arr, device):
    return torch.from_numpy(np.asarray(arr, dtype=np.float32)).to(device)

def train_torch(model, train, validation, cfg, seed, *, loss_fn=None):
    """训练 torch 模型，返回 (最优模型, 训练历史, 最佳验证MSE)。"""
    device = cfg.device
    torch.manual_seed(seed)
    np.random.seed(seed)
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate,
                            weight_decay=cfg.weight_decay)
    crit = nn.MSELoss() if loss_fn is None else loss_fn
    train_metric = "train_mse" if loss_fn is None else "train_loss"
    Xtr, ytr = train
    best = float("inf")
    best_state = None
    no_improve = 0
    history = []
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        total = 0.0
        cnt = 0
        for xb, yb in _batches(Xtr, ytr, cfg.batch_size, True, seed * 1000 + epoch):
            xb = _tensor(xb, device)
            yb = _tensor(yb, device)[..., 0]  # [B, H]
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            if cfg.gradient_clip_norm > 0:
                nn.utils.clip_grad_norm_(model.parameters(), cfg.gradient_clip_norm)
            opt.step()
            total += loss.item() * len(xb)
            cnt += len(xb)
        train_score = total / max(cnt, 1)
        val_mse = eval_mse(model, validation, cfg)
        history.append({"epoch": epoch, train_metric: train_score, "val_mse": val_mse})
        print(f"    epoch {epoch:3d}/{cfg.epochs}: {train_metric}={train_score:.4f} "
              f"val_mse={val_mse:.4f}{'  *' if val_mse < best - 1e-9 else ''}", flush=True)
        if val_mse < best - 1e-9:
            best = val_mse
            no_improve = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            no_improve += 1
        if cfg.patience and no_improve >= cfg.patience:
            break
    model.load_state_dict(best_state)
    return model, history, best

@torch.no_grad()
def eval_mse(model, split, cfg):
    model.eval()
    X, y = split
    total = 0.0
    cnt = 0
    for xb, yb in _batches(X, y, cfg.batch_size, False, 0):
        xb = _tensor(xb, cfg.device)
        yb = _tensor(yb, cfg.device)[..., 0]
        pred = model(xb)
        total += ((pred - yb) ** 2).sum().item()
        cnt += yb.numel()
    return total / max(cnt, 1)

def save_fit(model, history, config, best_epoch, folder):
    folder.mkdir()
    write_csv(folder / "history.csv", history)
    torch.save(dict(model_state={k: v.detach().cpu() for k, v in model.state_dict().items()},
                    config=asdict(config), best_epoch=best_epoch), folder / "best.pt")

def fit_single(model, train, validation, cfg, folder, *, loss_fn=None):
    model, history, best = train_torch(model, train, validation, cfg, cfg.seeds[0], loss_fn=loss_fn)
    np.testing.assert_allclose(eval_mse(model, validation, cfg), best, rtol=1e-6)
    best_epoch = min(history, key=lambda row: row["val_mse"])["epoch"]
    save_fit(model, history, cfg, best_epoch, folder)
    return model.eval()

@torch.inference_mode()
def short_predict(model, history, cfg):
    power = torch.as_tensor(history, dtype=torch.float32, device=cfg.device)
    return torch.cat([model(chunk) for chunk in power.split(cfg.batch_size)]).cpu().numpy()

def fit(model, X, y, cfg, output, *, loss_fn=None):
    pairs = [(X[name], y[name][..., None]) for name in ("train", "validation")]
    return fit_single(model, *pairs, cfg, output, loss_fn=loss_fn)

def fit_curve(model, inputs, targets, cfg, folder, *, loss_fn=None):
    trained = fit(model, inputs, targets, cfg, folder, loss_fn=loss_fn)
    predictions = {name: short_predict(trained, array, cfg)
                   for name, array in inputs.items()}
    model.load_state_dict(torch.load(folder / "best.pt", map_location="cpu",
                                     weights_only=True)["model_state"])
    replay = short_predict(model.eval(), inputs["validation"], cfg)
    np.testing.assert_array_equal(predictions["validation"], replay)
    return predictions
