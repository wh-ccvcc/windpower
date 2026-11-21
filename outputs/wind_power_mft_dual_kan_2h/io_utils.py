import csv
import json
import numpy as np

SCORE_KEYS = ("rmse_mw", "mae_mw", "r2")


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_csv(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def score(predicted, target):
    predicted, target = np.asarray(predicted, dtype=float), np.asarray(target, dtype=float)
    error = predicted - target
    squared = error ** 2
    return dict(rmse_mw=float(np.sqrt(squared.mean())), mae_mw=float(np.abs(error).mean()),
                r2=float(1 - squared.sum() / ((target - target.mean()) ** 2).sum()))
