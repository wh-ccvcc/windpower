# 风电功率预测代码（MFT-PreLN + CWT-Strip + CAFR + BiGRU）

左路对 4 个高度风速（90/70/30/10m）做动态门控加权后，与历史功率一起送入 **MFT（MHCB + TCN + Pre-LN 残差多头自注意力 TAN）** 及可选单层残差 **BiGRU**。右路对历史功率做 4 尺度墨西哥帽 CWT（保留发报前末端响应，关闭边界清零），经轻量 **AAAI 2026 StripModule**（时间条带 `(1, 11)` + 尺度条带 `(3, 1)`，替代原 13 层 VGG-16）与因果 TCN 投影。左右两路经 **AAAI 2026 CAFR1D**（跨特征通道细化融合）对齐融合，结合末步与注意力池化特征由线性层预测功率变化量，叠加最新 5 分钟实测锚点输出最终预测。历史序列为过去 24 小时，每 15 分钟一步。

## 主模型训练

进入 `outputs/wind_power_mft_dual_kan_2h`，原始数据位于 `data/wind_power_smi_shortgap.csv`，切分与校验值见 `data/protocol.json`。以下命令已用 Python 3.11（`windpower` 环境）实测通过。

### 最优配置（+20 分钟验证集 RMSE 7.550 MW，MAE 5.566 MW，R² 0.9560，相对 5m 持续法 Skill +5.59%）

```text
cd outputs/wind_power_mft_dual_kan_2h
python train_multiheight_short.py --leads 20 --width 32 --dropout 0.2 --cwt-encoder strip --fusion cafr --temporal-rnn bigru --output-mode anchor --ours-only
```

Windows PowerShell 快捷命令：

```powershell
.\train_windpower.ps1 -Lead 20 -Width 32 -CwtEncoder strip -Fusion cafr -TemporalRnn bigru
```

### 常用消融参数

* `--leads 15` / `20` / `25`：切换预测时效（分钟）。
* `--width 32` / `48`、`--dropout 0.2`：调整隐藏宽度与正则。
* `--fusion cafr` / `gated`：对比 AAAI 2026 CAFR1D 跨通道融合与逐时间步门控融合。
* `--temporal-rnn none` / `bigru` / `bilstm` / `lstm`：切换左路残差循环层。
* `--cwt-encoder strip` / `original`：对比 StripModule 与原始 CWT。
* `--input-layout power_cwt` / `wind_cwt`：切换功率小波或风速小波布局。
* `--output-mode anchor` / `revin`：对比 5 分钟实测锚点残差或 RevIN 直接输出。

## 已保存的新版实验（`runs/`）

| 目录关键字 | 配置摘要 (+20min) | RMSE (MW) | MAE (MW) | R² | 5m 持续法 Skill |
| :--- | :--- | :---: | :---: | :---: | :---: |
| `w32_20min_strip_cafr_rnn-bigru_anchor` | **最优：w32 + Strip + CAFR + BiGRU + anchor** | **7.550** | **5.566** | **0.9560** | **+5.59%** |
| `w48_20min_strip_cafr_rnn-none` | w48 + Strip + CAFR + 无 RNN + anchor | 7.633 | 5.713 | 0.9551 | +3.51% |
| `w32_20min_strip_gated_rnn-none_anchor` | w32 + Strip + Gated + 无 RNN + anchor | 7.634 | 5.695 | 0.9551 | +3.50% |
| `w32_20min_strip_cafr_rnn-none_anchor` | w32 + Strip + CAFR + 无 RNN + anchor | 7.660 | 5.730 | 0.9548 | +2.82% |
| `w32_20min_strip_cafr_rnn-none_revin` | w32 + Strip + CAFR + RevIN 直接输出 | 8.775 | 6.571 | 0.9406 | -27.50% |

## 基线对比

```text
python -m pip install statsmodels
python run_40_60min_baselines.py prepare --reference "runs/你的实验目录"
python run_40_60min_baselines.py neural --reference "runs/你的实验目录"
python run_40_60min_baselines.py arima --reference "runs/你的实验目录"
python run_40_60min_baselines.py report --reference "runs/你的实验目录"
```
