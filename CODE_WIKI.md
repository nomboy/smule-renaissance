# Smule Renaissance Small — Code Wiki

> 一个 10.4M 参数的生成式音频模型，用于在任何退化场景下修复受损人声（Vocal Restoration）。
> 在 iPhone 12 的 CPU 上以 10.5x 实时速度运行，主观质量超过所有开源模型，在歌声修复上媲美商用模型。

- 技术报告：https://arxiv.org/abs/2510.21659
- 模型权重：https://huggingface.co/smulelabs/Smule-Renaissance-Small
- 极端退化评测集：https://huggingface.co/datasets/smulelabs/ExtremeDegradationBench

---

## 1. 项目概览

本仓库是 Smule Renaissance Small 的**推理（Inference）实现**，包含：

1. 一个命令行入口，把任意采样率 / 声道的音频读入、预处理、送入模型、再写回增强后的音频。
2. 一个完全自定义的 PyTorch 网络结构 `Renaissance`，在**复数 STFT 频谱域**上直接估计干净人声的频谱。
3. 一对轻量的 STFT / iSTFT 封装，负责时域与频域的相互转换。

模型工作在频谱域，核心思路是：**按 Mel 频带切分子带 → 提取子带特征 → 在“频带维”和“时间维”上交替建模 → 合成各子带频谱 → 拼接回完整频谱**。

仓库结构极简（4 个源文件），没有任何训练代码，只有推理链路。

---

## 2. 目录与文件结构

```
/workspace
├── main.py            # 命令行入口 + 推理管线（预处理 / 前向 / 后处理 / 保存）
├── model.py           # 网络结构定义（Renaissance 及全部子模块）
├── spectral_ops.py    # STFT / iSTFT 封装
├── requirements.txt   # Python 依赖
├── README.md          # 快速开始与项目简介
└── LICENSE
```

| 文件 | 职责 | 是否可独立运行 |
|------|------|----------------|
| `main.py` | 用户入口，串联数据加载、模型推理、音频保存 | 是（需 checkpoint） |
| `model.py` | 定义所有 `nn.Module`：`Renaissance`、`BSNet`、`Roformer`、`ICB`、`ConvBlock`、`RMSNorm` 等 | 否（被 import） |
| `spectral_ops.py` | `STFT` / `iSTFT` 类，包装 `torch.stft` / `torch.istft` | 否（被 import） |
| `requirements.txt` | 依赖清单 | — |

---

## 3. 整体架构

### 3.1 端到端推理链路

```
输入音频文件
    │  torchaudio.load
    ▼
[预处理]                      main.load_and_preprocess_audio()
  多声道 → 单声道 (mean)
  重采样 → 48000 Hz
  高通滤波 60 Hz (highpass_biquad)
    │
    ▼
[峰值归一化]                   main.normalize_audio() → norm_factor
    │
    ▼
[STFT]  n_fft=4096 hop=2048   spectral_ops.STFT
  时域 [1, L] → 频谱 [1, 2049, T, 2] (实部/虚部)
    │
    ▼
[Renaissance 前向]             model.Renaissance.forward()
  频谱 [B, 2049, T, 2] → 增强频谱 [B, 2049, T, 2]
    │  (CUDA 时用 autocast 半精度)
    ▼
[iSTFT]                       spectral_ops.iSTFT
  频谱 [1, 2049, T, 2] → 时域 [1, L]
    │
    ▼
[反归一化] × norm_factor
    │
    ▼
torchaudio.save  →  输出 wav (48 kHz, FP32)
```

### 3.2 模型内部结构（Renaissance）

```
输入频谱 spec: [B, 2049, T, 2]
    │
    ▼  spec_band_split()  —— 按 Mel 频带宽度切分为 80 个子带
    │
    ▼  feature_extraction()  —— 每个子带独立提取特征
    │     功率归一化 RI + log 功率 → RMSNorm + Conv1d(1x1)
    ▼
features: [B, 80 (nband), 128 (feature_dim), T]
    │
    ▼  9 × BSNet（全局残差包裹整段堆叠）
    │     ┌─────────────────────────────────────────┐
    │     │  BSNet = band_net(Roformer) + seq_net(ICB)│
    │     │   · band_net：沿【频带维】做 RoPE 注意力   │
    │     │   · seq_net ：沿【时间维】做空洞卷积       │
    │     └─────────────────────────────────────────┘
    ▼
processed + residual_features   （全局残差）
    │
    ▼  每个子带独立输出合成 output_layers (GLU → 实部/虚部)
    ▼  拼接所有子带
    ▼
输出频谱 est_spec_full: [B, 2049, T, 2]
```

### 3.3 双维度建模的核心设计

`BSNet` 是本模型的关键，它让同一组特征交替地在**频率轴**和**时间轴**上建模：

- **band_net（频带维）**：把每个时间帧 tutti 的所有频带视作一条序列，用 `Roformer` 做自注意力，让不同频带间共享信息（谐波、共振关系）。
- **seq_net（时间维）**：把每个频带视作一条独立序列，用 `ICB`（膨胀深度可分离卷积）沿时间建模（音高连续性、瞬态）。

两者交替堆叠 9 层，逐步细化频谱估计。

---

## 4. 数据流与张量形状

下表汇总关键张量的形状（`B`=batch，`T`=时间帧数，`nband`=80，`N`=feature_dim=128）。

| 阶段 | 张量 | 形状 |
|------|------|------|
| 时域输入 | waveform | `[B, L]` |
| STFT 输出 | input_spec | `[B, 2049, T, 2]`（最后一维 = 实部/虚部） |
| 子带切分 | subband_spec_ri (list) | `nband` 个 `[B, band_width_i, T, 2]` |
| 子带功率 | subband_power | `[B, nband, T]` |
| 特征提取输出 | features | `[B, nband, N, T]` |
| BSNet 内部（频带建模） | band_input | `[B*T, N, nband]` |
| BSNet 内部（时间建模） | seq_input | `[B*nband, N, T]` |
| 主网络输出 | processed | `[B, nband, N, T]` |
| 输出合成（单子带） | band_output | `[B, band_width_i*2, T]` → `[B, band_width_i, T, 2]` |
| 最终输出 | est_spec_full | `[B, 2049, T, 2]` |

> 说明：`band_width` 是一个长度 80 的整数列表，其元素之和恰好等于 2049（等于 `n_freqs`）。它由 Mel 刻度生成（见 §5.1）。

---

## 5. 模块详解

### 5.1 `model.py :: Renaissance`（主模型，第 280–437 行）

顶层容器，负责：Mel 频带划分、特征提取、主网络堆叠、输出合成。

**构造参数（默认值）**

| 参数 | 默认 | 含义 |
|------|------|------|
| `n_freqs` | 2049 | STFT 频点数（= n_fft/2 + 1，n_fft=4096） |
| `feature_dim` | 128 | 每个子带特征的通道数 `N` |
| `layer` | 9 | BSNet 层数 |
| `sample_rate` | 48000 | 采样率 |
| `dilation_start_layer` | 3 | 从第几层开始引入膨胀卷积（之后按 2 的幂增长，上限 4） |
| `n_bands` | 80 | Mel 子带数量 |
| `num_heads` | 16 | Roformer 注意力头数 |
| `max_seq_len` | 10000 | 最大序列长度 |
| `band_rope_base` / `temporal_rope_base` | 10000 | RoPE 基频 |

**关键方法**

| 方法 | 行号 | 说明 |
|------|------|------|
| `_init_weights` | 315–319 | 对 `Linear` / `Conv1d` 使用 Kaiming 初始化，bias 置零 |
| `_generate_mel_bandwidths` | 321–350 | 生成 Mel 频带宽度列表，并修正余数使宽度之和 == `enc_dim`(2049) |
| `_build_feature_extractor` | 352–356 | 为每个子带构建 `RMSNorm + Conv1d(1x1)` |
| `_build_main_network` | 358–374 | 堆叠 `layer` 个 `BSNet`，膨胀率随层数递增 |
| `_build_output_synthesis` | 376–385 | 为每个子带构建输出头 |
| `spec_band_split` | 387–400 | 切分子带，并计算每个子带的 L2 功率（幅度） |
| `feature_extraction` | 402–416 | 功率归一化 + 对数功率拼接 → 逐子带特征提取 |
| `forward` | 418–437 | 主前向：提特征 → 9 层 BSNet → 全局残差 → 逐子带合成 → 拼接 |

**Mel 频带划分逻辑（`_generate_mel_bandwidths`）**
- 使用标准 Mel 变换：`hz_to_mel(hz)=2595*log10(1+hz/700)`，`mel_to_hz` 为其逆。
- 在 `[0.1 Hz, sr/2]` 之间取 `n_bands+1 = 81` 个 Mel 等分点，换算回 Hz 求相邻差值。
- 差值 ÷ 频率分辨率（`bin_width = sr/2/enc_dim`），四舍五入得到每个频带占用的 bin 数，最小为 1。
- 若总和与 2049 有偏差，则按宽度从小到大（或从大到小）逐个 ±1 补偿，最终保证 `sum(band_width) == 2049`。
- 低频段窄（少量 bin）、高频段宽（大量 bin），符合 Mel 的心理声学刻度。

**特征提取逻辑（`feature_extraction`）**
对第 `i` 个子带：
1. `power_for_norm` = 该子带的 L2 功率 `[B,1,1,T]`。
2. 用功率归一化子带复数谱（实部/虚部分别除以功率），得到相位/相对结构信息。
3. 展平为 `[B, band_width*2, T]`，再在通道维拼接 `log(power)` 得到 `[B, band_width*2+1, T]`。
4. 送入该子带专属的 `RMSNorm + Conv1d(1x1) → feature_dim`。
5. 所有子带结果 `stack` 成 `[B, nband, N, T]`。

**前向逻辑（`forward`）**
1. `features = feature_extraction(input_spec)`
2. `residual_features = features`（全局残差）
3. 依次通过 9 个 `BSNet`
4. `processed = processed + residual_features`
5. 每个子带经 `output_layers[i]` 合成 `[B, band_width, T, 2]`，拼接为 `[B, 2049, T, 2]`

> 注意：模型**直接回归增强后的频谱**（而非预测掩码或残差），因此输出不再与输入频谱相加。

---

### 5.2 `model.py :: BSNet`（第 246–278 行）

Band-Sequence Network，模型的主干单元，实现“频带 / 时间”双维交替建模。

- `band_net = Roformer(feature_dim, feature_dim, num_head=num_heads, window=max_bands, causal=False)` — 在频带维做非因果自注意力。
- `seq_net = ICB(...)` — 在时间维做膨胀深度可分离卷积。

**前向（输入 `[B, nband, N, T]`）**
1. `band_input = input.permute(0,3,2,1).reshape(B*T, N, nband)` — 把 `T` 折进 batch，序列长度 = `nband`。
2. `band_output = band_net(band_input)`，再 reshape 回 `[B, nband, N, T]`。
3. `seq_input = band_output.reshape(B*nband, N, T)` — 把 `nband` 折进 batch，序列长度 = `T`。
4. `seq_output = seq_net(seq_input)` → `[B, nband, N, T]`。

---

### 5.3 `model.py :: Roformer`（第 126–206 行）— 实际使用的注意力模块

一个基于 **RoPE（旋转位置编码）+ 自注意力 + SwiGLU 风格 MLP** 的模块，全部用 `Conv1d(kernel=1)` 实现（等价于在特征维做线性投影，在最后一维序列上独立）。

**构造**：`Roformer(input_size, hidden_size, num_head=8, theta=10000, window=10000, input_drop=0., attention_drop=0., causal=True)`
- `self.hidden_size = hidden_size // num_head`（即每个头的维度 head_dim）。
- `weight`：`Conv1d(input_size → head_dim*num_head*3)`，一次生成 Q/K/V。
- `output`：`Conv1d(head_dim*num_head → input_size)`。
- `MLP`：`RMSNorm → Conv1d(input_size → input_size*8) → SiLU`。
- `MLP_output`：`Conv1d(input_size*4 → input_size)`（配合 GLU 门控）。
- 预计算并注册 `cos_freq` / `sin_freq` RoPE 缓存，长度 = `window`。

**关键方法**

| 方法 | 行号 | 说明 |
|------|------|------|
| `_calc_rotary_emb` | 155–164 | 预计算 RoPE 的 cos/sin 表 |
| `_add_rotary_emb` | 166–177 | 基于单一位置 `pos` 施加 RoPE（**当前未被调用**） |
| `_add_rotary_sequence` | 179–189 | 沿序列维施加 RoPE（`forward` 实际使用） |
| `forward` | 191–206 | Q/K/V 投影 → RoPE → SDPA → 输出投影 + 残差 → GLU-MLP + 残差 |

**前向数据流（输入 `[B, N, T_seq]`）**
1. `weight` 生成 `[B, num_head, T_seq, head_dim*3]`，`split` 得到 Q/K/V。
2. 对 Q、K 施加 `_add_rotary_sequence`。
3. `F.scaled_dot_product_attention(Q, K, V, is_causal=self.causal)`。
4. 输出投影 + 残差 → `output`。
5. `gate, z = MLP(output).chunk(2)`；`output + MLP_output(SiLU(gate) * z)`（门控 MLP）。

> `_add_rotary_sequence` 通过 `flip` + 符号翻转实现 RoPE 的旋转配对，作用于序列维（最后一个维度）。

---

### 5.4 `model.py :: ICB`（第 230–243 行）与 `ConvBlock`（第 208–227 行）

**`ConvBlock`**：门控深度可分离卷积块
- `dwconv`：深度卷积（`groups=channels`），带 `dilation`，`padding=(k-1)*dilation//2`（保持长度）。
- `RMSNorm` → `pwconv1`（通道 ×4）→ `GLU(dim=1)`（通道减半 → ×2）→ `pwconv2`（→ 原通道数）。

**`ICB`**：Inverted/Inception-style Convolution Block
- 串联三个 `ConvBlock`：`block1`(dilation=1) → `block2`(dilation=dilation_rate) → `block3`(dilation=1)。
- `gamma`：LayerScale 参数（初始化 `1e-6`），输出 `x * gamma + residual`。

---

### 5.5 `model.py :: RMSNorm`（第 7–17 行）

自定义的 Root Mean Square LayerNorm。
- 权重形状为 `[dimension]`，在 `forward` 中 `unsqueeze(0).unsqueeze(-1)` 以适配两种张量布局：
  - 通道在第一维（如 `Conv1d` 输出 `[B, C, T]`）；
  - 通道在中间维（如 `[B, N, T, ...]` 的频带特征）。
- 内部用 `half()` 计算方差以省显存，最后 `type_as(input)` 还原精度。

---

### 5.6 `model.py :: RoformerLayer` / `RotaryEmbedding` / `rotate_half`（第 20–124 行，未在主模型中启用）

这三个定义是**另一套基于 `nn.Linear` 的注意力实现**，包含：
- `RotaryEmbedding`：标准 HF 风格的 RoPE 缓存（`inv_freq`、`cos_cached`、`sin_cached`）。
- `rotate_half`：RoPE 的半维旋转辅助函数。
- `RoformerLayer`：RMSNorm + `qkv_proj` + RoPE + SDPA + SwiGLU MLP 的标准 Transformer 层。

⚠️ **注意**：经全局检索，`RoformerLayer` 在本仓库中**没有被任何模块实例化**；`BSNet` 使用的是 `Roformer`（§5.3）。这些类属于保留的备选/实验实现，阅读代码时不要误以为它们是主干。

---

### 5.7 `spectral_ops.py`（第 1–33 行）

**`STFT`**
- 构造：`n_fft`、`hop_length`、`win_length`，内部持有一个 `hann_window`。
- `__call__(y)`：`torch.stft(..., return_complex=False, center=True, pad_mode='reflect')`，返回**非复数张量**，形状 `[B, n_fft/2+1, T, 2]`（末维为 `[real, imag]`）。
- 每次调用把窗口搬到输入所在设备（避免设备/精度不匹配）。

**`iSTFT`**
- `__call__(X)`：先用 `torch.view_as_complex(X.contiguous())` 把末维为 2 的实数张量转回复数，再 `torch.istft(..., center=True)` 还原时域波形。

> 之所以用 `return_complex=False` + `view_as_complex`，是为了让模型直接以实数张量（实部/虚部通道）处理频谱，便于各种卷积/归一化算子。

---

### 5.8 `main.py`（第 1–113 行）

| 函数 | 行号 | 说明 |
|------|------|------|
| `load_and_preprocess_audio` | 8–26 | 读音频 → 转单声道 → 重采样至 48 kHz → 60 Hz 高通 → 搬至设备与精度 |
| `normalize_audio` | 28–34 | 峰值归一化，返回归一化波形与归一化因子 |
| `process_audio` | 37–51 | 归一化 → no_grad → STFT → autocast(若 CUDA) → 模型 → iSTFT → 乘回归一化因子 |
| `main` | 54–109 | 参数解析、设备/精度选择、加载模型、构建 STFT/iSTFT、跑管线、保存 |

**CLI 参数**

| 参数 | 必需 | 默认 | 说明 |
|------|------|------|------|
| `input`（位置参数） | 是 | — | 输入音频路径 |
| `-o` / `--output` | 否 | `{input_stem}_enhanced.wav`（同目录） | 输出路径 |
| `-c` / `--checkpoint` | 是 | — | 模型权重 `.pt` 路径 |

**设备与精度策略**
- 有 CUDA → `torch.float16` + `autocast`；否则 CPU → `torch.float32`。
- 模型先 `.to(device).to(dtype)` 再 `load_state_dict(torch.load(..., map_location=device))`，最后 `eval()`。

**STFT 配置**：`n_fft=4096, hop_length=2048, win_length=4096`（与模型的 `n_freqs=2049` 严格对应）。

---

## 6. 关键类与函数速查表

| 名称 | 所在 | 一句话职责 |
|------|------|-----------|
| `Renaissance` | `model.py` | 顶层网络：Mel 分带 → 特征提取 → 9 层 BSNet → 逐带合成 |
| `BSNet` | `model.py` | 频带维（Roformer）+ 时间维（ICB）交替建模单元 |
| `Roformer` | `model.py` | RoPE 自注意力 + 门控 MLP（Conv1d 实现，实际使用） |
| `ICB` | `model.py` | 3 个膨胀深度卷积块 + LayerScale 残差（时间维建模） |
| `ConvBlock` | `model.py` | 深度卷积 + RMSNorm + GLU 门控点卷积 |
| `RMSNorm` | `model.py` | 自定义 RMS 归一化 |
| `RoformerLayer` / `RotaryEmbedding` / `rotate_half` | `model.py` | 备选 RoPE 注意力实现（**未启用**） |
| `STFT` / `iSTFT` | `spectral_ops.py` | 时频转换封装（实部/虚部表示） |
| `load_and_preprocess_audio` | `main.py` | 音频单声道化 / 重采样 / 高通 |
| `normalize_audio` | `main.py` | 峰值归一化 |
| `process_audio` | `main.py` | 单条音频的完整推理流程 |
| `main` | `main.py` | CLI 入口与设备/精度选择 |

---

## 7. 依赖关系

### 7.1 外部依赖（`requirements.txt`）

| 依赖 | 版本 | 用途 |
|------|------|------|
| `torch` | 2.7.1 | 张量计算、`scaled_dot_product_attention`、`stft`/`istft` |
| `torchaudio` | 最新 | 音频 I/O、`Resample`、`highpass_biquad` |
| `numpy` | 2.2.6 | Mel 频带宽度计算（`np.log10`、`np.linspace` 等） |
| `argparse` | — | CLI 参数解析（标准库，无需安装） |

Python 版本建议 3.10（README 中使用 `uv venv --python=3.10`）。

### 7.2 模块间依赖

```
main.py
  ├── imports spectral_ops.STFT, iSTFT
  └── imports model.Renaissance

model.py
  ├── torch / torch.nn / torch.nn.functional
  └── numpy（仅在 Mel 带宽计算中）

spectral_ops.py
  └── torch
```

**无循环依赖**。层级关系：`main.py`（应用层） → `model.py` / `spectral_ops.py`（领域层） → PyTorch/NumPy（基础库）。

### 7.3 运行时外部资源

- 需要在根目录放置模型权重 `smule-renaissance-small.pt`（从 HuggingFace 下载，见 §8）。

---

## 8. 项目运行方式

### 8.1 环境搭建

```bash
# 创建虚拟环境（Python 3.10）
uv venv cleanup --python=3.10
source cleanup/bin/activate
uv pip install -r requirements.txt
```

### 8.2 下载模型权重

```bash
wget https://huggingface.co/smulelabs/Smule-Renaissance-Small/resolve/main/smule-renaissance-small.pt
```

### 8.3 运行推理

```bash
python main.py {path-to-input} -o {path-to-output} -c {path-to-checkpoint}
```

示例：

```bash
python main.py degraded_vocal.wav \
  -c smule-renaissance-small.pt \
  -o enhanced.wav
```

若不指定 `-o`，输出默认为输入同目录下的 `{输入文件名}_enhanced.wav`。

### 8.4 运行产物

- 输出 WAV，48 kHz、单声道、FP32。
- 终端会打印：使用的设备与精度、音频时长、各阶段进度与 “Done!”。

---

## 9. 模型配置与关键超参数

| 参数 | 值 | 影响 |
|------|-----|------|
| STFT `n_fft` / `hop` / `win` | 4096 / 2048 / 4096 | 频点 2049、帧移 2048 |
| 采样率 | 48 kHz | 覆盖至 24 kHz 奈奎斯特频率 |
| `n_bands` | 80 | Mel 子带数量 |
| `feature_dim` | 128 | 子带特征通道数 |
| `layer` | 9 | BSNet 层数 |
| `num_heads` | 16 | 频带注意力的头数 |
| `dilation_start_layer` | 3 | 膨胀卷积起始层 |
| 膨胀率序列 | `[1,1,1,2,4,4,4,4,4]` | 由 `min(2^max(0,i-2), 4)` 生成 |
| 参数量 | ~10.4M | 轻量、可移动端实时 |

---

## 10. 注意事项 / 已知细节

1. **未启用代码**：`model.py` 中的 `RoformerLayer`、`RotaryEmbedding`、`rotate_half` 以及 `Roformer._add_rotary_emb` 均未在主推理链路中被调用（备选/实验实现）。
2. **`torch.cuda.amp.autocast`**：`main.py` 使用的是旧式 API，在较新 PyTorch 中已被 `torch.amp.autocast(device_type='cuda')` 取代，但当前版本仍可正常工作。
3. **单声道信息打印**：`load_and_preprocess_audio` 中“Converted to mono from N channels”打印的 `waveform.shape[0]` 是在已取均值之后的值（恒为 1），属于提示信息不精确，不影响功能。
4. **模型直接回归频谱**：`Renaissance.forward` 返回的是合成频谱本身，未与输入频谱相加（输出头即为最终结果）。
5. **设备自适应**：窗口张量、精度在每次 `STFT`/`iSTFT` 调用时都会对齐到输入的设备/精度，CPU 与 CUDA 均可运行。
6. **无训练代码**：本仓库仅含推理，不含数据加载、损失函数或优化器。

---

*本文档基于仓库当前代码（`main.py`、`model.py`、`spectral_ops.py`、`requirements.txt`、`README.md`）静态分析生成。*
