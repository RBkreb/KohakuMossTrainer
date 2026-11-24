<p align="right">
  <a href="./README_EN.md" title="English"><img src="https://img.shields.io/badge/Language-English-blue?logo=github" alt="English"></a>
</p>

# KohakuMossTrainer · NaRA / SORA / EVA / BlockSkip fork

[KohakuBlueleaf/KohakuMossTrainer](https://github.com/KohakuBlueleaf/KohakuMossTrainer)

| 新增 | 简介 |
| --- | --- |
| **NaRA** | Noise-aware Rank Adaptation：低秩增量随去噪噪声水平变化 |
| **SORA** | Skewed Orthogonal Rotation Adaptation：对角标定 + 近正交旋转 | 
| **EVA** | Explained Variance Adaptation：用激活 SVD 初始化输入侧因子 |
| **BlockSkip** | DiT-BlockSkip：动态 patch 采样 + 跳层残差预计算，**有质量损失** |


## 新增内容

### NaRA

- **NARA 开关**：`NARA = True` 像 `DORA` 一样可以作用于 lora / locon / loha / lokr
- lora / locon 等于 `ALGO = "nara"`

  LoHa 的两个秩通道读同一个核心

  LoKr 的核心在 `w2` 的秩收缩中
- **NaRA × DoRA 组合**：归一化在 `BA·C(sigma)` 之后

  LoRA / LoKr 用 Gram 展开 `||W + dW||² = ||W||² + 2<W, dW> + ||dW||²`
- **训练 / 采样**：每步自动加入 `(sigma, seq)`；
  采样器在接收到 `needs_sigma` 时自动开启动态求值
- **导出**：`lycoris` 格式保留完整 NaRA

  `comfy` / `merged` 在 `export_sigma`（默认 0.5）将核心冻结导出

*建议使用 lycoris 导出*

### SORA

可单独或同时开启：

- **`SORA_PRE_DIAG`**（Pre-Diag）：base 输出的逐输出通道可学习缩放，
  DoRA 的廉价替代。**与 `DORA` 互斥**
- **`SORA_POST_ORTHO`**（Post-Ortho）：对 整个输出施加近正交旋转，由低秩斜对称矩阵的一阶 Taylor 展开构成，参数极省。可与 DoRA 叠加。
- **导出精度**：`sora_diag` 的偏离量通常只有 ~1e-3 量级，bf16 有 83% 的值退化。要保证 Pre-Diag 质量请导出 **fp32**。

`Pre-Diag+Post-Ortho=SORA`

```
W' = R (diag(d) W_pre + s · BA)
```
*建议使用 lycoris 导出*
### EVA
```
EVA = {
    "batches": 32,       # batch of one epoch
    "min_batches": 4,
    "sim_thresh": 0.99,
    "precision": "fp32",  # 8x time cost on fp64
}
```
训练开始前用真实训练数据的**输入激活**做主元素分析，将适配器的输入侧因子初始化为大致方向。提升收敛起点。

- 共享输入的层（q / k / v）由激活指纹识别，只校准一次。
- `ALGO` 必须支持 EVA（LoRA / LoCon、LoKr 及其 NaRA 变体）；
  `DROPOUT['module']` 必须为 0。

### DiT-BlockSkip

- **动态 patch 采样** `RESOLUTION_MODE = "patch"` ：所有样本在固定低分辨率
  `smin × smin` 训练；裁剪边长随该样本的 sigma 从 `smin` 线性增大到 `smax`，sigma 在 host 规划期按种子预采样并随批携带。
- **跳层 + 残差预计算** `BLOCKSKIP` ：跳过前 n + 后 m 个 joint block。前段特征冻结精确、直接缓存绝对值；后段缓存残差 `Δ = f_L − f_{L−m}`。预计算与训练两阶段共用数据 / sigma / 噪声流，每个 micro-batch 校验 sigma 对齐。

> ⚠️ **质量警示：BlockSkip 是有损近似。**
> 原论文（FLUX，DreamBooth）主体保真度 DINO：LoRA 0.7324 → 跳层 30% 0.7194（−1.8%）→ 跳层 50% **0.6963（−4.9%）**；CLIP-I 0.8146 → 0.7877（−3.3%）。
> 本复现用比例映射选层，损失可能更大。
> **生产前务必自行做质量对比**。

## 用法

### 训练

NaRA + DoRA：

```python
ALGO = "lora"            # "lora" | "locon" | "loha" | "lokr"
NARA = True
DORA = True

ADAPTER_MODE = "bypass"  # NaRA 只支持 bypass
SAVE_FORMAT = "both"
```

SORA（Pre-Diag + Post-Ortho）：

```python
ALGO = "lora"
DORA = False 
SORA_PRE_DIAG = True
SORA_POST_ORTHO = {"rank": 1, "initial_s_p": 1.0}   # 或 SORA_POST_ORTHO = True
ADAPTER_MODE = "bypass"
DROPOUT = {"input": 0.0, "rank": 0.0, "module": 0.0}
```

EVA：

```python
EVA = {"batches": 32, "min_batches": 4, "sim_thresh": 0.99, "precision": "fp32"} # 或{},以默认值启用EVA
```

`RESUME` / `INIT_FROM` 会跳过校准。

BlockSkip：

```python
RESOLUTION_MODE = "patch"                 # 动态 patch 采样
PLANNER_KWARGS = {"smin": 512, "smax": 1024}

# 阶段 1：预计算
BLOCKSKIP = {"mode": "precompute", "skip": 0.5, "cache": "path/to/features"}
SAVE_FORMAT = "none"

# 阶段 2：训练
BLOCKSKIP = {"mode": "train", "skip": 0.5, "cache": "path/to/features"}
```

两阶段的 `SEED` / `DATASETS` / `STEPS` / `GRAD_ACC` / `TIMESTEP` / 分辨率设置必须一致。

静态质量检查（`QUALITY_CHECK`）：

```python
QUALITY_CHECK = True   # 或 {"tau": 3.0, "tau_low_rank": 3.5, "ema": False, "top": 5}
```

每次写 checkpoint（以及无 final checkpoint 时的收尾导出）对保存的 `Wdown` 因子计算
SVIR（奇异值侵入比）与 EK（超额峰度）：谱能量集中 → 中/重度过拟合；EK 由负转正 →
轻度过拟合重尾。结果逐次追加到 `<output>/quality.jsonl` 并打一行日志；只支持带
`lora_down` 因子的算法（LoHa / LoKr / OFT 等会自动跳过）。252 层 rank-32 适配器约
0.3 秒，纯 CPU。阈值在 SDXL / FLUX / Kolors 社区 LoRA 上标定，MOSS 上建议看同一
run 内的跨 checkpoint 趋势，细节见 [docs/internals/quality.md](docs/internals/quality.md)。

## 详细参数
### NaRA 超网络参数（`ALGO_KWARGS`）

| key | 默认 | 含义 |
| --- | --- | --- |
| `nara_hidden1`, `nara_hidden2` | 256, 512 | mapper 隐藏层宽度 |
| `nara_embed_dim` | 64 | 嵌入宽度 |
| `nara_embed` | `fourier` | `fourier` / `mlp` / `raw` |
| `nara_fourier_scale` | 16.0 | 固定 Fourier 嵌入的频率尺度 |
| `nara_c_scale` | 1.0 | 核心残差尺度 eta |
| `export_sigma` | 0.5 | 固定 sigma 导出所用的噪声水平 |

### SORA 参数

| key | 默认 | 含义 |
| --- | --- | --- |
| `SORA_POST_ORTHO` | `0` | `r_p`，旋转的半秩（斜对称项秩 2·r_p）|
| `SORA_POST_ORTHO["initial_s_p"]` | `1.0` | 斜对称项的上界；`s_p = min(1, s_p0 / (‖C‖+‖D‖))` |
| `SORA_PRE_DIAG` | `False` | 逐输出通道的 base 缩放 |

`r_p` 的敏感性：`r_p=1` 训练最快、精度最低但差距小；
加大 `r_p` 收益递减而推理成本单调上升。

### EVA 参数

| key | 默认 | 含义 |
| --- | --- | --- |
| `batches` | 32 | 校准前向的上限批次 |
| `min_batches` | 4 | 下限批次 |
| `sim_thresh` | 0.99 | 相邻两次增量 SVD 的逐分量余弦相似度判定校准成功的阈值 |
| `precision` | `"fp64"` | `"fp64"` / `"fp32"`|

### BlockSkip 参数

| key | 默认 | 含义 |
| --- | --- | --- |
| `BLOCKSKIP["mode"]` | `"train"` | `"precompute"`（预计算）/ `"train"`（跳层训练） |
| `BLOCKSKIP["skip"]` | `0.5` | 跳层比例，或显式 `(n, m)`；比例按 n ≈ 0.6k 前 / m ≈ 0.4k 后拆分 |
| `BLOCKSKIP["cache"]` | 必填 | 特征目录（每步一个文件，bf16） |
| `PLANNER_KWARGS["smin"]` | 512 | patch 训练分辨率（`RESOLUTION_MODE="patch"`） |
| `PLANNER_KWARGS["smax"]` | 1024 | sigma=1 时的裁剪边长上限 |

## 推理
### ComfyUI 推理

[推理节点](https://github.com/RBkreb/comfyui-moss-nara)

**含 SORA 的 lycoris 权重必须使用插件加载**
## 限制

### 存储：以`comfy`形式导出含SORA层的权重 

SORA 层导出到 `comfy` 格式时**写入精确稠密 diff**（`convert.comfy`）——
旋转和 base 缩放作用于 `W` 本身。规模高达 d² 级。按 MOSS-Image 的实际形状（42 层，`dim=3072`，
`ffn=8192`）计算：

| 预设 | target 数 | `comfy` 导出体积（bf16） |
| --- | --- | --- |
| `attn` | 168 | ≈ 2.95 GiB / 检查点 |
| `attn-mlp` | 252 | ≈ 8.86 GiB / 检查点 |

- 建议在使用SORA时`SAVE_FORMAT = "hf"`
- `SAVE_MERGED` 不可用。

### 计算：EVA  `precision` 默认为 fp64

`torch-incremental-pca`和`torch.svd_lowrank`位于fp64计算，故 QR 和 GEMM 都位于 fp64
- 对于消费级显卡，建议使用fp32

| `precision` | 行为 | 成本 |
| --- | --- | --- |
| `"fp64"`（默认） | pca包原行为 | 约 fp32 10倍时间成本 |
| `"fp32"` | `svd_lowrank` 使用 fp32 | 较快 |

EVA 的成本是 `MAX_TOKENS × batches × 层数`，**与数据集大小和步数无关**。

fp32精度损失：固定种子下 fp32/fp64 的最大角度差为 0.10。

### 计算：Pre-Diag 的成本取决于 delta 的代价

Pre-Diag 在每次前向计算时都需要 delta 才能做逐通道标定，
**性价比取决于 delta 的代价。**

- LoRA 的 delta 只有两次 `F.linear`，廉价。
- NaRA 的 delta 是超网络的函数：`C(sigma)` 每层、每次前向都要跑一次 mapper，
  再加两次 `F.linear` 和一次逐 token 的 `bmm`。同一个 attn 形状下，光 hyper
  一项就约等于一次 delta body 的 80%。

所以 **NaRA + Pre-Diag 的步时必然高于 LoRA + Pre-Diag**：前者是后者的严格超集，
多出来的部分不会让任何其他环节变便宜。

RTX 3090，attn 3072×3072 / 12288 token，fwd+bwd，bf16，单层合成基准

| 组合 | 对比纯 LoRA |
| --- | --- |
| LoRA + Pre-Diag | 1.75× |
| LoRA + Post-Ortho | 2.28× |
| LoRA + SORA（两者同开） | 3.05× |

旋转（Post-Ortho）是主要成本：它要把整个 `[tokens, out]` 激活读两次、写两次。

### 质量与对齐：BlockSkip

- **有损且未评估**：见上文质量警示（论文跳层 50% DINO −4.9%）。
- 预计算与训练两阶段的 `SEED` / `DATASETS` / caption / dropout / `STEPS` / `GRAD_ACC` /
  `TIMESTEP` / 分辨率设置必须完全一致。
- 拒绝 `PARALLEL != "none"`、`ALGO = "full"`、`STREAM_LOSS` / `CONCENTRATION_LOSS`。
- **EVA 可与 BlockSkip 共存**：两阶段都会把 `PRESET` 限定到保留的中间 block，EVA
  只校准这些层；两阶段的 EVA 参数必须相同（EVA 校准消耗的数据 / 随机数流随之一致，
  逐 micro-batch sigma 校验兜底）。无 BlockSkip 时 EVA 行为不变。
- toml 形式的 `PRESET` 不会被自动限定，需自行把 block 正则限制到保留区间。
- 特征盘占用：每步 `2 × N_joint × 3072 × 2 B`（bf16）；512² 单样本 micro-batch 约 17 MB。
- `Throughput` 的 TFLOP/s 按全层估算，跳层时偏高；step_time / tokens/s / VRAM 不受影响。

### 其他

- 拒绝 `ADAPTER_MODE = "merge"`（NaRA 的 sigma 相关 delta 无法折进一个权重）
- 开启 NaRA 时 `rank dropout` / `use_tucker` / `use_scalar` / `rs_lora` 被拒绝
- 共享超网络不能跨 FSDP 分片或流水线阶段：fsdp / pipeline / threaded 布局被拒绝
- LoKr 变体要求 `w2` 有秩可放核心：`w2` 满秩的布局在构造时被拒绝
  （需降低 `DIM` 或调大 `ALGO_KWARGS["factor"]`）
- LoHa 变体的 bypass 逐样本重建稠密增量，建议配合 block checkpointing 档位
- Pre-Diag 要求 `DROPOUT['input'] = 0` 且 `DROPOUT['rank'] = 0`
- SORA 的 lycoris 导出遇**逐行** fold 缩放不可用（`R·S ≠ S·R`，因子集里没有对应项）：
  均匀 fold 可精确导出（`R(cW) = cR(W)`），逐行 fold 会按层拒绝。
  默认 `PRESET` 下逐行层数为 0。
- EVA 的收敛日志（`N/M converged`）在退化子空间上会偏低：逐分量余弦会在 ±1 之间摆动，这是正常现象。

## 原理

## NaRA

NaRA 让适配器的低秩增量随去噪噪声水平变化。两个静态 LoRA 因子之间放一个动态核心：

```
h = W0 x + s · B C(sigma) A x
C(sigma) = I_r + c_scale · F(e_sigma)
```

- `F` 是一个小型 MLP，`e` 是噪声水平的高斯 Fourier 嵌入；
  **所有被适配层共享同一个超网络**，每层只保存 A（`lora_down`）/ B（`lora_up`）。
- mapper 末层零初始化，初始 `C = I`，NaRA 退化为普通 LoRA。
- 条件化按 packed batch 的**每个样本**进行（sigma `[B]` -> 核心 `[B, r, r]`，fp32），
  沿层的 token 轴展开，与 T-LoRA 的 掩码机制一致。

## SORA

SORA 给整层的权重加正交且可学习的条件化。

- **Pre-Diag**：`W = W_pre D + s·BA`。将 D 从"每次前向都要重算列范数的测量量"转化为"自由参数"。
- **Post-Ortho**：`W' = (W_pre D + s·BA) P`，`P ≈ I + s_p(D_p C_pᵀ − C_p D_pᵀ)`。正交变换形成范数。乘法被重构为`(W D_p) C_pᵀ − (W C_p) D_pᵀ`，复杂度从 `O(mn²)` 降到 `O(mn r_p)`。

## EVA

EVA 更改训练起步初始值。

- 每层一个 `SVDHook`，对层输入的 token 行做增量 PCA。
- LoRA / LoCon / NaRA：`lora_down = components[:DIM]`，`lora_up = 0`。
- LoKr / NaRA-LoKr：取块 Gram 矩阵的顶部特征向量（Kronecker 约束下的主导方向）；`w2_a = 0`，`w1` 保持随机初始化。

## 样例

[DoRA+NaRA](https://huggingface.co/RBXLSP/TextureLux-DoRAN)
[Pre-diag](https://huggingface.co/RBXLSP/RigidFrontier-SORA)

## 引用

### SORA — Skewed Orthogonal Rotation Adaptation

```bibtex
@inproceedings{chang2026sora,
  title     = {Calibrating and Rotating: A Unified Framework for Weight Conditioning in PEFT},
  author    = {Chang, Da and Xue, Peng and Li, Yu and Liu, Yongxiang and Xu, Pengxiang and Zhang, Shixun},
  booktitle = {The Fortieth AAAI Conference on Artificial Intelligence (AAAI-26)},
  year      = {2026}
}
```

- 论文：<https://arxiv.org/abs/2511.00051>
- 仓库：<https://github.com/MaeChd/SORA>

### EVA — Explained Variance Adaptation

```bibtex
@article{paischer2024eva,
  title   = {One Initialization to Rule them All: Fine-tuning via Explained Variance Adaptation},
  author  = {Paischer, Fabian and Hauzenberger, Lukas and Schmied, Thomas and Alkin, Benedikt and Deisenroth, Marc Peter and Hochreiter, Sepp},
  journal = {arXiv preprint arXiv:2410.07170},
  year    = {2024}
}
```

- 论文：<https://arxiv.org/abs/2410.07170>
- 仓库：<https://github.com/ml-jku/EVA>

### NaRA — Noise-Aware Rank Adaptation

```bibtex
@inproceedings{nara2026,
  title={Na{RA}: Noise-Aware Lo{RA} for Parameter-Efficient Fine-Tuning of Diffusion {LLM}s},
  author={Shuaidi Wang and Zhan Zhuang and Ruping Huang and Yu Zhang},
  year={2026},
  booktitle={Forty-third International Conference on Machine Learning},
  year={2026},
  url={https://openreview.net/forum?id=s8UGIRF38a}
}
```

- 论文：https://arxiv.org/abs/2605.29716
- 仓库：https://github.com/generaldi/NaRA

### DiT-BlockSkip — Memory-Efficient Fine-Tuning Diffusion Transformers

```bibtex
@article{park2026ditblockskip,
  title   = {Memory-Efficient Fine-Tuning Diffusion Transformers via Dynamic Patch Sampling and Block Skipping},
  author  = {Park, Sunghyun and Kim, Jeongho and Park, Hyoungwoo and Das, Debasmit and Yun, Sungrack and Hayat, Munawar and Choo, Jaegul and Porikli, Fatih and Choi, Seokeon},
  journal = {arXiv preprint arXiv:2603.20755},
  year    = {2026}
}
```

- 论文：<https://arxiv.org/abs/2603.20755>

## License

[KohakuMoss License 1.0](LICENSE)
