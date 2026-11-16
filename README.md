# KohakuMossTrainer · NaRA fork

[KohakuBlueleaf/KohakuMossTrainer](https://github.com/KohakuBlueleaf/KohakuMossTrainer)

在原基础上增加了 **NaRA（Noise-aware Rank Adaptation，噪声感知秩适配）** 的训练、采样与导出支持。

## 新增内容

- **NARA 开关**：`NARA = True` 像 `DORA` 一样可以作用于 lora / locon / loha / lokr
- lora / locon 下等价于 `ALGO = "nara"`
  
  LoHa 的两个秩通道读同一个核心

  LoKr 的核心在 `w2` 的秩收缩中
- **NaRA × DoRA 组合**：归一化在 `BA·C(sigma)` 之后
  
  LoRA / LoKr 用 Gram 展开 `||W + dW||² = ||W||² + 2<W, dW> + ||dW||²`
- **训练 / 采样**：每步自动加入 `(sigma, seq)`；
  采样器在接收到 `needs_sigma` 时自动开启动态求值

- **导出**：`lycoris` 格式保留完整 NaRA
  
  `comfy` / `merged` 在 `export_sigma`（默认 0.5）将核心冻结导出

  *建议使用both全部导出*

- 配套测试（`tests/` 中的 nara 用例）与设计文档 [docs/internals/adapter-nara.md](docs/internals/adapter-nara.md)。

## 用法

### 训练

在训练配置中打开

```python
ALGO = "lora"            # "lora" | "locon" | "loha" | "lokr"
NARA = True              # NaRA开关
DORA = True              # 可选：开启DoRA

ADAPTER_MODE = "bypass"  # NaRA 只支持 bypass
SAVE_FORMAT = "both"     
```

NaRA超网络可选参数 `ALGO_KWARGS` ：

| key | 默认 | 含义 |
| --- | --- | --- |
| `nara_hidden1`, `nara_hidden2` | 256, 512 | mapper 隐藏层宽度 |
| `nara_embed_dim` | 64 | 嵌入宽度 |
| `nara_embed` | `fourier` | `fourier`/ `mlp`/ `raw` |
| `nara_fourier_scale` | 16.0 | 固定 Fourier 嵌入的频率尺度 |
| `nara_c_scale` | 1.0 | 核心残差尺度 eta |
| `export_sigma` | 0.5 | 固定 sigma 导出所用的噪声水平 |

### ComfyUI 推理

[推理节点](https://github.com/RBkreb/comfyui-moss-nara)

## 限制

- 拒绝`ADAPTER_MODE = "merge"`
- 开启NaRA时`rank dropout`/`use_tucker` / `use_scalar` / `rs_lora` 被拒绝
- 共享超网络不能跨 FSDP 分片或流水线阶段：fsdp / pipeline / threaded 布局被拒绝，
- LoKr 变体要求 `w2` 有秩可放核心：`w2` 满秩的布局在构造时被拒绝（需降低 `DIM` 或调大 `ALGO_KWARGS["factor"]`）。
- LoHa 变体的 bypass 逐样本重建稠密增量，建议配合 block checkpointing 档位。

## 样例
[Adapter样例](https://huggingface.co/RBXLSP/TextureLux-DoRAN)

## NaRA

NaRA 让适配器的低秩增量随去噪噪声水平变化。两个静态 LoRA 因子之间放一个动态核心：

```
h = W0 x + s · B C(sigma) A x
C(sigma) = I_r + c_scale · F(e_sigma)
```

- `F` 是一个小型 MLP（mapper），`e` 是噪声水平的高斯 Fourier 嵌入；
  **所有被适配层共享同一个超网络**，每层只保存 A（`lora_down`）/ B（`lora_up`）。
- mapper 末层零初始化，初始 `C = I`，NaRA 退化为普通 LoRA。
- 条件化按 packed batch 的**每个样本**进行（sigma `[B]` -> 核心 `[B, r, r]`，fp32），
  沿层的 token 轴展开，与 T-LoRA 的 mask 机制一致。

## License

[KohakuMoss License 1.0](LICENSE)
