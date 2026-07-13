# Kimi Linear：一种表达力强且高效的注意力架构

原文：`2510.26692v2_KDA.pdf`  
标题：Kimi Linear: An Expressive, Efficient Attention Architecture  
团队：Kimi Team / Moonshot AI  
说明：本文档为论文正文的中文翻译与阅读版整理。PDF 中的复杂公式、图示和参考文献排版在文本提取时会有损失，因此这里保留关键公式与结论，重点翻译论文的核心内容。

## 摘要

作者提出了 Kimi Linear，一种混合线性注意力架构。它首次在公平比较下，在短上下文、长上下文以及强化学习扩展等多种场景中超过了全注意力模型。

Kimi Linear 的核心是 Kimi Delta Attention，简称 KDA。KDA 是一种表达能力更强的线性注意力模块，它在 Gated DeltaNet 的基础上引入了更细粒度的门控机制，使有限状态 RNN 记忆能够被更有效地利用。作者还设计了专门的分块算法，通过一种特殊形式的 Diagonal-Plus-Low-Rank，简称 DPLR，转移矩阵提升硬件效率。相比通用 DPLR 形式，这种设计显著减少计算量，同时仍然保持与经典 delta rule 的一致性。

作者预训练了一个 Kimi Linear 模型：激活参数 3B，总参数 48B，架构上按层混合 KDA 与 Multi-Head Latent Attention，简称 MLA。实验显示，在完全相同的训练配方下，Kimi Linear 在所有评测任务上都明显超过全 MLA，同时 KV cache 使用量最多减少 75%，在 1M 上下文长度下解码吞吐最高达到 6 倍。

这些结果说明，Kimi Linear 可以作为全注意力架构的直接替代方案，在性能和效率上都更优，尤其适合输入和输出都更长的任务。

作者开源了 KDA kernel、vLLM 实现，并发布了预训练和指令微调模型权重。

## 1. 引言

随着大语言模型逐渐演化为更强的智能体，推理阶段的计算需求，尤其是在长程任务和强化学习场景中，正在成为核心瓶颈。模型需要在推理时处理长轨迹、工具调用交互和复杂决策空间，这暴露了标准注意力机制的根本低效。

Softmax attention 的二次时间复杂度，以及随序列长度线性增长的 KV cache，会带来巨大的计算和显存开销，限制吞吐、上下文长度扩展和实时交互能力。

线性注意力提供了一种降低复杂度的原则性方法，但它长期以来在语言建模中弱于 softmax attention，即使在短序列上也是如此，主要原因是表达能力不足。近期进展显著缩小了这个差距，核心来自两类机制：

- gating / decay 机制；
- delta rule。

这些进展让线性注意力在中等长度序列上更接近 softmax 质量。然而，纯线性结构仍然受到有限状态容量的限制，长序列建模和上下文内检索在理论上仍然困难。

因此，混合架构成为实用折中方案：用少量全局注意力层配合大量更快的线性注意力层，在质量和效率之间取得平衡。但之前的混合模型往往规模有限，或者缺少跨多类基准的全面评测。

本文提出 Kimi Linear，目标是在不牺牲质量的情况下满足智能体和 test-time scaling 的效率需求。核心模块 KDA 在 Gated DeltaNet 基础上引入 channel-wise 门控。GDN 和 Mamba2 类似，使用较粗粒度的 head-wise 遗忘门；KDA 则让每个特征维度拥有独立遗忘率，类似 Gated Linear Attention。这种细粒度设计让模型能更精确地调控有限状态 RNN 记忆。

KDA 使用一种特殊的 DPLR 矩阵参数化转移动态，从而支持专门的 chunkwise parallel 算法。相对于通用 DPLR 形式，它显著减少计算，同时保持与经典 delta rule 的一致性。

Kimi Linear 按统一的 3:1 比例交替堆叠 KDA 层和全注意力层。这个混合结构在长序列生成时最多减少 75% 的内存和 KV cache 使用，同时通过全注意力层保留全局信息流。

### 主要贡献

- Kimi Delta Attention：一种改进 gated delta rule 的线性注意力机制，提升循环记忆管理能力和硬件效率。
- Kimi Linear 架构：采用 3:1 的 KDA 与全局注意力比例，在降低内存占用的同时超过全注意力质量。
- 大规模公平实验验证：在 1.4T token 训练下，Kimi Linear 在短上下文、长上下文和 RL 风格评测中超过全注意力和其他基线，并开源 kernel、vLLM 集成和模型权重。

## 2. 预备知识

### 2.1 记号

论文中使用 `q, k, v, o, u, w` 表示各时间步的列向量，`S_t` 表示矩阵形式的记忆状态。`M` 和 `M^-` 分别表示带对角线和不带对角线的下三角 mask。

在分块形式中，长度为 `L` 的序列被切成长度为 `C` 的块。每个块内的 `Q, K, V, O, U, W` 被堆叠成矩阵。块初始状态等于前一块的最后状态。

论文还定义了累计衰减项 `gamma`，用于描述细粒度 decay 在块内不同位置之间的累乘关系。

### 2.2 线性注意力与 gated delta rule

线性注意力维护一个矩阵值循环状态，用来累计 key-value 关联：

```text
S_t = S_{t-1} + k_t v_t^T
o_t = S_t^T q_t
```

从 fast-weight 角度看，`S_t` 是一个关联记忆，保存临时的 key 到 value 映射。这个更新可以看作在无界相关性目标上做梯度下降，它不断强化最近的 key-value 对，但没有遗忘机制。因此，累积状态会无界增长，长上下文中容易产生干扰。

DeltaNet 将这个过程重新解释为对重建损失做在线梯度下降：

```text
L_t(S) = 1/2 || S^T k_t - v_t ||^2
S_t = (I - beta_t k_t k_t^T) S_{t-1} + beta_t k_t v_t^T
```

这就是经典 delta rule。它把 `S` 看作可学习的关联记忆，不断修正自身，使其更接近 `k_t -> v_t` 的映射。这个 rank-1 更新结构等价于一种广义 Householder 变换，适合硬件高效的 chunkwise parallel。

Gated DeltaNet 在 DeltaNet 基础上引入标量遗忘门 `alpha_t`：

```text
S_t = alpha_t (I - beta_t k_t k_t^T) S_{t-1} + beta_t k_t v_t^T
```

`alpha_t` 相当于 fast weights 上的 weight decay，也就是一种遗忘机制，类似数据相关的 L2 正则。它能控制记忆寿命、减少干扰，并改善稳定性和长上下文泛化。

## 3. Kimi Delta Attention：用细粒度门控改进 delta rule

KDA 是一种新的 gated linear attention 变体。它把 GDN 的标量 decay 扩展为细粒度的对角门控 `Diag(alpha_t)`，从而更精确地控制记忆衰减和位置感知。

核心递推可以写成：

```text
S_t = (I - beta_t k_t k_t^T) Diag(alpha_t) S_{t-1} + beta_t k_t v_t^T
o_t = S_t^T q_t
```

### 3.1 硬件高效的分块算法

作者将上述递推展开为 chunkwise 形式。块内的多次 rank-1 矩阵变换可以被压缩成密集表示，同时在对角门控下保持稳定。

论文使用 WY representation 把一系列 rank-1 更新打包成紧凑表示。然后使用 UT transform 减少非矩阵乘 FLOPs，这对训练时提升硬件利用率很重要。

在输出阶段，KDA 采用“块间循环 + 块内并行”的策略来最大化矩阵乘吞吐，从而充分利用 Tensor Core。

### 3.2 效率分析

从表达能力看，KDA 与广义 DPLR 形式一致：

```text
S_t = (D - a_t b_t^T) S_{t-1} + k_t v_t^T
```

二者都具备细粒度 decay 行为。但通用 DPLR 会在除法操作中引入数值精度问题。之前如 GLA 的做法是在 log domain 计算，并引入二级 chunking，但这会妨碍半精度矩阵乘的充分利用，降低算子速度。

KDA 将 DPLR 中的两个变量约束并绑定到 `k`，从而缓解瓶颈。具体效果是：

- 二级 chunk 矩阵计算从 4 个减少到 2 个；
- 额外消除 3 次矩阵乘；
- kernel 效率相比 DPLR 大约提升 100%。

论文的 benchmark 显示，在输入长度从 2K 到 64K 时，KDA kernel 接近 DPLR 的 2 倍速度。

## 4. Kimi Linear 模型架构

Kimi Linear 的主干架构继承自 Moonlight。除了细粒度门控，作者还引入多个组件来增强表达能力。

### 神经参数化

给定第 `t` 个 token 的输入表示 `x_t`，每个 head 的 KDA 输入由以下形式计算：

```text
q_t^h, k_t^h = L2Norm(Swish(ShortConv(W_{q/k}^h x_t)))
v_t^h = Swish(ShortConv(W_v^h x_t))
alpha_t^h = f(W_alpha_up W_alpha_down x_t)
beta_t^h = Sigmoid(W_beta^h x_t)
```

`q, k, v` 都经过 ShortConv 和 Swish。`q` 和 `k` 进一步做 L2Norm，用于保证特征值稳定。每通道 decay `alpha_t` 通过低秩投影和 decay 函数得到。`beta_t` 用 sigmoid 得到。

输出投影前，作者使用 head-wise RMSNorm 和数据相关的输出门：

```text
o_t = W_o [ Sigmoid(W_g_up W_g_down x_t) * RMSNorm(KDA(q_t, k_t, v_t, alpha_t, beta_t)) ]
```

输出门也使用低秩参数化，用于公平控制参数量，同时保持接近全秩门控的性能，并缓解 attention sink。

### 混合模型架构

纯线性注意力的主要瓶颈仍是长上下文检索。因此 Kimi Linear 将 KDA 与少量 full global attention，也就是 Full MLA 层混合。

作者选择 layerwise 混合，而不是在同一层内混合不同 head。原因是 layerwise 混合在基础设施上更简单，训练也更稳定。

实验上，统一的 3:1 比例效果最好，也就是每 3 层 KDA 后接 1 层 MLA。这个比例在质量和吞吐之间取得最佳平衡。

### MLA 层使用 NoPE

Kimi Linear 对所有全注意力 MLA 层使用 NoPE，即不使用显式位置编码。这样，位置编码和近因偏置的责任主要交给 KDA 层。

这种设计使 KDA 成为主要的位置感知算子，作用类似甚至强于短卷积或滑动窗口注意力等辅助组件。NoPE 对 MLA 也有实际优势：

- 推理时可以转为高效的纯 MQA；
- 简化长上下文训练，不需要调整 RoPE base 或使用 YaRN 等方法。

## 5. 实验

### 5.1 合成任务

作者首先在三个合成任务上比较 KDA 与其他线性注意力方法：

- Palindrome：输入随机 token 序列，模型需要反向复现。复制任务通常对线性注意力很难，因为它们需要从压缩固定大小状态中精确取回全部历史。
- Multi Query Associative Recall，简称 MQAR：要求模型检索上下文中多个 query 对应的 value。该任务与语言建模性能高度相关。
- Stack：模拟 LIFO 栈操作，评估模型状态跟踪能力。模型需要处理多个独立栈的 push / pop，并在 pop 时预测正确元素。

实验结果显示，在序列长度从 256 增加到 2048 时，KDA 在所有任务上都取得最高准确率。尤其在 Palindrome 和 MQAR 上，KDA 比 GDN 收敛明显更快。

这验证了细粒度 decay 的收益：模型可以选择性遗忘无关信息，同时更精确地保留关键记忆。Mamba2 在这些设置下全部失败，因为它只有乘性 decay，没有 delta rule。

### 5.2 关键组件消融

作者用 16 heads、16 layers 的第一档 scaling law 模型做消融，所有模型使用相同 FLOPs 预算和超参数。

关键结果：

- 3:1 的 KDA:MLA 比例训练 PPL 为 9.23，验证 PPL 为 5.65，是最优配置。
- 全注意力 0:1 的验证 PPL 为 5.77，弱于 Kimi Linear。
- 1:1 比例验证 PPL 接近，但推理开销更高。
- 7:1 比例训练损失类似，但验证性能明显变差。
- 去掉输出门会退化。
- Swish 输出门明显弱于 Sigmoid 输出门。
- 去掉卷积层也会退化，说明轻量 depthwise convolution 在混合模型中仍有作用。

### NoPE vs. RoPE

Kimi Linear 在长上下文评测上持续更强，而 Kimi Linear with RoPE 在短上下文上接近。

作者认为差异来自位置偏置在层间的分布方式。RoPE 版本中，全局注意力层带有很强的显式相对位置偏置，而线性注意力贡献较弱的隐式位置归纳偏置。这种不匹配会让全局层过分强调短程顺序，有利于短上下文，却降低了中途扩展长上下文时的灵活性。

相比之下，Kimi Linear 通过 KDA 在层间形成更平衡的位置偏置，因此对长距离更鲁棒，外推更好。

### 5.3 Scaling law

作者沿用 Moonlight 架构训练一系列 MoE 模型。所有实验激活 64 个专家中的 8 个，并使用 Muon optimizer。

对于 MLA，作者按 Chinchilla scaling law 方法训练 5 个不同规模模型，并网格搜索超参数。对于 KDA，固定使用消融得到的 3:1 混合比例，其余训练配置严格沿用 MLA。

结果显示，在 compute-optimal 训练下，Kimi Linear 相比 MLA baseline 达到约 1.16 倍计算效率。作者认为，如果进一步精调 KDA 超参数，scaling 曲线还会更好。

### 5.4 实验设置

作者比较三类模型：

- full-attention MLA baseline；
- hybrid Gated DeltaNet，简称 GDN-H；
- Kimi Linear。

三者使用相同架构、参数量和训练设置以保证公平。模型大体对齐 Moonlight，但 MoE 稀疏度提高到 32。每次 forward 激活 256 个专家中的 8 个，包括一个共享专家。总参数 48B，每次前向激活参数 3B。第一层使用 dense 层，不使用 MoE，以保证训练稳定。

评测覆盖：

- 通用理解与推理：HellaSwag、ARC-Challenge、Winogrande、MMLU、TriviaQA、MMLU-Redux、MMLU-Pro、GPQA-Diamond、BBH、LiveBench；
- 代码生成：LiveCodeBench v6、EvalPlus；
- 数学与推理：AIME 2025、MATH 500、HMMT 2025、PolyMath-en；
- 长上下文：MRCR、RULER、Frames、HELMET-ICL、RepoQA、Long Code Arena、LongBench v2；
- 中文理解与推理：C-Eval、CMMLU。

预训练使用 4096 token 上下文窗口、MuonClip optimizer 和 WSD 学习率计划，在 K2 预训练语料上共同训练 1.4T token。最终发布的 Kimi Linear checkpoint 使用同样流程训练 5.7T token，并支持最高 1M token 上下文。

SFT 数据在 Kimi K2 的 SFT 数据基础上加入更多推理任务，采用多阶段 SFT：先训练通用指令跟随，再按计划强化推理密集数据。

RL 阶段主要使用数学、代码和 STEM 数据，目标是增强推理能力。为避免 RL 导致通用能力退化，作者在 RL 中加入 PTX loss，也就是在 RL 过程中同步对高质量、多分布数据做 SFT。作者还使用 truncated importance sampling、动态 KL penalty 和动态 mini-batch size 来提升训练稳定性。

## 5.5 主要结果

### 1.4T 预训练结果

在 1.4T 预训练语料上，Kimi Linear 与 MLA、GDN-H 比较。Kimi Linear 在几乎所有类别上都超过两个 baseline。

代表性结果：

| 类型 | 指标 | MLA | GDN-H | Kimi Linear |
|---|---:|---:|---:|---:|
| General | HellaSwag | 81.7 | 82.2 | 82.9 |
| General | ARC-Challenge | 64.6 | 66.5 | 67.3 |
| General | MMLU | 71.6 | 72.2 | 73.8 |
| General | MMLU-Pro | 47.2 | 47.9 | 51.0 |
| General | TriviaQA | 68.9 | 70.1 | 71.7 |
| Math & Code | GSM8K | 83.7 | 81.7 | 83.9 |
| Math & Code | EvalPlus | 59.5 | 63.1 | 60.2 |
| Math & Code | CRUXEval-O-cot | 61.5 | 58.1 | 62.0 |
| Chinese | CEval | 79.3 | 79.1 | 79.5 |
| Chinese | CMMLU | 79.5 | 80.7 | 80.8 |

总结来说，Kimi Linear 在短上下文预训练评测中表现最强，是全注意力架构的有力替代。

### SFT 结果

经过相同 SFT 后，Kimi Linear 在通用任务和数学/代码任务上继续保持强势。

代表性结果：

| 指标 | MLA | GDN-H | Kimi Linear |
|---|---:|---:|---:|
| BBH | 68.2 | 68.5 | 69.4 |
| MMLU | 75.7 | 75.6 | 77.0 |
| MMLU-Pro | 65.7 | 64.8 | 67.4 |
| MMLU-Redux | 79.2 | 78.7 | 80.3 |
| GPQA-Diamond Avg@8 | 57.1 | 58.6 | 62.1 |
| AIME 2025 Avg@64 | 20.6 | 21.1 | 21.3 |
| HMMT 2025 Avg@32 | 11.3 | 11.3 | 12.5 |
| PolyMath-en Avg@4 | 41.3 | 41.5 | 43.6 |
| LiveCodeBench v6 Pass@1 | 25.1 | 25.4 | 26.0 |

虽然 MATH500 和 EvalPlus 有小例外，但整体上 Kimi Linear 明显优于 GDN-H 和 MLA。

### 长上下文结果

在 128K 上下文长度下，Kimi Linear 与 MLA、GDN-H、Kimi Linear RoPE 比较。

| 模型 | RULER | MRCR | HELMET-ICL | LongBench V2 | Frames | RepoQA | Long Code Arena Lib | Commit | 平均 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| MLA | 81.3 | 22.6 | 88.0 | 36.1 | 60.5 | 63.0 | 32.8 | 33.2 | 52.2 |
| GDN-H | 80.5 | 23.9 | 85.5 | 32.6 | 58.7 | 63.0 | 34.7 | 30.5 | 51.2 |
| Kimi Linear RoPE | 78.8 | 22.0 | 88.0 | 35.4 | 59.9 | 66.5 | 31.3 | 32.5 | 51.8 |
| Kimi Linear | 84.3 | 29.6 | 90.0 | 35.0 | 58.8 | 68.5 | 37.1 | 32.7 | 54.5 |

Kimi Linear 在 RULER 和 RepoQA 上优势明显，并取得最高平均分 54.5。除 LongBench V2 和 Frames 外，它在多数任务上领先。

### RL 结果

作者用内部数学训练集做 RLVR，比较 Kimi Linear 与 MLA。算法和超参数完全一致。

结果显示，Kimi Linear 的 RL 收敛效率更好。训练集上，两者起点类似，但 Kimi Linear 的准确率增长更快，差距逐渐扩大。测试集上，MATH500 和 AIME 2025 也出现类似现象。

总体看，在推理密集的长文本生成 RL 场景中，Kimi Linear 明显优于 MLA。

### 整体发现

预训练和 SFT 阶段的性能层级是：

```text
Kimi Linear > GDN-H > MLA
```

长上下文评测中，Kimi Linear 仍然第一，但 GDN-H 下降到 MLA 之后。RL 阶段中，Kimi Linear 也超过 MLA。整体而言，Kimi Linear 在所有阶段都表现最佳。

## 5.6 效率比较

作者比较了 MLA、GDN-H 和 Kimi Linear 的 prefill 与 decode 时间。所有模型基于 Kimi Linear 48B 设置，层数和 attention heads 相同。

主要观察：

- Kimi Linear 虽然引入更细粒度 decay，但 prefill 延迟相比 GDN-H 几乎没有额外开销。
- 在短序列 4K 到 16K 上，Kimi Linear 与 MLA 接近。
- 从 128K 开始，Kimi Linear 明显更快。
- 512K 序列上，Kimi Linear 比 MLA 快 2.3 倍。
- 1M 序列上，prefill 快 2.9 倍；decode 阶段最高可达 6 倍以上。

由于线性 KDA 维护固定大小状态，不需要随序列增长的巨大 KV cache，Kimi Linear 可以把显存用于更大 batch，从而提升总体吞吐。论文在 1M token 场景中给出理论最高 6.3 倍解码加速。

## 6. 讨论

### 6.1 KDA 作为可学习位置编码

标准 Transformer attention 本身不感知输入顺序，因此需要显式位置编码。RoPE 是现代 LLM 的事实标准之一。

论文从广义 attention 形式分析乘性位置编码。RoPE 通过旋转矩阵的累乘，把绝对位置信息转为相对位置信息。

作者指出，带 gated delta rule 的线性注意力也可以写成类似形式。因此 GDN 可以被解释为一种乘性位置编码，只是它的转移矩阵是数据相关且可学习的。这放松了 RoPE 的正交约束，理论上可能更强。

这也为 RoPE 的外推问题提供了潜在解决方案。RoPE 的固定频率可能过拟合训练时见过的上下文长度，而 GDN/KDA 的位置机制是动态的。

KDA 相比标准 GDN 的关键改进在于：RoPE 的一个优势是细粒度位置编码，不同维度有不同旋转频率；标准 GDN 只有每 head 一个标量 decay，缺少按维度的多样性。因此作者提出 channel-wise gate，让 KDA 具备更细粒度的可学习位置偏置。

### 6.2 与 DPLR 的关系

Gated DeltaNet 可以推广到更有表达力的 DPLR 结构：

```text
D - a_t b_t^T
```

S4 等模型也使用过静态 DPLR 形式作为状态转移矩阵。但 DPLR 的问题是计算成本高、并行性较差，难以用于大规模或实时场景。

KDA 使用受限 DPLR：

```text
S_t = (Diag(alpha_t) - beta_t k_t k_t^T Diag(alpha_t)) S_{t-1}
      + beta_t k_t v_t^T
```

也就是令：

```text
D = Diag(alpha_t)
a_t = beta_t k_t
b_t = k_t * alpha_t
```

通过共享 `alpha_t`，KDA 可以先做细粒度乘性 decay，再做类似 DeltaNet 的 Householder 风格变换。这保留了表达力，同时大幅减少计算。

相对于通用 DPLR，KDA 的主要改进是：

- 避免累计 decay 倒数导致的数值不稳定；
- 去掉两个二级 chunking 步骤；
- 减少 inter-chunk 和 output 计算中的约三次矩阵乘；
- 在 64K 以内序列长度上接近 2 倍 kernel 加速。

### 6.3 复杂度分析

Kimi Linear 与全 MLA 保持类似参数量，线性投影计算也相同。主要差别在 attention computation FLOPs。

对于单个 head，head dim 为 `d_h`，固定 chunk size `C = 64`，KDA 的理论 FLOPs 为：

```text
FLOPs_KDA(T; C, d_h) = 6T d_h^2 + 3T C d_h + T C^2
```

全局 attention 的主导项为：

```text
FLOPs_Attn(T; d_h) = 2T^2 d_h
```

推理时，prefill 使用 FLOP 密集的 chunk kernel，自回归生成切换到更高效的 recurrent kernel。KDA 的状态大小固定为每 head `d_k x d_v`，与序列长度无关。在 3:1 混合模型中，随着序列增长，I/O bounded decode 时间会接近 3:1 的混合效率上限。

## 7. 相关工作

### 7.1 高效次二次注意力

标准 self-attention 的二次复杂度是处理长上下文的核心瓶颈。随着 LLM 需要处理百万 token 序列、智能体工具调用和仓库级代码分析，高效 attention 变得更加重要。

研究路线大体分为两类：

- 线性注意力；
- 稀疏注意力。

线性注意力把二次 attention map 改写为 kernelized feature interactions，用正特征映射替代 softmax，使 attention 可以通过两个结合律矩阵乘实现，从而避免显式 `O(T^2)` 相似度矩阵。

后续工作通过更精细的记忆控制显著增强了原始线性注意力：从数据无关 decay 发展到数据相关机制，从粗粒度 headwise decay 发展到 channel-wise decay。GLA 使用对角 channel-wise gates，在表达力和效率之间取得平衡，并保留 chunkwise parallel。

另一种视角把线性注意力看作 fast-weight memory：状态是低容量关联表，在线用 Hebbian-like rule 更新；慢权重则学习何时存储、更新或遗忘。Gating 和 decay 可以被理解为可学习的准则，用于缓解干扰和稳定优化。

但线性注意力在极长上下文精确复制和细粒度选择上仍弱于全注意力，因此需要混合设计和更结构化的更新。GDN/KDA 使用 gated delta rule，通过 rank-1 corrective update 改进 fast-weight 状态，在保持算子并行性的同时提升定向记忆能力。

稀疏注意力则通过只计算一部分 token 来近似全注意力。早期方法使用滑动窗口、扩张窗口或固定模式，但刚性结构会损害准确率。后续方法使用聚类、路由或 chunk-level 选择等动态策略，但动态选择也会引入额外计算开销。

线性注意力与稀疏注意力是两条不同路线。稀疏注意力更擅长细粒度历史检索，但仍需要存储完整 KV cache 用于 token selection；线性注意力则维持常量状态，更符合“压缩即智能”的思路。作者认为，未来可以结合两者优势：线性注意力负责压缩和泛化，稀疏注意力负责细粒度检索。

### 7.2 混合模型

纯线性注意力虽然高效，但在精确记忆检索和复制方面仍有困难，这限制了它们在工业级 LLM 中的采用。近期研究显示，线性注意力和全注意力可以互补，因此出现了多种混合设计。

一种是 intra-layer hybrid，也就是在同一层内部融合不同机制的输出。例如在每层中混合标准注意力 head 与 SSM head，或对输入的不同部分使用不同机制。

另一种是 inter-layer hybrid，也就是按固定比例堆叠不同层类型，如全注意力层和线性层。它的系统复杂度和推理开销更低，更适合 LLM 实际部署。

Kimi Linear 采用 inter-layer hybrid，以固定 3:1 比例交替 KDA 和全注意力层。这种结构简化了 KV cache 管理，也方便接入标准优化。作者没有使用常见的 Mamba2 作为线性部分，而是使用 KDA，因为 KDA 在检索和复制能力上更强。

关于位置编码，混合模型可能对 RoPE base frequency 敏感，这会让上下文窗口扩展更困难。近期模型开始采用 NoPE 或接近 NoPE 的方案。Kimi Linear 将 KDA 与 NoPE full attention 混合，也证明是非常有效的策略。

## 结论

Kimi Linear 是一种混合线性注意力架构，目标是在不牺牲质量的前提下满足智能体和 test-time scaling 的效率需求。

它的核心 KDA 通过 channel-wise gating 增强记忆控制能力，使 RNN 风格模型在混合架构中发挥更大作用。通过按 3:1 比例交替 KDA 和全局注意力，Kimi Linear 最多减少 75% 内存使用，最高实现 6.3 倍解码吞吐，并超过全注意力 baseline。

论文结果表明，Kimi Linear 是全注意力架构的一个强有力替代方案，尤其适合长上下文、长输出和 RL 推理密集场景。

## 附录要点

### 附录 A：贡献说明

论文列出了所有作者，并说明作者顺序按贡献重要性排列，项目由 Moonshot AI 开发，部分外部合作者以特殊标记注明。

### 附录 B：KDA chunkwise parallel 推导

附录从 KDA 的递推形式出发，将块内状态写成：

```text
S_r[t] = P_r[t] S_0[t] + H_r[t]
```

其中 `P_r[t]` 是广义 Householder 矩阵的累计乘积，`H_r[t]` 是块内输入项累计。作者证明：

- `P_r[t]` 可以用 WY representation 改写为可并行矩阵形式；
- `H_r[t]` 也可以写成可并行形式；
- 两者都可以通过辅助向量 `w` 和 `u` 的递推计算得到。

这些推导支撑了正文中的 chunkwise parallel 算法。

### 附录 C：KDA chunkwise 伪代码

附录给出 PyTorch 风格伪代码。流程包括：

- 将 `q, k, v, g, beta` 按 chunk 重排；
- 计算累计门控 `g.cumsum`；
- 构造下三角矩阵 `A`；
- 通过 forward substitution 近似矩阵逆；
- 计算 `w` 和 `u`；
- 块间维护状态 `S`，块内并行计算输出；
- 返回重排后的输出。

### 附录 D：Kimi Linear@5.7T 结果

作者还按照 Moonlight 设置训练了 5.7T token 的 Kimi Linear，用于展示最终发布模型的效果。相比 Moonlight，Kimi Linear 使用 3 倍稀疏度和新的注意力架构，在几乎所有 benchmark 上都更强。

Base 模型代表结果：

| Benchmark | Kimi-Linear-Base | Moonlight-Base |
|---|---:|---:|
| TriviaQA | 75.2 | 66.2 |
| SimpleQA | 10.1 | 5.6 |
| MMLU-Pro | 54.8 | 42.4 |
| MMLU-redux | 79.7 | 73.8 |
| WinoGrande | 81.5 | 74.6 |
| GPQA-Diamond Avg@8 | 40.4 | 35.2 |
| MATH | 58.5 | 45.3 |
| GSM8K | 86.3 | 77.2 |
| LiveCodeBench v6 | 20.0 | 14.3 |
| EvalPlus | 64.9 | 50.3 |
| C-Eval | 83.3 | 77.6 |
| CSimpleQA | 53.5 | 34.7 |

Instruct 模型代表结果：

| Benchmark | Kimi-Linear-Instruct | Moonlight-Instruct |
|---|---:|---:|
| RULER@128k | 95.4 | - |
| RULER@1M | 94.8 | - |
| GPQA-Diamond Avg@8 | 71.7 | 24.7 |
| MMLU-Redux EM | 86.9 | 66.9 |
| MMLU-Pro EM | 72.7 | 43.8 |
| FaithJudge | 64.2 | 56.0 |
| AIME 2025 Avg@64 | 58.6 | - |
| MATH500 Acc. | 94.6 | 58.0 |
| HMMT 2025 Avg@32 | 44.5 | - |
| LiveCodeBench v6 Pass@1 | 45.7 | 11.9 |
| Humaneval+ | 70.9 | 46.3 |
| MBPP+ | 72.4 | 56.3 |

Kimi Linear@5.7T 在 1M 上下文的 RULER 上达到 94.8，进一步说明它是全注意力架构的有前景替代方案，能在保持或超过性能的同时提升资源利用效率。

## 术语对照

| 英文 | 中文理解 |
|---|---|
| Linear Attention | 线性注意力 |
| Full Attention | 全注意力 |
| Kimi Delta Attention / KDA | Kimi Delta 注意力 |
| Gated DeltaNet / GDN | 门控 DeltaNet |
| Multi-Head Latent Attention / MLA | 多头潜在注意力 |
| Diagonal-Plus-Low-Rank / DPLR | 对角加低秩结构 |
| chunkwise parallel | 分块并行 |
| recurrent state | 循环状态 |
| fast-weight memory | 快权重记忆 |
| forget gate / decay | 遗忘门 / 衰减 |
| NoPE | 无显式位置编码 |
| RoPE | 旋转位置编码 |
| KV cache | 键值缓存 |
| prefill | 预填充阶段 |
| decoding | 解码阶段 |
| TPOT | 每输出 token 时间 |
