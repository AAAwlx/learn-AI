# 推理流程与 KV cache 的概念

当前主流大语言模型（如 Llama、Qwen、GPT 等）普遍采用 decoder-only 架构，其推理过程为 自回归生成（autoregressive generation）：逐个 token 地生成回答，每一步都依赖此前所有已知 token（包括用户输入和已生成内容）。

![alt text](../images/推理流程与kvcache/image.png)

## 整体流程分两阶段

* 阶段 1️⃣：**Prefill（预填充）** —— 处理用户输入  
* 阶段 2️⃣：**Decode（解码）** —— 逐个生成回答

## 🎯 示例设定

* **用户输入（Prompt）**: `"What is 2+2?"` → 共 5 个 token：`[W, i, s, 2+2, ?]`
* **模型回复（Response）**: `"2+2 is 4."` → 共 5 个 token：`[2+2, i, s, 4, .]`
* 总序列长度 = 10
* 模型架构：**Decoder-only Transformer**（无 encoder，无 cross-attention）
* 注意力：**Causal Self-Attention**（只能看当前位置及之前）
* 我们关注 **单个 attention head** 的行为（多头同理）

## 阶段 1️⃣：Prefill —— 处理 `"What is 2+2?"`

输入 token 序列（索引 0~4）：

```text
x₀ = "What"
x₁ = "is"
x₂ = "2+2"
x₃ = "?"
```

> 实际 tokenization 可能不同，这里简化为 4 个 token（含标点）。

### 步骤

1. 将 `[x₀, x₁, x₂, x₃]` 一次性送入模型。
2. 对每个位置 i ∈ {0,1,2,3}，计算：
   * $ Q_i = \text{LayerNorm}(x_i) W_Q $
   * $ K_i = \text{LayerNorm}(x_i) W_K $
   * $ V_i = \text{LayerNorm}(x_i) W_V $

3. 计算 **causal attention**（下三角）：
   * 例如，计算位置 2（"2+2"）的输出时：
     * Q₂ 与 [K₀, K₁, K₂] 做点积 → 得到 3 个 score
     * softmax → 权重
     * 加权 [V₀, V₁, V₂] → 输出表示

4. **关键操作**：将所有 K₀~K₃ 和 V₀~V₃ **存入 KV Cache**（显存中）。

5. 虽然模型也计算了每个位置的 logits，但 **我们只关心最后一个位置（x₃）的 logits**，用于启动生成。不过通常不采样，直接进入 decode。

在 Prefill 结束时会有以下产物：

* KV Cache = {K₀, K₁, K₂, K₃}, {V₀, V₁, V₂, V₃}
* hidden states（隐状态即QKV运算得到的结果）
* 当前序列长度 = 4
* 下一步要生成第 5 个 token（即 response 的第一个词）

## 阶段 2️⃣：Decode —— 逐个生成回答

在自回归生成，每次生成一个 token，追加到序列末尾。

### 🔹 Step 1：生成 y₀ = "2+2"

* 当前完整序列：`[x₀,x₁,x₂,x₃]`（长度=4）
* 要预测位置 4 的 token（y₀）

#### 计算过程

1. 取 **最后一个 hidden state**（对应 x₃ 的计算结果）→ 经过 LayerNorm → 得到向量 h₄
2. 计算当前 Q：
   * $ Q_4 = h_4 W_Q $ ← **这是第一个“生成阶段”的 Q**
3. 从 KV Cache 读取：
   * K = [K₀, K₁, K₂, K₃]（4 个）
   * V = [V₀, V₁, V₂, V₃]
4. 计算 attention：
   * scores = Q₄ · [K₀; K₁; K₂; K₃]ᵀ → shape (1,4)
   * weights = softmax(scores)
   * output = weights · [V₀; V₁; V₂; V₃]
5. 经过 FFN 等 → 得到 logits → 采样出 **y₀ = "2+2"**
6. **更新 KV Cache**：
   * 用 y₀ 计算 K₄ = y₀_emb W_K, V₄ = y₀_emb W_V
   * 存入 cache → KV Cache 现在有 5 个 K/V 对

> ✅ 此时 Q₄ 用完即弃，不缓存。

### 🔹 Step 2：生成 y₁ = "is"

* 当前序列：`[x₀,x₁,x₂,x₃, y₀]`（长度=5）
* 预测位置 5

1. 取 y₀ 的 hidden state → h₅
2. 计算 Q₅ = h₅ W_Q
3. 读取 KV Cache（K₀~K₄, V₀~V₄）
4. Q₅ · [K₀..K₄]ᵀ → (1,5) scores
5. softmax + 加权 V → output
6. 采样 → y₁ = "is"
7. 计算 K₅, V₅ → 存入 cache（现在 cache 有 6 项）

### 🔹 Step 3：生成 y₂ = "4"

* 序列长度=6，预测位置 6
* Q₆ = h₆ W_Q（h₆ 来自 y₁="is"）
* KV Cache: K₀~K₅, V₀~V₅（包含 prompt + "2+2", "is"）
* Q₆ 会 attend 到 **所有历史**，包括：
  * 用户问的 `"What is 2+2?"`
  * 自己已答的 `"2+2 is"`
* 从而知道该输出 `"4"`

→ 重复此过程，直到生成句号 `.` 并结束。

## 📊 关键总结表

| 步骤 | 当前 token | Q 来源 | K/V 来源 | 是否更新 KV Cache |
|------|-----------|--------|----------|------------------|
| Prefill | x₀~x₃ ("What is 2+2?") | 各自位置的 hidden state | 同序列 | ✅ 是（初始化 cache） |
| Decode 1 | y₀ ("2+2") | x₃ 的 hidden state | K₀~K₃（prompt） | ✅ 是（加入 K₄,V₄） |
| Decode 2 | y₁ ("is") | y₀ 的 hidden state | K₀~K₄（prompt + y₀） | ✅ 是（加入 K₅,V₅） |
| Decode 3 | y₂ ("4") | y₁ 的 hidden state | K₀~K₅（全部历史） | ✅ 是 |

## kv cache的存储

**KV Cache 默认且优先存储在 GPU 显存中；只有在显存不足且推理引擎支持 swap 时，才会临时换出到 CPU 内存。**  

### ✅ 1. **正常情况下：KV Cache 存在 GPU 显存中**

* LLM 推理是计算密集型任务，**所有张量（包括模型权重、激活值、KV Cache）都尽可能驻留在 GPU 显存中**，以避免频繁的 CPU-GPU 数据传输（带宽瓶颈）。
* KV Cache 是推理过程中**访问最频繁的数据之一**（每个 decode 步都要读取全部历史 K/V），必须放在高速显存中才能保证吞吐。
* 例如，在 vLLM、TensorRT-LLM、llama.cpp（CUDA 后端）等高性能推理引擎中，KV Cache 默认分配在 **GPU DRAM** 上。

> 📌 **典型大小**：  
> 对于 Llama-2-70B（64 heads, 8192 hidden dim），上下文长度 32k 时，单个请求的 KV Cache 可达 **~50 GB** —— 这必须依赖高端 GPU（如 A100/H100）的大显存。

### ⚠️ 2. **显存不足时：部分 KV Cache 可换出到 CPU 内存**

当并发请求多或上下文很长导致 GPU 显存不足时，**高级推理引擎（如 vLLM）会启用 swap 机制**：

* 将**不活跃请求**的 KV Cache 从 GPU 显存 **换出（swap out）到 CPU 内存**；
* 当该请求再次需要生成 token 时，再 **换入（swap in）回 GPU 显存**；
* 这类似于操作系统的虚拟内存机制。

> 🔧 vLLM 的调度器就维护了一个 **Swapped Queue**，专门管理这类被换出的请求。

✅ **优点**：突破 GPU 显存限制，支持更高并发或更长上下文。  
❌ **代价**：swap 操作引入延迟（PCIe 带宽远低于 GPU 显存带宽）。

### 📊 存储位置总结

| 推理模式 | KV Cache 主要存储位置 | 是否可 swap 到 CPU |
|--------|---------------------|------------------|
| **GPU 推理（默认）** | GPU 显存（VRAM） | ✅ 是（由 vLLM 等引擎管理） |
| **CPU 推理** | 系统内存（RAM） | ❌ 不适用（已在 CPU） |
| **混合推理（如部分 offload）** | 部分在 GPU，部分在 CPU | ✅ 是（需手动配置） |

### 💡 补充：为什么 KV Cache 对显存如此关键？

* KV Cache 大小 ≈ `2 × num_layers × num_kv_heads × context_len × head_dim × bytes_per_param`
* 它随 **上下文长度线性增长**，是长文本推理的主要瓶颈。
* 技术如 **PagedAttention（vLLM）**、**GQA**、**量化** 都是为了压缩 KV Cache，使其能更好地 fit 进 GPU 显存。
