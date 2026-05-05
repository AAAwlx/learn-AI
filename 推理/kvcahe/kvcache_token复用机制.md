# vLLM KV Cache复用机制详解

## 目录

1. [概述](#概述)
2. [Token化过程](#token化过程)
3. [KV Cache存储结构](#kv-cache存储结构)
4. [前缀匹配机制](#前缀匹配机制)
5. [分阶段复用策略](#分阶段复用策略)
6. [跨轮次复用](#跨轮次复用)
7. [性能优化](#性能优化)

---

## 概述

vLLM的KV cache复用机制是其性能优化的核心策略，通过智能地管理和复用KV cache，显著降低了大语言模型推理的计算成本。本文档详细解析这一机制的实现原理。

### 核心概念

- **Token IDs**: 文本经过tokenizer转换为整数序列，如 `[15496, 995]`
- **KV Cache**: Key-Value缓存，存储注意力机制的中间计算结果
- **前缀匹配**: 通过Token IDs的哈希值匹配相同的前缀
- **Block粒度**: KV cache以block为单位管理，通常16个token为一个block

---

## Token化过程

### 1.1 文本到Token IDs的转换

在`process_inputs`阶段完成文本到Token IDs的转换：

```python
# 位置：vllm/v1/engine/input_processor.py:244-247
processed_inputs = self.input_preprocessor.preprocess(
    prompt,  # 用户输入的文本
    tokenization_kwargs=tokenization_kwargs,
)
```

### 1.2 Tokenization的具体实现

```python
# 位置：vllm/renderers/base.py:386-389
def _tokenize_prompt(self, prompt: TextPrompt, params: TokenizeParams):
    tokenizer = self.get_tokenizer()
    prompt_token_ids = tokenizer.encode(
        prompt["prompt"],  # "Hello world"
        **params.get_encode_kwargs(),
    )
    return TokensPrompt(prompt_token_ids=prompt_token_ids, **prompt)
```

**示例：**

```python
text = "Hello world"
tokenizer.encode(text)  # → [15496, 995]
```

### 1.3 Token化不是向量化

重要区分：

- **Token IDs**: 整数序列 `[15496, 995]`，在`process_inputs`中计算
- **Embeddings**: 向量序列 `[[0.23, -0.45, ...], ...]`，在模型执行时计算

```python
# Token IDs计算位置
input_preprocessor.preprocess(prompt)  # → [15496, 995]

# Embeddings计算位置  
model.embed_input_ids(torch.tensor([15496, 995]))  # → 向量
```

### 1.4 为什么需要Token IDs

原因1：模块化设计

* Tokenizer 负责语言处理
* Embedding layer 负责语义表示

原因2：复用性

* 相同的Token IDs可以对应不同的Embedding（如多语言模型）
* 预计算的Embeddings可以跳过Tokenization步骤

原因3：效率

* Token IDs占用内存小（整数）
* Embeddings占用内存大（浮点数向量）
* 缓存和传输Token IDs更高效

## KV Cache存储结构

### 2.1 物理存储格式

```python
# 位置：vllm/v1/attention/backends/flash_attn.py:140-143
def get_kv_cache_shape(
    num_blocks: int,
    block_size: int, 
    num_kv_heads: int,
    head_size: int,
) -> tuple[int, ...]:
    return (2, num_blocks, block_size, num_kv_heads, head_size)
```

**存储结构：**

```python
KV_Cache.shape = (2, num_blocks, block_size, num_kv_heads, head_size)
#                 ↓  ↓          ↓          ↓             ↓
#                 2  块数量      每块token数  KV头数量      每头维度
```

### 2.2 Block粒度管理

```python
# 示例：LLaMA-2-7B模型
key_cache = kv_cache[0]   # Shape: (1000, 16, 32, 128)
value_cache = kv_cache[1] # Shape: (1000, 16, 32, 128)
# 1000个块，每块16个token，32个KV头，每头128维
```

### 2.3 内存布局

- **NHD格式**: `(num_blocks, 2, block_size, num_kv_heads, head_size)`
- **HND格式**: `(num_blocks, num_kv_heads, 2, block_size, head_size)`

---

## 前缀匹配机制

### 3.1 基于Token IDs哈希的匹配

**核心原理：不是基于向量，而是基于Token IDs的哈希值**

```python
# 位置：vllm/v1/core/kv_cache_utils.py:535-562
def hash_block_tokens(
    hash_function: Callable[[Any], bytes],
    parent_block_hash: BlockHash | None,
    curr_block_token_ids: Sequence[int],
    extra_keys: tuple[Any, ...] | None = None,
) -> BlockHash:
    curr_block_token_ids_tuple = tuple(curr_block_token_ids)
    return BlockHash(
        hash_function((parent_block_hash, curr_block_token_ids_tuple, extra_keys))
    )
```

### 3.2 Block粒度的链式哈希

**为什么要使用链式哈希？**

链式哈希的核心作用是确保前缀的连续性和顺序性。通过将父block的hash纳入当前block的hash计算，我们可以保证：

1. **前缀连续性验证**: 如果两个序列的前N个token相同，那么它们对应的前N个block的hash一定相同
2. **顺序保证**: hash值不仅取决于当前block的tokens，还依赖于前面所有blocks的hash
3. **前缀匹配自动实现**: 逐个比较hash值即可判断前缀是否相同

**链式哈希的具体工作过程：**

```python
# 位置：vllm/v1/core/kv_cache_utils.py:593-614
def request_block_hasher(request: Request) -> list[BlockHash]:
    block_size = 16  # 假设块大小为16
    start_token_idx = len(request.block_hashes) * block_size

    while True:
        end_token_idx = start_token_idx + block_size
        if end_token_idx > num_tokens:
            break  # 只处理完整的blocks

        # 提取当前block的tokens
        block_tokens = request.all_token_ids[start_token_idx:end_token_idx]

        # 计算当前block的hash（依赖于父block的hash）
        block_hash = hash_block_tokens(
            caching_hash_fn,
            prev_block_hash_value,  # 链式依赖
            block_tokens,           # 当前block的Token IDs
            extra_keys
        )

        new_block_hashes.append(block_hash)
        start_token_idx += block_size
        prev_block_hash_value = block_hash
```

**详细示例：**

```python
# 假设有两个请求：
# 请求A的Token IDs: [1,2,3,...,48]
# 请求B的Token IDs: [1,2,3,...,32, 100,101,...,116]

# 请求A的链式哈希计算：
# Block 0: tokens=[1,2,3,...,16]
#   hash_0 = hash(None, (1,2,3,...,16), None) = "abc123"
#
# Block 1: tokens=[17,18,...,32]
#   hash_1 = hash("abc123", (17,18,...,32), None) = "def456"
#
# Block 2: tokens=[33,34,...,48]
#   hash_2 = hash("def456", (33,34,...,48), None) = "ghi789"

# 请求B的链式哈希计算：
# Block 0: tokens=[1,2,3,...,16] (相同)
#   hash_0 = hash(None, (1,2,3,...,16), None) = "abc123" (相同！)
#
# Block 1: tokens=[17,18,...,32] (相同)
#   hash_1 = hash("abc123", (17,18,...,32), None) = "def456" (相同！)
#
# Block 2: tokens=[100,101,...,116] (不同)
#   hash_2 = hash("def456", (100,101,...,116), None) = "jkl012" (不同)

# 前缀匹配结果：
# Block 0和Block 1的hash完全相同 → 可以复用前32个token的KV cache
# Block 2的hash不同 → 需要重新计算剩余的KV cache
```

### 3.3 前缀匹配查找的具体实现

**缓存存储结构：**

```python
# 位置：vllm/v1/core/block_pool.py:34-61
class BlockHashToBlockMap:
    """
    哈希映射表：block_hash → KVCacheBlock
    支持一个hash对应多个block（使用dict存储）
    """
    def __init__(self):
        self._cache: dict[
            BlockHashWithGroupId, KVCacheBlock | dict[int, KVCacheBlock]
        ] = {}

    def get_one_block(self, key: BlockHashWithGroupId) -> KVCacheBlock | None:
        """根据hash获取任意一个匹配的block"""
        blocks = self._cache.get(key)
        if blocks is not None:
            if isinstance(blocks, KVCacheBlock):
                return blocks
            if isinstance(blocks, dict):
                return next(iter(blocks.values()))
        return None
```

**查找过程详解：**

```python
# 位置：vllm/v1/core/single_type_kv_cache_manager.py:446-456
for block_hash in itertools.islice(block_hashes, max_num_blocks):
    if cached_block := block_pool.get_cached_block(
        block_hash, kv_cache_group_ids
    ):
        # 找到匹配，复用KV cache
        computed_blocks.append(cached_block)
    else:
        break  # 立即停止匹配
```

**实际匹配示例：**

```python
# 缓存状态：
# cached_block_hash_to_block = {
#     "abc123": KVCacheBlock(block_id=5),   # Block 0的缓存
#     "def456": KVCacheBlock(block_id=12),  # Block 1的缓存
#     "ghi789": KVCacheBlock(block_id=20)   # Block 2的缓存
# }

# 新请求的block_hashes = ["abc123", "def456", "jkl012"]

# 匹配过程：
# 第1次迭代：block_hash = "abc123"
#   cached_block = block_pool.get_cached_block("abc123", [0])
#   → 返回 KVCacheBlock(block_id=5)  ✅ 匹配！
#   computed_blocks = [KVCacheBlock(block_id=5)]

# 第2次迭代：block_hash = "def456"
#   cached_block = block_pool.get_cached_block("def456", [0])
#   → 返回 KVCacheBlock(block_id=12)  ✅ 匹配！
#   computed_blocks = [KVCacheBlock(block_id=5), KVCacheBlock(block_id=12)]

# 第3次迭代：block_hash = "jkl012"
#   cached_block = block_pool.get_cached_block("jkl012", [0])
#   → 返回 None  ❌ 不匹配！
#   break  # 立即停止匹配

# 最终结果：复用了前32个token的KV cache (Block 0和Block 1)
```

### 3.4 只有完整Block才能匹配

```python
# 位置：vllm/v1/core/kv_cache_utils.py:595-597
end_token_idx = start_token_idx + block_size
if end_token_idx > num_tokens:
    # We only hash full blocks
    break
```

**为什么要这样设计？**

1. **缓存粒度统一**: 只缓存完整的block可以简化缓存管理逻辑
2. **性能考虑**: 频繁的小块缓存会增加哈希计算和查找开销
3. **内存效率**: 完整块缓存提高内存利用率，减少碎片
4. **实现简化**: 避免处理部分block的复杂逻辑

**影响：**

- 少于16个token的请求无法利用前缀缓存
- 部分匹配（如15个token相同）也无法命中缓存
- 如果请求长度不是16的倍数，最后不足16个token的部分无法缓存

### 3.5 前缀匹配的图示说明

**链式哈希的依赖关系：**

```
请求1: Token IDs [1,2,3,...,48]
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ Block 0         │    │ Block 1         │    │ Block 2         │
│ Tokens: [1..16] │    │ Tokens: [17..32]│    │ Tokens: [33..48]│
│                 │    │                 │    │                 │
│ hash_0 =        │    │ hash_1 =        │    │ hash_2 =        │
│ hash(None,      │──→ │ hash(hash_0,    │──→ │ hash(hash_1,    │
│   (1..16))      │    │   (17..32))     │    │   (33..48))     │
│ = "abc123"      │    │ = "def456"      │    │ = "ghi789"      │
└─────────────────┘    └─────────────────┘    └─────────────────┘

请求2: Token IDs [1,2,3,...,32, 100,101,...,116]
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ Block 0         │    │ Block 1         │    │ Block 2         │
│ Tokens: [1..16] │    │ Tokens: [17..32]│    │ Tokens: [100..116]│
│                 │    │                 │    │                 │
│ hash_0 =        │    │ hash_1 =        │    │ hash_2 =        │
│ hash(None,      │──→ │ hash(hash_0,    │──→ │ hash(hash_1,    │
│   (1..16))      │    │   (17..32))     │    │   (100..116))   │
│ = "abc123" ✅   │    │ = "def456" ✅   │    │ = "xyz999" ❌   │
└─────────────────┘    └─────────────────┘    └─────────────────┘

匹配结果：
✅ Block 0匹配: "abc123" == "abc123"
✅ Block 1匹配: "def456" == "def456"
❌ Block 2不匹配: "ghi789" != "xyz999"

可以复用前32个token的KV cache！
```

**哈希查找表的结构：**

```text
cached_block_hash_to_block = {
    "abc123" + group_id(0): KVCacheBlock(block_id=5, ref_cnt=2),
    "def456" + group_id(0): KVCacheBlock(block_id=12, ref_cnt=2),
    "ghi789" + group_id(0): KVCacheBlock(block_id=20, ref_cnt=1),
}

前缀匹配查找过程：
1. 查找 "abc123" + group_id(0) → 找到 block_id=5 ✅
2. 查找 "def456" + group_id(0) → 找到 block_id=12 ✅
3. 查找 "xyz999" + group_id(0) → 未找到 ❌ 停止匹配

结果：复用 [block_id=5, block_id=12]
```

### 3.6 为什么链式哈希能确保前缀匹配的正确性？

**数学原理：**

如果两个序列的前N个token完全相同，那么：

1. **第0个block**: hash(None,相同tokens) → 必然产生相同的hash_0
2. **第1个block**: hash(相同hash_0, 相同tokens) → 必然产生相同的hash_1
3. **第i个block**: hash(相同hash_{i-1}, 相同tokens) → 必然产生相同的hash_i

**反证法：**

假设两个序列的前N个token相同，但某个block的hash不同：

- 如果hash_i不同，那么要么hash_{i-1}不同，要么tokens不同
- 根据归纳假设，hash_{i-1}必须相同
- 如果tokens相同，hash_i必须相同
- 矛盾！因此如果前缀相同，所有对应block的hash都必须相同

**为什么不能使用独立哈希？**

```python
# 独立哈希（错误的做法）：
# hash_0 = hash((1,2,3,...,16))
# hash_1 = hash((17,18,...,32))     # 独立计算
# hash_2 = hash((33,34,...,48))     # 独立计算

# 问题：
# 请求A: [1,2,3,...,16, 17,18,...,32, 33,34,...,48]
# 请求B: [17,18,...,32, 1,2,3,...,16, 33,34,...,48]
#
# 使用独立哈希：
# hash_1_A = hash((17,18,...,32))  # 相同
# hash_1_B = hash((17,18,...,32))  # 相同
# hash_2_A = hash((1,2,3,...,16))  # 相同
# hash_2_B = hash((1,2,3,...,16))  # 相同
#
# 错误匹配！虽然tokens内容相同，但顺序不同
# 这会导致错误的前缀匹配结论

# 链式哈希（正确的做法）：
# hash_0_A = hash(None, (1,2,3,...,16))
# hash_1_A = hash(hash_0_A, (17,18,...,32))
# hash_0_B = hash(None, (17,18,...,32))  # 不同！
# hash_1_B = hash(hash_0_B, (1,2,3,...,16))  # 不同！
#
# 正确：链式哈希确保了顺序的重要性
```

---

## 分阶段复用策略

### 4.1 预填充阶段：跨轮次复用

**关键条件：`num_computed_tokens == 0`**

```python
# 位置：vllm/v1/core/sched/scheduler.py:610-614
if request.num_computed_tokens == 0:  # 只有首次调度
    # Get locally-cached tokens.
    new_computed_blocks, num_new_computed_tokens = (
        self.kv_cache_manager.get_computed_blocks(request)
    )
```

**复用流程：**

```python
# 请求A（历史轮次）
Request A: "Once upon a time there was a little princess"
→ 形成完整blocks并缓存到全局cached_block_hash_to_block

# 请求B（当前轮次预填充）
Request B: "Once upon a time there was a little princess"
→ num_computed_tokens = 0
→ 查找cached_block_hash_to_block
→ 找到请求A的匹配blocks
→ 复用KV cache，跳过计算
```

### 4.2 推理生成阶段：本轮内复用

**关键条件：`num_computed_tokens > 0`**

```python
# 推理阶段
request.num_computed_tokens = 17  # > 0
if request.num_computed_tokens == 0:  # 条件不满足
    # 不会调用get_computed_blocks
    # 不会跨请求查找
```

**复用机制：**

```python
# 当前请求推理生成
prompt_tokens: "Once upon a time" (复用历史缓存)
生成token1: "there" → 使用prompt_tokens的KV cache
生成token2: "was" → 使用prompt_tokens + token1的KV cache  
生成token3: "a" → 使用之前所有tokens的KV cache
```

### 4.3 关键区别

| 特性 | 预填充阶段 | 推理生成阶段 |
|------|------------|--------------|
| **复用范围** | 历史轮次 + 本轮prompt | 本轮推理内累积 |
| **查找机制** | 主动查找全局缓存 | 被动使用已有内容 |
| **条件** | `num_computed_tokens == 0` | `num_computed_tokens > 0` |
| **复用内容** | 其他请求的完整blocks | 当前请求的tokens序列 |
| **命中率** | 高（相同prompt常见） | 低（生成路径各异） |

---

## 跨轮次复用

### 5.1 全局缓存映射表

```python
class BlockPool:
    # 全局共享的缓存映射表
    cached_block_hash_to_block: BlockHashToBlockMap
```

### 5.2 缓存写入时机

```python
# 位置：vllm/v1/core/block_pool.py:271-272
def cache_full_blocks(...):
    for i, blk in enumerate(new_full_blocks):
        block_hash_with_group_id = make_block_hash_with_group_id(
            block_hash, kv_cache_group_id
        )
        blk.block_hash = block_hash_with_group_id
        # 添加到全局缓存
        self.cached_block_hash_to_block.insert(block_hash_with_group_id, blk)
```

cache_full_blocks 的调用时机：

✅ 形成新的完整block时 - 主要触发条件

✅ 预填充和推理阶段都可能调用 - 不分阶段

✅ 每次只缓存新的完整blocks - 增量缓存

✅ 将block加入全局缓存 - 实现跨请求复用

### 5.3 跨轮次复用流程

```python
# ===== 轮次1：请求A推理 =====
Request A: "The quick brown fox jumps over the lazy dog"
→ 形成完整Block: "The quick brown fox jumps over the "
→ 缓存: cached_block_hash_to_block[hash_A] = Block_A

# ===== 轮次2：请求B预填充 =====  
Request B: "The quick brown fox jumps over the moon"
→ 计算block hashes: hash_B1
→ hash_B1 == hash_A (前半部分相同)
→ 查找cached_block_hash_to_block
→ 找到匹配，复用Block_A的16个tokens
→ 只需要计算"moon"部分
```

### 5.4 时间维度复用

- ✅ 可以复用上一轮、上上轮、甚至几天前请求的KV cache
- ✅ 跨时间、跨批次的复用
- ✅ 只要有相同的block hash就能复用

---

## 性能优化

### 6.1 性能收益示例

#### 场景1：相同prompt多用户

```python
# ❌ 没有前缀缓存
100个用户 × "What is AI?" = 100 × 完整推理成本

# ✅ 有前缀缓存
第1个用户: 100% 计算
第2-100个用户: 20% 计算（复用prompt，只生成不同答案）
总计算 = 1×100% + 99×20% = 119% (而不是10000%)
节省 = 88% 计算成本
```

#### 场景2：多轮对话

```python
# 轮次1
User: "Tell me a story about a princess"
→ 生成: "Once upon a time, in a faraway kingdom, "
→ 缓存: Block[hash_story_1]

# 轮次2（相同上下文）
User: "Tell me a story about a princess"  
→ 预填充: 复用Block[hash_story_1]
→ 继续: "lived a beautiful princess..."
```

### 6.2 内存管理

#### 引用计数机制

```python
# 位置：vllm/v1/core/block_pool.py:418-422
def free_blocks(self, ordered_blocks):
    for block in blocks_list:
        block.ref_cnt -= 1  # 减少引用计数
    
    # 只有ref_cnt==0的blocks才真正释放
    self.free_block_queue.append_n(
        [block for block in blocks_list if block.ref_cnt == 0]
    )
```

#### 缓存淘汰策略

```python
# 位置：vllm/v1/core/block_pool.py:354-389
def _maybe_evict_cached_block(self, block):
    if block_hash is None:
        return False
    
    # 从全局缓存中移除
    if self.cached_block_hash_to_block.pop(block_hash) is None:
        return False
    
    block.reset_hash()  # 重置hash
    return True
```

### 6.3 Block粒度的权衡

**优势：**

- 减少哈希计算和比较次数
- 提高内存管理效率
- 适合内存分配的自然单位

**劣势：**

- 小于16个token的请求无法利用缓存
- 部分匹配会浪费计算资源

---

## 总结

### 核心设计原则

1. **Token IDs哈希匹配**：基于确定性的Token IDs，而非向量
2. **Block粒度管理**：以16个token为单位进行缓存和匹配
3. **分阶段策略**：
   - 预填充：跨轮次全局复用
   - 推理：本轮内累积复用
4. **引用计数管理**：智能的内存管理和淘汰

### 关键实现细节

- **哈希计算**：链式结构，每个block依赖前一个block
- **匹配条件**：`num_computed_tokens == 0` 决定是否查找缓存
- **缓存写入**：形成完整block时立即缓存到全局映射表
- **内存优化**：引用计数+LRU淘汰策略

### 性能影响

这种设计使得vLLM能够：

- 显著降低重复prompt的处理成本
- 支持跨时间、跨批次的KV cache复用
- 在保证推理效率的同时最大化前缀缓存收益

这是vLLM作为高性能推理系统的核心优化策略之一。

## 参考代码位置

- Tokenization: `vllm/inputs/preprocess.py:68`
- 前缀匹配: `vllm/v1/core/kv_cache_utils.py:535`
- Block管理: `vllm/v1/core/block_pool.py:211`
- 调度逻辑: `vllm/v1/core/sched/scheduler.py:610`
- KV缓存接口: `vllm/v1/kv_cache_interface.py:1`