# 分层检索rag

![alt text](image.png)

分层检索（Hierarchical Retrieval）是一种是一种将文档按“摘要 → 章节 → 片段”三级结构组织，并通过逐层导航高效检索相关信息的技术。它解决了传统“扁平检索”在面对海量数据时精度低、噪声多、计算开销大的问题。

## 为什么需要分层检索

在 RAG 或信息检索系统中，直接对百万级文档做向量检索会遇到：

| 问题 | 说明 |
|------|------|
| ❌ **高噪声** | Top-K 结果可能包含大量表面相似但语义无关的片段 |
| ❌ **上下文碎片化** | 检索到的是孤立句子，缺乏整体逻辑 |
| ❌ **计算成本高** | 全库扫描耗时长，尤其在长上下文场景 |
| ❌ **推理能力弱** | LLM 难以从零散片段中进行复杂推理 |

```mermaid
graph TD
    Doc[📄 原文档: medical_ai_report.pdf] --> Parse[🔍 解析: 提取文本 + 识别标题结构]
    
    subgraph "分层构建过程"
        Parse --> Split[✂️ 切分: 按标题层级分割]
        
        Split --> L2[Level 2: 片段层<br>Chunk 2.2.1: ResNet在CT...<br>Chunk 2.2.2: 准确率对比...]
        Split --> L1[Level 1: 章节层<br>Section 2.2: 深度学习诊断准确率分析]
        Split --> L0[Level 0: 摘要层<br>Summary 2: 医疗影像诊断应用综述]
        
        L2 -.->|parent_id | L1
        L1 -.->|parent_id | L0
        
        L0 --> Emb0[🧠 向量化 Summary 2]
        L1 --> Emb1[🧠 向量化 Section 2.2]
        L2 --> Emb2[🧠 向量化 Chunk 2.2.1/2.2.2]
    end
    
    Emb0 & Emb1 & Emb2 --> DB[(💾 向量数据库)]
```

这里的 Summary Section Chunk 都是存储各层级向量化存储的库。

## 文档分层结构（Document Hierarchy Structure）

以《医疗AI研究报告.pdf》为例，构建三层索引：

| 层级 | 名称 | 内容 | 作用 |
|------|------|------|------|
| **第1层：摘要层（Summary Level）** | 摘要索引 | 每章生成一个简短摘要（如 50~100 字） | 快速定位主题 |
| **第2层：章节层（Section Level）** | 章节索引 | 每个章节标题 + 页面数 | 缩小搜索范围 |
| **第3层：片段层（Chunk Level）** | 片段索引 | 实际文本块（如 512-token） | 获取精确内容 |

## 文档入库

```mermaid
graph TB
    subgraph "原始PDF文档"
        A[AI_Information.pdf<br/>15页]
    end

    subgraph "页面提取"
        A --> B1[Page 1]
        A --> B2[Page 2]
        A --> B3[Page 3]
        A --> B4[...]
        A --> B5[Page 15]
    end

    subgraph "双层索引生成"
        B1 --> C1[摘要1<br/>LLM生成]
        B1 --> D1[片段1.1, 1.2, 1.3<br/>chunk_size=1000]
        
        B2 --> C2[摘要2]
        B2 --> D2[片段2.1, 2.2, 2.3]
        
        B3 --> C3[摘要3]
        B3 --> D3[片段3.1, 3.2, 3.3]
    end

    subgraph "向量数据库 - 双存储结构"
        C1 & C2 & C3 --> E1[Summary_Store<br/>15条摘要向量]
        D1 & D2 & D3 --> E2[Detailed_Store<br/>47条片段向量]
    end

    style A fill:#e3f2fd
    style E1 fill:#c8e6c9
    style E2 fill:#fff9c4

```

```py
def process_document_hierarchically(pdf_path, chunk_size=1000, chunk_overlap=200):
    """
    将文档处理为分层索引。

    参数:
        pdf_path (str): PDF文件的路径
        chunk_size (int): 每个详细块的大小
        chunk_overlap (int): 块之间的重叠部分

    返回:
        Tuple[SimpleVectorStore, SimpleVectorStore]: 摘要和详细的向量存储
    """
    # 从PDF中提取页面
    pages = extract_text_from_pdf(pdf_path)
    
    # 为每一页创建摘要
    print("Generating page summaries...")
    summaries = []
    for i, page in enumerate(pages):
        print(f"Summarizing page {i+1}/{len(pages)}...")
        summary_text = generate_page_summary(page["text"])
        
        # 创建摘要元数据
        summary_metadata = page["metadata"].copy()
        summary_metadata.update({"is_summary": True})
        
        # 将摘要文本和元数据追加到摘要列表中
        summaries.append({
            "text": summary_text,
            "metadata": summary_metadata
        })
    
    # 为每一页创建详细的块
    detailed_chunks = []
    for page in pages:
        # 对页面的文本进行分块
        page_chunks = chunk_text(
            page["text"], 
            page["metadata"], 
            chunk_size, 
            chunk_overlap
        )
        # 使用当前页面的块扩展详细的块列表
        detailed_chunks.extend(page_chunks)
    
    print(f"Created {len(detailed_chunks)} detailed chunks")
    
    # 为摘要创建嵌入
    print("Creating embeddings for summaries...")
    summary_texts = [summary["text"] for summary in summaries]
    summary_embeddings = create_embeddings(summary_texts)
    
    # 为详细的块创建嵌入
    print("Creating embeddings for detailed chunks...")
    chunk_texts = [chunk["text"] for chunk in detailed_chunks]
    chunk_embeddings = create_embeddings(chunk_texts)
    
    # 创建向量存储
    summary_store = SimpleVectorStore()
    detailed_store = SimpleVectorStore()
    
    # 向摘要存储中添加摘要
    for i, summary in enumerate(summaries):
        summary_store.add_item(
            text=summary["text"],
            embedding=summary_embeddings[i],
            metadata=summary["metadata"]
        )
    
    # 向详细的存储中添加块
    for i, chunk in enumerate(detailed_chunks):
        detailed_store.add_item(
            text=chunk["text"],
            embedding=chunk_embeddings[i],
            metadata=chunk["metadata"]
        )
    
    print(f"Created vector stores with {len(summaries)} summaries and {len(detailed_chunks)} chunks")
    return summary_store, detailed_store
```

## 分层检索流程（Hierarchical Search Process）

### ✅ 示例查询

> “深度学习在医疗影像诊断中的最新应用和技术挑战”

```mermaid
sequenceDiagram
    participant User as 👤 用户
    participant Retriever as ⚙️ 检索引擎
    participant VecDB as 💾 向量数据库

    User->>Retriever: "ResNet 在 CT 诊断中的准确率如何？"
    
    rect rgb(240, 248, 255)
        Note right of Retriever: <b>Step 1: 摘要层检索 (Level 0)</b>
        Retriever->>VecDB: 搜索 vector(query)<br/>Filter: level==0
        VecDB-->>Retriever: 返回 Top 1 摘要:<br/>ID: node_sum_02 (医疗影像)
    end

    rect rgb(255, 250, 240)
        Note right of Retriever: <b>Step 2: 章节层定位 (Level 1)</b>
        Retriever->>VecDB: 查询 node_sum_02 的子节点<br/>Filter: parent_id=='node_sum_02' AND level==1
        VecDB-->>Retriever: 返回相关章节:<br/>ID: node_sec_22 (准确率分析)<br/>ID: node_sec_24 (多模态融合)
        Retriever->>Retriever: 过滤出最相关章节: node_sec_22
    end

    rect rgb(240, 255, 240)
        Note right of Retriever: <b>Step 3: 片段层精搜 (Level 2)</b>
        Retriever->>VecDB: 搜索 vector(query)<br/>Filter: parent_id=='node_sec_22' AND level==2
        VecDB-->>Retriever: 返回 Top 2 片段:<br/>1. ResNet在CT中表现...<br/>2. 实验数据准确率94.5%...
    end

    Retriever->>User: 🤖 生成答案:<br/>"根据报告，ResNet在CT诊断中准确率达94.5%..."
```

### 🔁 检索步骤（逐层递进）

#### **步骤1：摘要层检索（粗筛）**

- 查询：`"医疗影像诊断最新应用"`
- 在所有**章节摘要**中进行向量检索
- 匹配结果：
  - ✅ 摘要2: 医疗影像诊断应用 (0.91)
  - ✅ 摘要3: 技术挑战与限制 (0.83)

> 📌 结果：确定用户关心的是**第二章和第三章**

#### **步骤2：章节层检索（中筛）**

- 只在**匹配的章节内**继续搜索
- 查询：`"深度学习在CT诊断中的应用"`
- 在“第2章”下查找相关子章节
- 匹配结果：
  - ✅ 2.2 深度学习诊断准确率分析 (0.89)
  - ✅ 2.4 多模态影像融合技术 (0.86)


> 📌 结果：聚焦到两个具体技术点

#### **步骤3：片段层检索（精筛）**
- 在**匹配的章节内**进一步细化
- 查询：`"ResNet在CT诊断中的表现"`
- 查找对应片段
- 匹配结果：
  - ✅ Chunk 2.2.1: ResNet在CT诊断
  - ✅ Chunk 2.2.2: 准确率对比实验

> 📌 结果：获取最相关的原始内容用于生成答案