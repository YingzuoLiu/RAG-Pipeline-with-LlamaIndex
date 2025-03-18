# Retrieval Augmented Generation Pipeline with LlamaIndex and DeepSeek

使用DeepSeek 1.3B和LlamaIndex构建的检索增强生成(RAG)系统，用于智能文档检索和问答。

## 功能特点

- 使用DeepSeek 1.3B作为生成模型
- 基于LlamaIndex框架进行文档索引和检索
- 使用FAISS向量数据库进行高效检索
- 支持文档分块和嵌入生成
- 包含评估工具，可测量Hit Rate和MRR
- 实现了关键词和多样性重排序器

## 安装

1. 创建Conda环境
```bash
conda create -n rag_env python=3.9
conda activate rag_env
conda install numpy pandas matplotlib
pip install -r requirements.txt
