# src/index_builder.py
import os
import shutil
import faiss
import numpy as np
from typing import List, Dict, Any
from pathlib import Path
import logging

from llama_index import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    Document,
    ServiceContext,
    StorageContext,
)
from llama_index.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.faiss import FaissVectorStore

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class IndexBuilder:
    def __init__(
        self,
        embed_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        chunk_size: int = 512,
        chunk_overlap: int = 50,
    ):
        """
        初始化索引构建器
        
        Args:
            embed_model: 使用的嵌入模型名称
            chunk_size: 文档分块大小
            chunk_overlap: 文档分块重叠大小
        """
        self.embed_model = HuggingFaceEmbedding(model_name=embed_model)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # 设置解析器
        self.node_parser = SentenceSplitter(
            chunk_size=self.chunk_size, 
            chunk_overlap=self.chunk_overlap
        )
        
        logger.info(f"Initialized IndexBuilder with {embed_model}, chunk_size={chunk_size}, chunk_overlap={chunk_overlap}")
    
    def load_and_index_documents(self, data_path: str, index_path: str = "data/index") -> VectorStoreIndex:
        """
        加载文档并创建索引
        
        Args:
            data_path: 文档文件的路径
            index_path: 索引存储的路径
        
        Returns:
            构建好的向量索引
        """
        # 创建服务上下文，明确指定llm=None以禁用默认的OpenAI LLM
        service_context = ServiceContext.from_defaults(
            embed_model=self.embed_model,
            node_parser=self.node_parser,
            llm=None  # 禁用默认的OpenAI LLM
        )
        
        # 检查路径是否为单个文件
        if os.path.isfile(data_path):
            with open(data_path, 'r', encoding='utf-8') as f:
                text = f.read()
            documents = [Document(text=text)]
            logger.info(f"Loaded single file: {data_path}")
        else:
            # 否则作为目录处理
            documents = SimpleDirectoryReader(data_path).load_data()
            logger.info(f"Loaded directory: {data_path} with {len(documents)} documents")
        
        # 如果索引目录已存在，先删除它
        if os.path.exists(index_path):
            logger.info(f"Removing existing index at {index_path}")
            shutil.rmtree(index_path)
        
        logger.info(f"Creating new index at {index_path}")
        os.makedirs(index_path, exist_ok=True)
        
        # 创建FAISS索引
        try:
            # 获取嵌入维度
            embed_dim = len(self.embed_model.get_text_embedding("test"))
            logger.info(f"Using embedding dimension: {embed_dim}")
            
            # 创建FAISS索引
            faiss_index = faiss.IndexFlatL2(embed_dim)
            
            # 创建向量存储
            vector_store = FaissVectorStore(faiss_index=faiss_index)
            storage_context = StorageContext.from_defaults(vector_store=vector_store)
            
            # 构建索引
            index = VectorStoreIndex.from_documents(
                documents, 
                storage_context=storage_context,
                service_context=service_context
            )
            
            # 尝试保存索引 - 我们不在这里保存，因为它可能导致错误
            # 我们将在内存中使用索引
            logger.info(f"Index built successfully (in-memory only)")
            return index
            
        except Exception as e:
            logger.error(f"Error creating FAISS index: {e}")
            logger.info("Falling back to default in-memory index")
            
            # 如果FAISS不起作用，回退到默认的简单索引
            index = VectorStoreIndex.from_documents(
                documents,
                service_context=service_context
            )
            return index