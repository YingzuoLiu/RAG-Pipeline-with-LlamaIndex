# src/query_engine.py
import os
import logging
from typing import List, Dict, Any, Tuple
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from llama_index import VectorStoreIndex
from llama_index.retrievers import VectorIndexRetriever
from llama_index.query_engine import RetrieverQueryEngine
from llama_index.postprocessor import SimilarityPostprocessor
from llama_index.llms.huggingface import HuggingFaceLLM

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DeepseekRAGEngine:
    def __init__(
        self,
        index: VectorStoreIndex,
        model_name: str = "deepseek-ai/deepseek-coder-1.3b-base",
        similarity_top_k: int = 3,
        similarity_cutoff: float = 0.2,
    ):
        """
        初始化DeepSeek RAG引擎
        
        Args:
            index: 已构建的向量索引
            model_name: DeepSeek模型名称
            similarity_top_k: 检索的顶部K个文档
            similarity_cutoff: 相似度截断阈值
        """
        self.index = index
        self.similarity_top_k = similarity_top_k
        self.similarity_cutoff = similarity_cutoff
        
        # 设置设备
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {device}")
        
        # 初始化模型和分词器
        logger.info(f"Loading DeepSeek model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # 避免OOM错误的模型加载设置
        if device == "cuda":
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
                device_map="auto"
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map=None  # CPU模式
            )
        
        # 构建LlamaIndex兼容的LLM
        self.llm = HuggingFaceLLM(
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=512,
            generate_kwargs={"temperature": 0.7, "do_sample": True}
        )
        
        # 构建检索器
        self.retriever = VectorIndexRetriever(
            index=self.index,
            similarity_top_k=self.similarity_top_k,
        )
        
        # 构建后处理器
        self.postprocessor = SimilarityPostprocessor(similarity_cutoff=self.similarity_cutoff)
        
        # 构建查询引擎 - 处理API兼容性问题
        try:
            # 尝试新版本API（带llm参数）
            self.query_engine = RetrieverQueryEngine(
                retriever=self.retriever,
                node_postprocessors=[self.postprocessor],
                llm=self.llm
            )
        except TypeError:
            # 尝试旧版本API（不带llm参数）
            self.query_engine = RetrieverQueryEngine(
                retriever=self.retriever,
                node_postprocessors=[self.postprocessor]
            )
            logger.info("Using legacy RetrieverQueryEngine without LLM parameter")
        
        logger.info("DeepseekRAGEngine initialized successfully")
    
    def retrieve_relevant_nodes(self, query: str) -> List[Dict[str, Any]]:
        """
        检索与查询相关的节点
        
        Args:
            query: 查询文本
        
        Returns:
            检索到的相关节点列表
        """
        nodes = self.retriever.retrieve(query)
        processed_nodes = self.postprocessor.postprocess_nodes(nodes)
        
        # 构建结果列表
        results = []
        for node in processed_nodes:
            results.append({
                "text": node.get_content(),
                "score": node.get_score(),
                "id": node.node_id,
            })
        
        return results
    
    def query(self, query: str, verbose: bool = False) -> Tuple[str, List[Dict[str, Any]]]:
        nodes = self.retrieve_relevant_nodes(query)
        
        if verbose:
            logger.info(f"Query: {query}")
            logger.info(f"Retrieved {len(nodes)} relevant nodes")
            for i, node in enumerate(nodes):
                logger.info(f"Node {i+1} (score: {node['score']:.4f}): {node['text'][:100]}...")
        
        # 如果没有检索到任何节点，创建一个默认响应
        if not nodes:
            logger.warning("No relevant documents found for the query")
            return "I couldn't find specific information about that in the provided documents.", nodes
        
        # 执行查询
        try:
            response = self.query_engine.query(query)
            return str(response), nodes
        except Exception as e:
            logger.error(f"Error during query: {e}")
            # 使用检索到的节点直接生成响应
            context = "\n\n".join([node["text"] for node in nodes[:3]])
            
            # 这里是备用生成方法
            try:
                prompt = f"""Based on the following information, please answer the question.

    Information:
    {context[:300]}...

    Question: {query}

    Answer:"""
                
                inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=512,
                    do_sample=True,
                    temperature=0.7,
                )
                response = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
                
                return response, nodes
            except:
                # 如果备用方法也失败，返回简单响应
                return f"Based on the retrieved information: {context[:300]}...", nodes