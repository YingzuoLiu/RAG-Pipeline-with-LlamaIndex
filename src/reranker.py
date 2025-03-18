# src/reranker.py
import logging
from typing import List, Dict, Any
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class KeywordReranker:
    """关键词重新排序器"""
    
    def __init__(self, keywords: List[str], boost_factor: float = 0.2):
        """
        初始化关键词重新排序器
        
        Args:
            keywords: 要提升的关键词列表
            boost_factor: 提升因子
        """
        self.keywords = [k.lower() for k in keywords]
        self.boost_factor = boost_factor
        logger.info(f"Initialized KeywordReranker with {len(keywords)} keywords")
    
    def postprocess_nodes(self, nodes):
        """
        对节点进行后处理，提升包含关键词的节点
        
        Args:
            nodes: 要处理的节点列表
        
        Returns:
            处理后的节点列表
        """
        if not nodes:
            return nodes
        
        for node in nodes:
            text = node.get_content().lower()
            keyword_count = sum(1 for keyword in self.keywords if keyword in text)
            
            # 计算加权分数
            if keyword_count > 0:
                boost = self.boost_factor * keyword_count
                # 在原有得分基础上进行提升
                new_score = node.get_score() * (1 + boost)
                node.score = min(new_score, 1.0)  # 确保得分不超过1
        
        # 重新排序
        sorted_nodes = sorted(nodes, key=lambda x: x.get_score(), reverse=True)
        
        return sorted_nodes

class DiversityReranker:
    """多样性重新排序器"""
    
    def __init__(self, diversity_weight: float = 0.3, threshold: float = 0.85):
        """
        初始化多样性重新排序器
        
        Args:
            diversity_weight: 多样性权重
            threshold: 相似度阈值
        """
        self.diversity_weight = diversity_weight
        self.threshold = threshold
        logger.info(f"Initialized DiversityReranker with weight={diversity_weight}, threshold={threshold}")
    
    def postprocess_nodes(self, nodes):
        """
        对节点进行后处理，提升内容多样性
        
        Args:
            nodes: 要处理的节点列表
        
        Returns:
            处理后的节点列表
        """
        if len(nodes) <= 1:
            return nodes
        
        # 从最高分开始
        result_nodes = [nodes[0]]
        remaining_nodes = nodes[1:]
        
        while remaining_nodes and len(result_nodes) < len(nodes):
            max_score = -float('inf')
            best_node_idx = 0
            
            for i, node in enumerate(remaining_nodes):
                # 基础分数
                base_score = node.get_score()
                
                # 计算与已选节点的平均相似度
                similarity_penalty = 0
                for result_node in result_nodes:
                    # 这里使用简单文本重叠作为相似度度量
                    overlap = self._calculate_text_overlap(
                        node.get_content(), result_node.get_content()
                    )
                    similarity_penalty += overlap
                
                avg_similarity = similarity_penalty / len(result_nodes)
                
                # 计算最终分数：原始分数 - 多样性惩罚
                final_score = base_score - (self.diversity_weight * avg_similarity)
                
                if final_score > max_score:
                    max_score = final_score
                    best_node_idx = i
            
            # 添加最优节点
            result_nodes.append(remaining_nodes[best_node_idx])
            remaining_nodes.pop(best_node_idx)
        
        return result_nodes
    
    def _calculate_text_overlap(self, text1: str, text2: str) -> float:
        """
        计算两段文本的重叠度
        
        Args:
            text1: 第一段文本
            text2: 第二段文本
        
        Returns:
            文本重叠度
        """
        # 简单实现：使用共同单词的比例
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        common_words = words1.intersection(words2)
        similarity = len(common_words) / min(len(words1), len(words2))
        
        return similarity