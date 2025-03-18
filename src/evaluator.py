# src/evaluator.py
import logging
from typing import List, Dict, Any, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAGEvaluator:
    def __init__(self):
        """初始化评估器"""
        self.results = []
        self.metrics = {}
    
    def evaluate_retrieval(
        self, 
        rag_engine, 
        test_queries: List[str], 
        ground_truth_keywords: List[List[str]],
        output_dir: str = "."
    ) -> Dict[str, float]:
        """
        评估检索性能
        
        Args:
            rag_engine: RAG查询引擎
            test_queries: 测试查询列表
            ground_truth_keywords: 每个查询对应的地面真实关键词列表
            output_dir: 输出目录
        
        Returns:
            评估指标字典
        """
        if len(test_queries) != len(ground_truth_keywords):
            raise ValueError("Test queries and ground truth must have the same length")
        
        logger.info(f"Starting evaluation with {len(test_queries)} test queries")
        
        hit_count = 0
        reciprocal_ranks = []
        
        for idx, (query, keywords) in enumerate(zip(test_queries, ground_truth_keywords)):
            # 检索相关节点
            logger.info(f"Evaluating query {idx+1}/{len(test_queries)}: {query}")
            _, retrieved_nodes = rag_engine.query(query)
            
            # 检查是否命中
            hit = False
            rank = float('inf')
            
            for i, node in enumerate(retrieved_nodes):
                node_text = node['text'].lower()
                # 检查是否包含任何关键词
                for keyword in keywords:
                    if keyword.lower() in node_text:
                        hit = True
                        rank = min(rank, i + 1)
                        break
                if hit and rank == i + 1:
                    break
            
            # 计算指标
            if hit:
                hit_count += 1
                reciprocal_rank = 1.0 / rank
            else:
                reciprocal_rank = 0.0
                
            reciprocal_ranks.append(reciprocal_rank)
            
            # 保存结果
            self.results.append({
                'query': query,
                'hit': hit,
                'rank': rank if hit else None,
                'reciprocal_rank': reciprocal_rank,
                'top_node_score': retrieved_nodes[0]['score'] if retrieved_nodes else None,
                'retrieved_count': len(retrieved_nodes)
            })
            
            logger.info(f"  - Hit: {hit}, Rank: {rank if hit else 'N/A'}, RR: {reciprocal_rank:.4f}")
        
        # 计算总体指标
        hit_rate = hit_count / len(test_queries)
        mrr = np.mean(reciprocal_ranks)
        
        self.metrics = {
            'hit_rate': hit_rate,
            'mrr': mrr,
        }
        
        logger.info(f"Evaluation complete: Hit Rate = {hit_rate:.4f}, MRR = {mrr:.4f}")
        
        # 保存结果为CSV
        output_path = Path(output_dir) / "evaluation_results.csv"
        pd.DataFrame(self.results).to_csv(output_path, index=False)
        logger.info(f"Results saved to {output_path}")
        
        # 生成可视化
        self._visualize_metrics(output_dir)
        
        return self.metrics
    
    def _visualize_metrics(self, output_dir: str = "."):
        """
        可视化评估指标
        
        Args:
            output_dir: 输出目录
        """
        plt.figure(figsize=(12, 5))
        
        # 命中率
        plt.subplot(1, 2, 1)
        plt.bar(['Hit Rate'], [self.metrics['hit_rate']], color='steelblue')
        plt.title(f'Hit Rate: {self.metrics["hit_rate"]:.4f}', fontsize=14)
        plt.ylim(0, 1)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # MRR
        plt.subplot(1, 2, 2)
        plt.bar(['MRR'], [self.metrics['mrr']], color='forestgreen')
        plt.title(f'Mean Reciprocal Rank: {self.metrics["mrr"]:.4f}', fontsize=14)
        plt.ylim(0, 1)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        
        # 保存图表
        output_path = Path(output_dir) / "evaluation_metrics.png"
        plt.savefig(output_path)
        logger.info(f"Visualization saved to {output_path}")
        
        # 个别查询的评估
        if self.results:
            plt.figure(figsize=(14, 6))
            df = pd.DataFrame(self.results)
            
            # 绘制每个查询的倒数排名
            plt.bar(range(len(df)), df['reciprocal_rank'], color='teal')
            plt.xlabel('Query Index', fontsize=12)
            plt.ylabel('Reciprocal Rank', fontsize=12)
            plt.title('Reciprocal Rank for Each Query', fontsize=14)
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.xticks(range(len(df)), range(1, len(df) + 1))
            
            # 保存图表
            output_path = Path(output_dir) / "query_performance.png"
            plt.savefig(output_path)
            logger.info(f"Query performance visualization saved to {output_path}")