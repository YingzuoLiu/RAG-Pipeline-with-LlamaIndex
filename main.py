import os
import sys
import logging
from pathlib import Path
import argparse
import gc
import torch

from src.index_builder import IndexBuilder
from src.query_engine import DeepseekRAGEngine
from src.evaluator import RAGEvaluator
from src.reranker import KeywordReranker, DiversityReranker

# 设置内存优化
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.backends.cudnn.benchmark = True

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('rag_pipeline.log')
    ]
)
logger = logging.getLogger(__name__)

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="DeepSeek RAG Pipeline")
    parser.add_argument("--data", type=str, default="data/paul_graham_essays.txt", 
                        help="Path to the data file or directory")
    parser.add_argument("--eval", action="store_true", 
                        help="Run evaluation")
    parser.add_argument("--interactive", action="store_true", 
                        help="Run in interactive mode")
    
    args = parser.parse_args()
    
    # 确保数据路径存在
    if not os.path.exists(args.data):
        logger.error(f"Data path {args.data} does not exist!")
        return
    
    # 清理内存
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # 初始化索引构建器
    logger.info("Initializing index builder...")
    index_builder = IndexBuilder(
        chunk_size=512,
        chunk_overlap=50
    )
    
    # 加载和索引文档
    logger.info(f"Loading and indexing documents from {args.data}...")
    index = index_builder.load_and_index_documents(
        data_path=args.data,
        #index_path=args.index
    )
    
    # 初始化RAG引擎
    logger.info("Initializing DeepSeek RAG engine...")
    rag_engine = DeepseekRAGEngine(
        index=index,
        similarity_top_k=5
    )
    
    # 运行评估
    if args.eval:
        logger.info("Running evaluation...")
        
        # 示例测试查询和地面真实关键词
        test_queries = [
            "What advice does Paul Graham have for early stage entrepreneurs?",
            "What are the advantages of programming languages?",
            "How to identify promising startup ideas?",
            "What are Paul Graham's views on venture capital?",
            "Why is Silicon Valley the center of tech innovation?"
        ]
        
        ground_truth_keywords = [
            ["startup", "founder", "entrepreneur", "business"],
            ["programming language", "code", "software", "lisp"],
            ["startup idea", "idea", "potential", "problem"],
            ["VC", "venture capital", "investor", "funding"],
            ["Silicon Valley", "valley", "tech hub", "innovation"]
        ]
        
        # 创建评估器并运行评估
        evaluator = RAGEvaluator()
        metrics = evaluator.evaluate_retrieval(
            rag_engine=rag_engine,
            test_queries=test_queries,
            ground_truth_keywords=ground_truth_keywords,
            output_dir="."
        )
        
        logger.info(f"Evaluation results: Hit Rate = {metrics['hit_rate']:.4f}, MRR = {metrics['mrr']:.4f}")
    
    # 交互模式
    if args.interactive:
        logger.info("Starting interactive mode. Type 'exit' to quit.")
        print("\n" + "="*50)
        print("DeepSeek RAG Interactive Mode")
        print("Type 'exit' to quit")
        print("="*50 + "\n")
        
        while True:
            query = input("\nEnter your query: ")
            if query.lower() == 'exit':
                break
            
            # 查询RAG引擎
            print("\nRetrieving relevant information...")
            response, nodes = rag_engine.query(query, verbose=True)
            
            # 打印检索到的节点
            print("\nRetrieved relevant information:")
            for i, node in enumerate(nodes[:3]):  # 只显示前3个
                print(f"\n--- Chunk {i+1} (Score: {node['score']:.4f}) ---")
                preview = node['text'][:150] + "..." if len(node['text']) > 150 else node['text']
                print(preview)
            
            # 打印生成的回答
            print("\nGenerated answer:")
            print(response)
            print("\n" + "-"*50)

if __name__ == "__main__":
    main()