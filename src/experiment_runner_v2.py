import time
import os
import json
import gc
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import torch

from chunker_v2 import HierarchicalChunker
from metrics import MetricsEvaluator
from performance_monitor import PeakMemoryMonitor
from seed_utils import set_global_seed
from rag_pipeline_v2 import RAGPipeline
from vector_store_v2 import VectorStore

class ExperimentRunner:
    """
    Class for testing RAG system performance with different parent/child size, overlap and temperature parameters
    LlamaIndex HierarchicalNodeParser + AutoMergingRetriever kullanır
    """

    def __init__(self, book_path: str, test_questions_path: str, results_dir: str = "results",
                 seed: Optional[int] = None):
        self.book_path = book_path
        self.test_questions_path = test_questions_path
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        self.seed = seed

        set_global_seed(seed)

        # Load book text
        with open(book_path, 'r', encoding='utf-8') as f:
            self.book_text = f.read()

        # Load test questions
        self.test_df = pd.read_csv(test_questions_path)

        # Create evaluator
        self.evaluator = MetricsEvaluator()

    def run_single_experiment(self, parent_size: int, child_size: int, temperature: float, chunk_overlap: int = 20,
                             vs: VectorStore = None, nodes: List = None, node_mapping: Dict = None,
                             precomputed_chunking_time: float = None, precomputed_indexing_time: float = None) -> Dict:
        """
        Run experiment with a single configuration

        Args:
            parent_size: Parent chunk size
            child_size: Child chunk size
            temperature: LLM temperature
            chunk_overlap: Chunk overlap amount
            vs: Ready VectorStore (optional - if provided, chunking/indexing is skipped)
            nodes: Ready nodes (optional)
            node_mapping: Ready node mapping (optional)
        """

        print(f"Experiment: parent_size={parent_size}, child_size={child_size}, temperature={temperature}, overlap={chunk_overlap}")

        # Start memory monitoring
        # NOTE: In grid search when vs!=None, initial memory will be after chunking+indexing
        #      In this case only RAG pipeline + inference memory is measured
        monitor = PeakMemoryMonitor()
        monitor.record()

        # Initial metrics
        start_time = time.time()

        # Chunking and indexing times
        chunking_time = precomputed_chunking_time if precomputed_chunking_time is not None else 0
        indexing_time = precomputed_indexing_time if precomputed_indexing_time is not None else 0
        chunk_stats = {}
        db_path = f"./milvus_p{parent_size}_c{child_size}_o{chunk_overlap}.db"

        # Use Vector Store if ready, otherwise create
        if vs is None:
            # Chunking
            print("1. Chunking in progress")
            chunking_start = time.time()
            chunker = HierarchicalChunker(
                parent_size=parent_size,
                child_size=child_size,
                chunk_overlap=chunk_overlap
            )
            nodes, node_mapping = chunker.chunk_text(self.book_text)
            chunking_time = time.time() - chunking_start

            chunk_stats = chunker.get_chunk_stats(nodes)
            monitor.record()

            # Vector Store & index
            print("2. Creating and indexing vector store")
            indexing_start = time.time()
            vs = VectorStore(db_path=db_path, wipe_existing=True)
            vs.create_index(nodes, node_mapping)
            indexing_time = time.time() - indexing_start
            monitor.record()
        else:
            print(f"1-2. Using ready index (chunking: {chunking_time:.2f}s, indexing: {indexing_time:.2f}s)")
            # Get chunk stats from node_mapping
            if node_mapping:
                chunker = HierarchicalChunker(parent_size, child_size, chunk_overlap)
                chunk_stats = chunker.get_chunk_stats(list(node_mapping.values()))
            monitor.record()

        # RAG Pipeline 
        print(f"3. Answering questions (temperature={temperature})...")
        rag = RAGPipeline(vector_store=vs, temperature=temperature, seed=self.seed)
        monitor.record()

        questions = self.test_df['question'].tolist()

        inference_start = time.time()
        question_times = []
        rag_results = []

        for i, question in enumerate(questions, 1):
            q_start = time.time()
            result = rag.answer_question(question, top_k=3, max_new_tokens=100)
            q_time = time.time() - q_start
            question_times.append(q_time)
            rag_results.append(result)
            monitor.record()
            print(f"  Question {i}/{len(questions)} - {q_time:.2f}s")

        inference_time = time.time() - inference_start
        avg_question_time = sum(question_times) / len(question_times)
        monitor.record()

        # Metrics
        print("4. Calculating metrics")
        predictions = [r['answer'] for r in rag_results]
        references = []
        for _, row in self.test_df.iterrows():
            refs = [row['answer1'], row['answer2']]
            references.append(refs)

        metrics = self.evaluator.evaluate(predictions, references)
        monitor.record()

        # Resource usage
        memory_snapshot = monitor.record()
        total_time = time.time() - start_time

        # Vector DB size
        db_size = os.path.getsize(db_path) / 1024 / 1024 if os.path.exists(db_path) else 0

        # Results
        results = {
            'config': {
                'parent_size': parent_size,
                'child_size': child_size,
                'temperature': temperature,
                'chunk_overlap': chunk_overlap
            },
            'chunk_stats': chunk_stats,
            'metrics': metrics,
            'performance': {
                'chunking_time': chunking_time,
                'indexing_time': indexing_time,
                'inference_time': inference_time,
                'avg_question_time': avg_question_time,
                'total_time': total_time,
                'db_size_mb': db_size,
                'initial_memory_mb': monitor.initial_mb,
                'peak_memory_mb': memory_snapshot.peak_mb,
                'memory_used_mb': memory_snapshot.delta_mb
            }
        }

        print(f"Total time: {total_time:.2f}s")

        # Clean up model and pipeline (prevent memory leak)
        del rag
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        return results

    def run_grid_search(self, parent_sizes: List[int], child_sizes: List[int],
                       temperatures: List[float], chunk_overlaps: List[int] = [20]) -> List[Dict]:
        """
        Run experiments for all parameter combinations

        NOTE: Temperature loop is innermost - so the same index is tested with different temperatures

        Memory Measurement Strategy:
        1. Chunking + indexing is done once for each (parent, child, overlap) combination
        2. Different temperatures are tested with this index
        3. For each temperature test:
           - New monitor starts (initial = chunking+index memory)
           - RAG pipeline + inference memory is measured
           - Model is cleaned after test (gc.collect)
        4. Index is cleaned after all temperature tests

        This way there is a clean start for each temperature and only
        RAG pipeline + inference overhead is measured.
        """
        all_results = []
        total_experiments = len(parent_sizes) * len(child_sizes) * len(temperatures) * len(chunk_overlaps)
        current_experiment = 0

        for parent_size in parent_sizes:
            for child_size in child_sizes:
                for chunk_overlap in chunk_overlaps:
                    # Do chunking + indexing once for this combination
                    print(f"CREATING INDEX: parent={parent_size}, child={child_size}, overlap={chunk_overlap}")

                    # Chunking - TIME MEASUREMENT
                    chunking_start = time.time()
                    chunker = HierarchicalChunker(
                        parent_size=parent_size,
                        child_size=child_size,
                        chunk_overlap=chunk_overlap
                    )
                    nodes, node_mapping = chunker.chunk_text(self.book_text)
                    chunking_time = time.time() - chunking_start
                    print(f"  Chunking completed: {chunking_time:.2f}s")

                    # Create Vector Store - TIME MEASUREMENT
                    indexing_start = time.time()
                    db_path = f"./milvus_p{parent_size}_c{child_size}_o{chunk_overlap}.db"
                    vs = VectorStore(db_path=db_path, wipe_existing=True)
                    vs.create_index(nodes, node_mapping)
                    indexing_time = time.time() - indexing_start
                    print(f"  Indexing completed: {indexing_time:.2f}s")

                    # Test different temperatures with this index
                    for temperature in temperatures:
                        current_experiment += 1
                        print(f"OVERALL PROGRESS: {current_experiment}/{total_experiments}")
                        print(f"Testing temperature={temperature}")

                        try:
                            result = self.run_single_experiment(
                                parent_size, child_size, temperature, chunk_overlap,
                                vs=vs, nodes=nodes, node_mapping=node_mapping,
                                precomputed_chunking_time=chunking_time,
                                precomputed_indexing_time=indexing_time
                            )
                            all_results.append(result)

                            # Save each experiment
                            self.save_single_result(result)

                        except Exception as e:
                            print(f"ERROR: {e}")
                            import traceback
                            traceback.print_exc()
                            continue

                    # Index usage complete, clean up
                    vs.close()
                    del vs
                    del nodes
                    del node_mapping
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    gc.collect()
                    print(f"Completed: parent={parent_size}, child={child_size}, overlap={chunk_overlap}")

        return all_results

    def save_single_result(self, result: Dict):
        """
        Save single experiment result
        """
        config = result['config']
        filename = f"exp_p{config['parent_size']}_c{config['child_size']}_t{config['temperature']}_o{config['chunk_overlap']}.json"
        filepath = self.results_dir / filename

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

    def save_summary(self, all_results: List[Dict], summary_filename: str = "experiment_summary"):
        """
        Save summary of all results as CSV and JSON

        Args:
            all_results: List of experiment results
            summary_filename: Summary filename (without extension)
        """
        summary_data = []
        for result in all_results:
            config = result['config']
            metrics = result['metrics']
            perf = result['performance']
            chunk_stats = result['chunk_stats']

            summary_data.append({
                'parent_size': config['parent_size'],
                'child_size': config['child_size'],
                'temperature': config['temperature'],
                'chunk_overlap': config['chunk_overlap'],
                'total_nodes': chunk_stats['total_nodes'],
                'parent_nodes': chunk_stats['total_parents'],
                'child_nodes': chunk_stats['total_children'],
                'bleu': metrics['bleu'],
                'rouge1': metrics['rouge1'],
                'rouge2': metrics['rouge2'],
                'rougeL': metrics['rougeL'],
                'chunking_time': perf['chunking_time'],
                'indexing_time': perf['indexing_time'],
                'inference_time': perf['inference_time'],
                'avg_question_time': perf['avg_question_time'],
                'total_time': perf['total_time'],
                'db_size_mb': perf['db_size_mb'],
                'initial_memory_mb': perf.get('initial_memory_mb'),
                'peak_memory_mb': perf.get('peak_memory_mb'),
                'memory_used_mb': perf['memory_used_mb']
            })

        df = pd.DataFrame(summary_data)

        csv_path = self.results_dir / f"{summary_filename}.csv"
        df.to_csv(csv_path, index=False)
        print(f"Summary CSV saved at: {csv_path}")

        json_path = self.results_dir / f"{summary_filename}.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        print(f"Summary JSON saved at: {json_path}")

        # Best results
        self.print_best_results(df)

    def print_best_results(self, df: pd.DataFrame):
        print("BEST RESULTS")

        # Highest BLEU
        best_bleu = df.loc[df['bleu'].idxmax()]
        print(f"Highest BLEU ({best_bleu['bleu']:.2f}):")
        print(f"  parent_size={best_bleu['parent_size']}, child_size={best_bleu['child_size']}, temp={best_bleu['temperature']}")

        # Highest ROUGE-L
        best_rougeL = df.loc[df['rougeL'].idxmax()]
        print(f"Highest ROUGE-L ({best_rougeL['rougeL']:.2f}):")
        print(f"  parent_size={best_rougeL['parent_size']}, child_size={best_rougeL['child_size']}, temp={best_rougeL['temperature']}")

        # Fastest
        fastest = df.loc[df['total_time'].idxmin()]
        print(f"Fastest ({fastest['total_time']:.2f}s):")
        print(f"  parent_size={fastest['parent_size']}, child_size={fastest['child_size']}, temp={fastest['temperature']}")

        # Least memory
        min_memory = df.loc[df['memory_used_mb'].idxmin()]
        print(f"Least Memory ({min_memory['memory_used_mb']:.2f} MB):")
        print(f"  parent_size={min_memory['parent_size']}, child_size={min_memory['child_size']}, temp={min_memory['temperature']}")


if __name__ == "__main__":

    # NOT: Temperature loop en içte - aynı index farklı temperature'lerle test edilir
    parent_sizes = [2048, 4096]      
    child_sizes = [512, 1024]        
    temperatures = [0.1, 0.3]       
    chunk_overlaps = [0, 100, 200]  


    # Experiment runner 
    runner = ExperimentRunner(
        book_path="data/children_of_new_forest.txt",
        test_questions_path="data/questions_test.csv",
        results_dir="results/experiments_v2"
    )

    # Grid search
    print(f"Total index creation: {len(parent_sizes) * len(child_sizes) * len(chunk_overlaps)} times")
    print(f"Total number of experiments: {len(parent_sizes) * len(child_sizes) * len(temperatures) * len(chunk_overlaps)} experiments")
    print("NOTE: Each index will be created once and tested with different temperatures")

    all_results = runner.run_grid_search(parent_sizes, child_sizes, temperatures, chunk_overlaps)

    # Save results
    runner.save_summary(all_results, summary_filename="experiment_summary_v2")

    print("experiments completed")
