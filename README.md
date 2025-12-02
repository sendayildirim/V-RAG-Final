# V-RAG-Final: Hierarchical RAG System for Literary Question-Answering

A Retrieval-Augmented Generation (RAG) system using hierarchical chunking and vector search for question-answering on "The Children of the New Forest" by Frederick Marryat.

## Overview

V-RAG-Final implements a hierarchical RAG pipeline that combines:
- Hierarchical text chunking with parent-child relationships using LlamaIndex
- Dense vector embeddings for semantic search (BAAI/bge-large-en-v1.5)
- AutoMergingRetriever strategy for adaptive context granularity
- Instruction-tuned language model for answer generation (google/gemma-3-1b-it)

### Objectives

This project aims to:
- Implement a hierarchical RAG system using LlamaIndex and Milvus Lite for literary question-answering
- Compare hierarchical chunking against a baseline model without retrieval augmentation
- Evaluate the impact of different chunking parameters (parent size, child size, overlap) on answer quality
- Optimize generation parameters (temperature) for better responses
- Measure performance trade-offs between answer quality, inference time, and resource usage
- Demonstrate that contextual retrieval improves answer quality over pure language model generation

### Book Selection

**Book:** "The Children of the New Forest" by Frederick Marryat (Project Gutenberg #21558)

**Rationale:**
- Narrative structure with clear chapter divisions suitable for hierarchical chunking
- Rich character relationships and plot details requiring contextual understanding
- Available test questions from the NarrativeQA dataset
- Sufficient length (631,926 characters, 27 chapters) to test retrieval effectiveness
- Literary text representative of RAG applications in educational and research contexts

## System Architecture

### 1. Text Chunking (src/chunker_v2.py)

**Chunking Strategy:** Hierarchical chunking using LlamaIndex's HierarchicalNodeParser

**Default Parameters:**
- Parent chunk size: 2048 characters
- Child chunk size: 512 characters
- Chunk overlap: 20 characters (configurable: 0, 100, 200 in experiments)

**How it works:**
1. Extracts and cleans Project Gutenberg text (removes headers/footers)
2. Splits book into chapters using regex patterns (CHAPTER ONE, CHAPTER TWO, etc.)
3. Creates hierarchical nodes for each chapter:
   - Parent nodes: Larger context windows (2048 chars)
   - Child nodes: Smaller, focused chunks (512 chars)
   - Each child node maintains a reference to its parent node

**Metadata:** Each chunk includes chapter number, chapter title, and book information

### 2. Embedding Model

**Model:** BAAI/bge-large-en-v1.5

**Specifications:**
- Type: Dense bi-encoder from BAAI (Beijing Academy of Artificial Intelligence)
- Embedding dimension: 1024
- Batch size: 32
- Framework: HuggingFace Transformers via LlamaIndex

**Why BGE-large-en-v1.5:**
- State-of-the-art performance on MTEB benchmark
- Optimized for English semantic search
- Strong performance on retrieval tasks
- Suitable for academic/literary text

### 3. Vector Database (src/vector_store_v2.py)

**Database:** Milvus Lite

**Configuration:**
- Storage: File-based (.db file)
- Collection name: book_chunks
- Vector dimension: 1024 (matching BGE embedding size)
- Index type: Default Milvus auto-indexing
- Distance metric: Cosine similarity

**Storage Components:**
1. Vector Store: Stores embeddings and enables similarity search
2. Document Store: Maintains hierarchical node relationships (parent-child)

**Retrieval Strategy:**
- Base retrieval: Standard vector similarity search
- AutoMergingRetriever: Intelligently merges child chunks into parent chunks when multiple children from the same parent are retrieved
- Top-k: Default 10 for base retrieval, 3 for final results (configurable)

### 4. RAG Pipeline (src/rag_pipeline_v2.py)

**Language Model:** google/gemma-3-1b-it (Gemma 3, 1B parameters, instruction-tuned)

**Generation Parameters:**
- Temperature: 0.1 (default, experiments test 0.1, 0.3, 0.5)
- Max new tokens: 100
- Sampling: Nucleus sampling (top_p=0.9)
- Device: Auto-detection (CUDA if available, else CPU)

**Pipeline Flow:**
1. Query → Retrieve relevant chunks via AutoMergingRetriever
2. Retrieved chunks → Format as context with node type labels (PARENT/CHILD)
3. Context + Question → Language model
4. Model → Generate answer based on book context

### 5. Baseline Model (src/baseline_model_v2.py)

**Purpose:** Comparison benchmark without RAG

**Configuration:**
- Same LLM: google/gemma-3-1b-it
- No retrieval: Direct question answering without context
- Same generation parameters for fair comparison

### 6. Performance Monitoring (src/performance_monitor.py)

**Metrics Tracked:**
- CPU memory usage (RSS via psutil)
- GPU memory usage (torch.cuda.memory_allocated)
- Combined total memory (CPU + GPU)
- Peak memory tracking across experiment lifecycle
- Initial, current, and delta memory measurements

### 7. Reproducibility (src/seed_utils.py)

**Seed Management:**
- Global seed setting for reproducible experiments
- Controls: Python random, NumPy, PyTorch (CPU and CUDA)
- Ensures deterministic behavior across runs

## Experiment Configuration (src/experiment_runner_v2.py)

The system supports grid search over multiple hyperparameters:

**Tested Chunking Parameters:**
- Parent sizes: [2048, 4096] characters
- Child sizes: [512, 1024] characters
- Chunk overlaps: [0, 100, 200] characters

**Tested Generation Parameters:**
- Temperatures: [0.1, 0.3, 0.5]

**Evaluation Metrics (src/metrics.py):**
- BLEU-4 score (sacrebleu)
- ROUGE-1, ROUGE-2, ROUGE-L scores (rouge-score)
- Inference time per question
- Total processing time (chunking, indexing, inference)
- Memory usage (CPU + GPU combined)
- Database size

## Installation

### Requirements

Install all dependencies via requirements.txt:

```bash
pip install -r requirements.txt
```

The requirements.txt includes:
- llama-index-core, llama-index-vector-stores-milvus, llama-index-embeddings-huggingface
- pymilvus, milvus-lite
- sentence-transformers, transformers
- rouge-score, sacrebleu
- accelerate, bitsandbytes (for GPU optimization)
- pandas, requests, psutil
- matplotlib, seaborn (for visualization)

### Data Setup

The notebook automatically downloads data. If running Python scripts directly, place files at:

```
data/children_of_new_forest.txt
data/questions_test.csv
```

CSV format should include columns: question, answer1, answer2

## Running the System

### Google Colab (Recommended)

The easiest way to run this system is through Google Colab using the provided notebook:

1. Open the notebook in Google Colab:
   - Navigate to [notebooks/main_rag_notebook_v2_all.ipynb](notebooks/main_rag_notebook_v2_all.ipynb)
   - Upload to Google Drive or use "Open in Colab"

2. Execute the notebook cells sequentially:
   - The notebook will automatically install dependencies
   - Download the book and test questions from the dataset
   - Create vector indices and run experiments
   - Generate evaluation metrics and visualizations

3. The notebook includes:
   - Data preparation and hierarchical chunking
   - Vector store creation with Milvus Lite
   - Baseline model comparison (with memory cleanup)
   - RAG pipeline with AutoMergingRetriever
   - Hyperparameter grid search experiments
   - Performance evaluation and visualization

**Note:** For Grid Search experiments, an A100 GPU is recommended due to parallel processing (2 workers). Google Colab's free T4 is sufficient for all other steps.

### Local Environment

For local execution, install dependencies:

```bash
pip install -r requirements.txt
```

Then run the notebook locally or use the Python modules directly from the src/ directory.

## Project Structure

### Initial Structure (t=0)

When you first clone this repository:

```
V-RAG-Final/
├── notebooks/
│   └── main_rag_notebook_v2_all.ipynb
├── src/
│   ├── baseline_model_v2.py       # Baseline LLM without RAG
│   ├── chunker_v2.py               # Hierarchical text chunking
│   ├── data_loader.py              # Data loading utilities
│   ├── experiment_runner_v2.py     # Grid search experimentation
│   ├── metrics.py                  # BLEU and ROUGE evaluation
│   ├── performance_monitor.py      # CPU + GPU memory tracking
│   ├── rag_pipeline_v2.py          # RAG pipeline with Gemma
│   ├── seed_utils.py               # Reproducibility utilities
│   └── vector_store_v2.py          # Milvus vector database interface
├── requirements.txt
├── README.md
└── report.md
```

### Structure After Running Notebook

After executing the notebook, additional directories and files are created:

```
V-RAG-Final/
├── notebooks/
│   └── main_rag_notebook_v2_all.ipynb
├── src/
│   ├── baseline_model_v2.py
│   ├── chunker_v2.py
│   ├── data_loader.py
│   ├── experiment_runner_v2.py
│   ├── metrics.py
│   ├── performance_monitor.py
│   ├── rag_pipeline_v2.py
│   ├── seed_utils.py
│   └── vector_store_v2.py
├── data/                           # Created by notebook
│   ├── children_of_new_forest.txt  # Downloaded book text
│   └── questions_test.csv          # Test questions (39 questions)
├── results/                        # Created by notebook
│   ├── baseline_QA.csv             # Baseline model answers
│   ├── RAG_QA.csv                  # RAG model answers
│   ├── rag_vs_baseline.json        # Comparison metrics
│   ├── performance_metrics.png     # Visualizations
│   └── experiments_v2/             # Grid search results
│       ├── exp_*.json              # Individual experiment results
│       ├── experiment_summary_v2.csv
│       └── experiment_summary_v2.json
├── milvus_*.db                     # Milvus vector database files
├── docstore.json                   # Document store for node relationships
├── requirements.txt
├── README.md
└── report.md
```

## Technical Details

### Hierarchical Chunking Rationale

Hierarchical chunking provides multiple context granularities:
- Child chunks (512 chars): Precise matching for specific details
- Parent chunks (2048 chars): Broader context for understanding
- AutoMerging: Dynamically expands to parent when multiple related children are retrieved

This approach balances precision and context coverage, outperforming flat chunking strategies.

### Chunk Overlap

Overlap prevents information loss at chunk boundaries:
- 0 characters: No overlap, maximum efficiency
- 20 characters: Default, minimal overlap
- 100-200 characters: Higher overlap, better context continuity for narrative text

Experiments show 200-character overlap improves ROUGE-L by 8.3% compared to no overlap.

### Temperature Settings

- 0.1: Low randomness, focused and deterministic answers
- 0.3: Moderate randomness, more diverse responses
- 0.5: Higher randomness, creative and varied answers

Results show temperature has minimal impact (<1% difference) when chunking configuration is optimal.

## Evaluation

The system is evaluated using:
- BLEU-4: Measures n-gram overlap with reference answers
- ROUGE-1/2/L: Measures recall of unigrams, bigrams, and longest common subsequence
- Latency metrics: Chunking, indexing, and inference times
- Resource metrics: Memory usage (CPU + GPU combined) and database size

Results are saved in results/experiments_v2/ with:
- Individual experiment JSON files
- Summary CSV with all metrics
- Summary JSON with complete results

## Results Summary

### Best Configuration

After extensive hyperparameter grid search across 24 configurations, the optimal RAG setup was identified:

**Optimal Parameters:**
- Parent chunk size: 2048 characters
- Child chunk size: 512 characters
- Chunk overlap: 200 characters
- Temperature: 0.1 (or 0.3, negligible difference)
- Top-k retrieval: 3 parent nodes


For detailed analysis, visualizations, and complete experimental results, refer to:
- Notebook: [notebooks/main_rag_notebook_v2_all.ipynb](notebooks/main_rag_notebook_v2_all.ipynb)
- Report: [report.md](report.md)

## Citation

### Book
- Title: The Children of the New Forest
- Author: Frederick Marryat
- Source: Project Gutenberg
- eBook: #21558
- Release Date: May 21, 2007

### Models
- Embedding Model: BAAI/bge-large-en-v1.5
- Language Model: google/gemma-3-1b-it

## Report

For a comprehensive analysis of the system design, experimental methodology, results, and conclusions, please refer to the final project report:

[Link to Final Report](./report.md)

The report includes:
- Detailed explanation of the RAG architecture and design choices
- Complete experimental setup and methodology
- In-depth analysis of results across all hyperparameter configurations
- Comparison with baseline approaches
- Discussion of trade-offs and future work
- Visualizations and statistical analysis

## License

This project is for educational and research purposes.
