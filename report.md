# V-RAG-Final: Hierarchical RAG System for Literary Question-Answering

## 1. Introduction

This project implements a hierarchical Retrieval-Augmented Generation (RAG) system designed to answer questions about literary texts using advanced chunking strategies and vector-based semantic search. The primary goal is to evaluate whether hierarchical chunking with parent-child relationships improves answer quality compared to a baseline language model without retrieval augmentation.

**Book Selection:** "The Children of the New Forest" by Frederick Marryat (Project Gutenberg #21558) was selected for this study. This novel, spanning 631926 characters and 27 chapters, provides a rich narrative structure with complex character relationships and plot developments. The availability of test questions from the NarrativeQA dataset makes it ideal for evaluating question-answering systems.

**Scope:** The project scope includes implementing a complete RAG pipeline with hierarchical chunking, conducting hyperparameter optimization through grid search, and comparing performance against a baseline model across multiple evaluation metrics (BLEU and ROUGE-L). The system runs on Google Colab with GPU acceleration.

## 2. Approach & Methodology

### 2.1 Preprocessing

The text preprocessing pipeline consists of the following steps:
- Extraction of book text from Project Gutenberg format with automatic removal of headers and footers
- Chapter segmentation using regex pattern matching (CHAPTER ONE, CHAPTER TWO, etc.)
- Preservation of chapter metadata (chapter number and title) for contextual enrichment
- Text normalization while maintaining narrative structure

### 2.2 Embedding Model

**Model:** BAAI/bge-large-en-v1.5

**Justification:**
- Optimized specifically for English text retrieval
- 1024-dimensional dense embeddings providing rich semantic representations
- Suitable for literary text with strong performance on long-form narrative content
- Efficient batch processing (batch size: 32) for fast indexing
- Open-source and well-supported through HuggingFace Transformers

### 2.3 Hierarchical Chunking Parameters

**Selected Configuration:**
- Parent chunk size: 2048 characters
- Child chunk size: 512 characters
- Chunk overlap: 200 characters

**Justification:**
- Parent chunks (2048 chars) provide broad contextual understanding spanning multiple paragraphs
- Child chunks (512 chars) enable precise matching for specific details and facts
- 200-character overlap prevents information loss at chunk boundaries and improves context continuity
- Hierarchical structure allows AutoMergingRetriever to dynamically expand context when multiple related child chunks are retrieved

### 2.4 Vector Database

**Database:** Milvus Lite

**Justification:**
- Lightweight file-based storage suitable for single-book applications
- Supports efficient similarity search (with P, COSINE and L2) for semantic retrieval
- Integrated seamlessly with LlamaIndex for document and vector store management
- No external server requirements, making it ideal for Colab environments
- Compact database size with fast query response times 

### 2.5 Retrieval Strategy

**Strategy:** AutoMergingRetriever with hierarchical node relationships

**Configuration:**
- Top-k retrieval: 3 parent nodes
- Base retrieval: 10 child nodes initially retrieved
- Merging logic: When multiple child chunks from the same parent are retrieved, they are automatically merged into the parent chunk

**Justification:**
- Balances precision (child chunks for specific details) with context (parent chunks for broader understanding)
- Reduces redundancy by merging related child chunks
- Provides flexible context granularity based on retrieval patterns
- Maintains parent-child relationships through LlamaIndex document store

### 2.6 Prompt Design 

The systems use structured prompt templates for answer generation:


### for RAG

```
Answer the question based on the given context from the following book:

Title: The Children of the New Forest
Author: Frederick Marryat
Release date: May 21, 2007 [eBook #21558]
Language: English
Credits: Produced by Nick Hodson of London, England

Context:
{context}

Question: {question}

Answer:
```

### for Baseline

```
Answer the question about the following book:

Title: The Children of the New Forest
Author: Frederick Marryat
Release date: May 21, 2007 [eBook #21558]
Language: English
Credits: Produced by Nick Hodson of London, England

Question: {question}

Answer:
```

**Design Choices:**
- Clear separation between context and question
- Explicit instruction to use provided context
- Minimal prompt engineering to evaluate retrieval effectiveness
- Temperature variations [0.1, 0.3, 0.5] tested to optimize response determinism

## 3. Implementation Details

### 3.1 Key Libraries

- **LlamaIndex:** Framework for hierarchical node parsing, indexing, and retrieval
- **Milvus Lite:** Vector database for embedding storage and similarity search
- **HuggingFace Transformers:** Embedding model and language model loading
- **PyTorch:** Backend for model inference with GPU acceleration
- **Evaluate (HuggingFace):** BLEU and ROUGE metric computation
- **Pandas:** Data manipulation and results analysis

### 3.2 Challenges and Solutions

**Challenge 1: Vector Database Setup**
- **Issue:** Initial complexity in configuring Milvus with LlamaIndex storage context
- **Solution:** Used Milvus Lite file-based storage and integrated LlamaIndex's document store for hierarchical node relationships. Persistent storage to disk ensured index preservation across sessions.

**Challenge 2: Resource Constraints**
- **Issue:** Memory limitations when loading both embedding model and LLM simultaneously in Grid Search phase with two parallel workes.
- **Solution:** Leveraged Google Colab's GPU (A100 80GB) and optimized batch sizes. Used mixed precision inference where possible.

**Challenge 3: Chunking Complexity**
- **Issue:** Balancing chunk size for semantic coherence while maintaining retrieval precision
- **Solution:** Implemented grid search across multiple chunking, overlap and temperature configurations. Hierarchical approach with AutoMergingRetriever provided adaptive context granularity.

**Challenge 4: Evaluation Metric Interpretation**
- **Issue:** Low absolute BLEU scores across all configurations
- **Solution:** Focused on relative improvements and ROUGE scores, which better capture semantic similarity for open-ended question-answering tasks.

## 4. Results & Discussion

### 4.1 Evaluation Results

#### 4.1.1 Baseline vs RAG (Initial Configuration)

| Approach | BLEU-4 (vs. Ground Truth) | ROUGE-L (vs. Ground Truth) | Notes / Qualitative Resource Impact |
|----------|----------------------------|----------------------------|--------------------------------------|
| **Baseline**<br>(No RAG,<br>temperature=0.5) | **1.54** | **6.45** | Fast inference (1.11 sec per Q, total inference 43.2 sec)<br>Peak memory usage: 5.21 GB<br>No indexing time, no vector DB disk usage |
| **RAG**<br>(parent_size=2048,<br>child_size=512,<br>chunk_overlap=100,<br>temperature=0.5) | **1.96** | **13.56** | Slower inference (1.56 sec per Q, total inference 60.9 sec)<br>Peak memory usage: 7.18 GB<br>**+27% BLEU improvement**<br>**+110% ROUGE-L improvement** |

**Key Observations:**
- RAG achieves **110% improvement in ROUGE-L**, indicating significantly better semantic alignment with ground truth
- RAG shows **27% improvement in BLEU**, demonstrating more precise token-level matching
- Trade-off: 40% slower inference (0.45 sec/Q overhead) due to retrieval + generation
- Trade-off: ~2 GB additional memory for embedding model + vector store

---

#### 4.1.2 Grid Search: Hyperparameter Optimization (3 Experiments)

| Approach | BLEU-4 (vs. Ground Truth) | ROUGE-L (vs. Ground Truth) | Notes / Qualitative Resource Impact |
|----------|----------------------------|----------------------------|--------------------------------------|
| **exp_5**<br>(parent=2048,<br>child=512,<br>overlap=200,<br>temp=0.10) | **1.90** | **15.64** | Total nodes: 652 (93 parents, 559 children)<br>Higher overlap → better context preservation<br>Lower temperature → more deterministic outputs<br>**Best ROUGE-L** among experiments |
| **exp_6**<br>(parent=2048,<br>child=512,<br>overlap=200,<br>temp=0.30) | **1.92** | **15.63** | Total nodes: 652 (93 parents, 559 children)<br>Same chunking as exp_5, higher temperature<br>Slightly higher BLEU, nearly identical ROUGE-L<br>More diverse outputs |
| **exp_7**<br>(parent=2048,<br>child=1024,<br>overlap=0,<br>temp=0.10) | **2.17** | **14.44** | Total nodes: 298 (84 parents, 214 children)<br>Larger child chunks → fewer, longer contexts<br>**Best BLEU** among experiments<br>No overlap → potential boundary issues |

**Key Observations:**
- **exp_7** achieves highest BLEU (2.17) with larger child chunks (1024 tokens), suggesting longer contexts improve token-level precision
- **exp_5** achieves highest ROUGE-L (15.64) with smaller child chunks (512) + overlap (200), indicating better semantic coverage
- Temperature impact is minimal: exp_5 (0.10) vs exp_6 (0.30) show nearly identical metrics


---

### 4.1.3 Configuration Analysis & Trade-offs

#### **Analysis 1: Baseline vs RAG (Why RAG Wins)**

**Metrics Comparison:**
- Baseline: BLEU 1.54, ROUGE-L 6.45
- RAG: BLEU 1.96 (+27%), ROUGE-L 13.56 (+110%)

**Why RAG Outperforms:**
1. **Grounded Context:** RAG retrieves relevant book passages before generation, reducing hallucination and improving factual accuracy
2. **Semantic Alignment:** Retrieved contexts provide narrative continuity, significantly boosting ROUGE-L (semantic overlap)
3. **Specificity:** AutoMergingRetriever adapts context granularity, balancing detailed child chunks (512 tokens) with broader parent chunks (2048 tokens)

**Trade-offs:**
- **Inference Time:** +40% slower (1.56 vs 1.11 sec/Q) due to retrieval overhead
- **Memory:** +38% peak memory (7.18 vs 5.21 GB) for embedding model + vector store
- **Setup:** Requires indexing (chunking + vector DB creation)

**Best for:** Tasks prioritizing answer quality and semantic coherence over raw speed

---

#### **Analysis 2: Grid Search Results - Chunk Size Impact**

**exp_5 vs exp_7:**
- exp_5 (child=512, overlap=200): BLEU 1.90, **ROUGE-L 15.64** 
- exp_7 (child=1024, overlap=0): **BLEU 2.17** , ROUGE-L 14.44

**Key Insights:**
1. **Smaller child chunks (512) + overlap → Better ROUGE-L**
   - More granular retrieval captures diverse semantic aspects
   - Overlap (200 chars) prevents boundary context loss
   - 8.3% ROUGE-L improvement over exp_7

2. **Larger child chunks (1024) + no overlap → Better BLEU**
   - Longer contexts provide more n-gram matches
   - 14.2% BLEU improvement over exp_5
   - Fewer nodes (298 vs 652) = faster retrieval

**Recommendation:**
- **For semantic coherence (narratives, summaries):** Use exp_5 config (child=512, overlap=200)
- **For factual precision (specific details):** Use exp_7 config (child=1024, overlap=0)

---

#### **Analysis 3: Temperature Impact (exp_5 vs exp_6)**

**Metrics:**
- exp_5 (temp=0.10): BLEU 1.90, ROUGE-L 15.64
- exp_6 (temp=0.30): BLEU 1.92, ROUGE-L 15.63

**Observation:** Temperature has **minimal impact** (<1% difference) when chunking configuration is identical.

**Implication:** Chunking strategy (chunk size, overlap) is far more critical than generation temperature for RAG systems. Temperature tuning should be a secondary optimization after optimal retrieval configuration is established. 
Higher temperature values ​​can be tried. In previous experiments, I observed that the BLUE score decreases at higher temperatures.
---

#### **Final Recommendation**

**Best Overall Configuration:**
- **Parent size:** 2048
- **Child size:** 512
- **Chunk overlap:** 200
- **Temperature:** 0.10 (or 0.30, negligible difference)
- **Rationale:** Achieves highest ROUGE-L (15.64), indicating superior semantic alignment for literary QA tasks where answer coherence matters more than exact n-gram matches

### 4.2 Performance Analysis

**Quantitative Improvements:**

The hierarchical RAG system demonstrates substantial improvements over the baseline LLM-only approach:

| Metric | Baseline | RAG (Initial) | Best Grid (exp_5) | Diff/Improvement (Initial) | Diff/Improvement (Best) |
|--------|----------|---------------|-------------------|-----------------------|--------------------|
| BLEU-4 | 1.54 | 1.96 | 1.90 | **+27%** | **+23%** |
| ROUGE-L | 6.45 | 13.56 | 15.64 | **+110%** | **+142%** |
| Inference Time/Q | 1.11s | 1.56s | ~1.54s | +40% | +39% |
| Peak Memory | 5.21 GB | 7.18 GB | N/A | +38% | N/A |

**Key Observations:**

1. **Dramatic ROUGE-L Improvement:** RAG achieves 110-142% improvement in ROUGE-L, demonstrating that retrieval-augmented generation significantly enhances semantic alignment with reference answers. This is crucial for literary QA where answer coherence matters more than exact wording.

2. **Modest BLEU Improvement:** BLEU improvements are more modest (23-27%) but still meaningful. The relatively low absolute BLEU scores (1.54-2.17) across all approaches reflect the open-ended nature of narrative question-answering, where exact n-gram matches are inherently rare.

3. **Acceptable Latency Trade-off:** Inference time increases by ~0.45 seconds per question with RAG due to retrieval + merging overhead. For a quality gain of 110-142% in ROUGE-L, this trade-off is highly acceptable for non-real-time applications.

4. **Memory Overhead:** Peak memory increases by 38% (from 5.21 GB to 7.18 GB) to accommodate the embedding model and vector store. This is within acceptable limits for GPU-accelerated environments like Google Colab T4 or A100.

5. **Grid Search Refinement:** Hyperparameter optimization (exp_5) further improves ROUGE-L from 13.56 to 15.64 (+15% over initial RAG), demonstrating the value of systematic chunking configuration tuning.

### 4.3 Hierarchical RAG Effectiveness

The hierarchical chunking strategy with AutoMergingRetriever proves highly effective for literary question-answering:

1. **Adaptive Context Granularity:** AutoMergingRetriever dynamically adjusts context based on retrieval patterns:
   - Retrieves child chunks (512 or 1024 tokens) for precision and specificity
   - Expands to parent chunks (2048 tokens) when broader narrative context is needed
   - This two-tier approach balances detailed information retrieval with contextual understanding, outperforming flat chunking strategies

2. **Overlap Benefits:** Chunk overlap (200 tokens) prevents information loss at boundaries:
   - exp_5 (overlap=200): ROUGE-L 15.64
   - exp_7 (overlap=0): ROUGE-L 14.44
   - **8.3% improvement** demonstrates overlap's importance for narrative text where context flows across paragraphs
   - Trade-off: 2.2x more nodes (652 vs 298), but retrieval remains fast

3. **Chunk Size Trade-offs:**
   - **Smaller chunks (512):** Better semantic coverage → higher ROUGE-L (15.64 vs 14.44)
   - **Larger chunks (1024):** More n-gram matches → higher BLEU (2.17 vs 1.90)
   - For literary QA, semantic coherence (ROUGE-L) is more valuable than exact phrasing (BLEU)

4. **Temperature Impact (Minimal):** Temperature (0.10 vs 0.30) shows <1% metric difference when chunking is constant (exp_5 vs exp_6), indicating **retrieval configuration is far more critical than generation parameters**

5. **Resource Efficiency:** Despite retrieval overhead, the system remains practical:
   - Vector DB storage: Minimal (3-6 MB, <10 MB typically)
   - Peak memory: 7.18 GB (within A100 limits, works on Colab)
   - Inference: ~1.55 sec/question (acceptable for non-real-time use)
   - The quality gains (110-142% ROUGE-L improvement) justify the resource trade-offs


### 5 Future Improvements

**Chunking Strategies:**
- Sliding window approaches with larger overlaps can be implemented for dense context coverage.
- Chunking-level search can be tested as an additional hierarchical layer.

**Embedding Models:**
- Evaluate other domain-specific models fine-tuned on literary texts
- Test smaller models (e.g., all-MiniLM-L6-v2) for faster inference with acceptable quality trade-offs

**Re-ranking:**
- Implement cross-encoder re-ranking to improve retrieval precision
- Metadata filtering (e.g., chapter-based constraints) can be added when question context is known

**Generation Optimization:**
- Experiment with larger models (e.g., Gemma 7B) when resources permit

**Evaluation:**
- Perhaps the AnswerRelevancyMetric could also be used as an evaluation metric in addition to BLUE or ROUGE.

---

**Repository:** [https://github.com/sendayildirim/V-RAG-Final](https://github.com/sendayildirim/V-RAG-Final)
