from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from typing import List, Dict, Optional
from vector_store_v2 import VectorStore
from performance_monitor import PeakMemoryMonitor
from seed_utils import set_global_seed

class RAGPipeline:
    """
    Retrieval-Augmented Generation pipeline
    Uses LlamaIndex AutoMergingRetriever + google/gemma-3-1b-it model
    """

    def __init__(self, vector_store: VectorStore, model_name="google/gemma-3-1b-it",
                 temperature=0.1, seed: Optional[int] = None):
        self.vector_store = vector_store
        self.model_name = model_name
        self.temperature = temperature
        self.seed = seed

        set_global_seed(seed)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Device: {self.device}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None,
            low_cpu_mem_usage=True
        )

        if self.device == "cpu":
            self.model = self.model.to(self.device)

        print("Model loaded")

    def retrieve_context(self, question: str, top_k: int = 3) -> str:
        """
        Retrieves most relevant nodes for the question and creates context
        AutoMergingRetriever automatically selects parent/child nodes
        """
        _, results = self.vector_store.hybrid_search(
            query=question,
            top_parents=top_k
        )
        context_parts = []
        for i, result in enumerate(results, 1):
            node_type = "PARENT" if result.get('is_parent', False) else "CHILD"
            context_parts.append(f"[{i}] ({node_type}) {result['text']}")

        context = "\n\n".join(context_parts)
        return context

    def generate_answer(self, question: str, context: str, max_new_tokens: int = 100) -> str:

        messages = [
            {"role": "user", "content": f"""Answer the question based on the given context from the following book:

            Title: The Children of the New Forest
            Author: Frederick Marryat
            Release date: May 21, 2007 [eBook #21558]
            Language: English
            Credits: Produced by Nick Hodson of London, England

            Context:
            {context}

            Question: {question}

            Answer:"""}
                    ]
        inputs = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=self.temperature,
                do_sample=True,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id
            )

        # Decode only new tokens
        answer = self.tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)

        return answer.strip()

    def answer_question(self, question: str, top_k: int = 3, max_new_tokens: int = 100) -> Dict:

        context = self.retrieve_context(question, top_k=top_k)

        answer = self.generate_answer(question, context, max_new_tokens=max_new_tokens)

        return {
            'question': question,
            'context': context,
            'answer': answer
        }

    def batch_answer_questions(self, questions: List[str], top_k: int = 3,
                               max_new_tokens: int = 100,
                               memory_monitor: Optional[PeakMemoryMonitor] = None) -> List[Dict]:
        """
        Answer multiple questions
        """
        results = []
        for i, question in enumerate(questions, 1):
            print(f"\nAnswering question {i}/{len(questions)}")
            result = self.answer_question(question, top_k, max_new_tokens)
            results.append(result)
            if memory_monitor:
                memory_monitor.record()

        if memory_monitor:
            memory_monitor.record()

        return results

if __name__ == "__main__":
    print("RAG pipeline test")

    # Create vector store (must be pre-indexed)
    vs = VectorStore(db_path="./test_milvus_llama.db")

    # Create dummy index for testing
    from chunker_v2 import HierarchicalChunker
    with open('data/children_of_new_forest.txt', 'r', encoding='utf-8') as f:
        text = f.read()

    chunker = HierarchicalChunker(parent_size=2048, child_size=512, chunk_overlap=20)
    nodes, node_mapping = chunker.chunk_text(text)
    vs.create_index(nodes, node_mapping)

    rag = RAGPipeline(vector_store=vs)

    test_question = "What is the title of this story?"
    result = rag.answer_question(test_question, top_k=3)

    print(f"\nQuestion: {result['question']}")
    print(f"\nContext:\n{result['context'][:300]}")
    print(f"\nAnswer: {result['answer']}")
