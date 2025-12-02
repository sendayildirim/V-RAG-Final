from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from typing import List, Dict, Optional

from performance_monitor import PeakMemoryMonitor
from seed_utils import set_global_seed

class BaselineModel:
    """
    Baseline model that generates answers using only LLM without RAG
    Uses google/gemma-3-1b-it model
    """

    def __init__(self, model_name="google/gemma-3-1b-it", seed: Optional[int] = None):
        self.model_name = model_name
        self.seed = seed

        set_global_seed(seed)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Device: {self.device}")

        print(f"Loading baseline model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None,
            low_cpu_mem_usage=True
        )

        if self.device == "cpu":
            self.model = self.model.to(self.device)

        print("Baseline model loaded")

    def create_prompt(self, question: str) -> str:  # this is only for baseline

        prompt = f"""You are a helpful assistant. Answer the question about the following book:

                Title: The Children of the New Forest
                Author: Frederick Marryat
                Release date: May 21, 2007 [eBook #21558]
                Language: English
                Credits: Produced by Nick Hodson of London, England

                Question: {question}

                Answer:"""
        return prompt

    def generate_answer(self, question: str, max_new_tokens: int = 100) -> str:  # generate answer without context

        messages = [
            {"role": "user", "content": f"""Answer the question about the following book:

            Title: The Children of the New Forest
            Author: Frederick Marryat
            Release date: May 21, 2007 [eBook #21558]
            Language: English
            Credits: Produced by Nick Hodson of London, England

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
                temperature=0.5,
                do_sample=True,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id
            )

        # Decode only new tokens
        answer = self.tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)

        return answer.strip()

    def answer_question(self, question: str, max_new_tokens: int = 100) -> Dict:  # answer single question

        answer = self.generate_answer(question, max_new_tokens=max_new_tokens)

        return {
            'question': question,
            'answer': answer
        }

    def batch_answer_questions(
        self,
        questions: List[str],
        max_new_tokens: int = 100,
        memory_monitor: Optional[PeakMemoryMonitor] = None
    ) -> List[Dict]:  # answer multiple questions

        results = []
        for i, question in enumerate(questions, 1):
            print(f"\nAnswering question {i}/{len(questions)}")
            result = self.answer_question(question, max_new_tokens)
            results.append(result)
            if memory_monitor:
                memory_monitor.record()

        if memory_monitor:
            memory_monitor.record()

        return results

if __name__ == "__main__":
    # Test
    print("Baseline Model test")

    baseline = BaselineModel()

    test_question = "What is the title of this story?"
    result = baseline.answer_question(test_question)

    print(f"\nQuestion: {result['question']}")
    print(f"\nAnswer: {result['answer']}")
