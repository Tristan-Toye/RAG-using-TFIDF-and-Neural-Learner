from transformers import AutoModelForCausalLM, AutoTokenizer

from QueryRouter import QueryRouter


class RecipeRAG:
    def __init__(self,
                 intent_model_path="mistral-ft-intent",
                 base_model_name="mistralai/Mistral-7B-Instruct-v0.2",
                 use_zero_shot=True):
        """
        Initializes the RAG system with:
        - A fine-tuned Mistral classifier (for query intent)
        - A base Mistral model (for answer generation)
        """
        self.router = QueryRouter(
            mistral_path=intent_model_path,
            use_mistral=True,
            use_zero_shot=use_zero_shot
        )

        # Load tokenizer and base model for generation
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            device_map="auto",
            torch_dtype="auto"
        )

    def generate_response(self, query, docs, tags, ingredients, names, max_new_tokens=300):
        """
        Generates a response to a user query based on retrieved documents.

        Returns:
        - The model's answer (str)
        - The intent classification used (str)
        - The full prompt sent to the model (str)
        """
        prompt, intent = self.router.build_prompt(query, docs, tags, ingredients, names)

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            top_p=0.95,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id
        )

        decoded = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        answer = decoded.split("[/INST]")[-1].strip()
        return answer, intent, prompt
