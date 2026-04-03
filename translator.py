import ollama

class Translator:
    def __init__(self, model="gemma3:1b", target_language="Persian"):
        self.model = model
        self.target_language = target_language
        print(f"Translator initialized with model '{model}' targeting '{target_language}'.")

    def translate(self, text):
        if not text:
            return ""
        
        prompt = f"Translate the following text to {self.target_language}. Provide only the translation, no extra text:\n\n{text}"
        
        try:
            response = ollama.chat(model=self.model, messages=[
                {'role': 'user', 'content': prompt},
            ])
            return response['message']['content'].strip()
        except Exception as e:
            print(f"Error in translation with model {self.model}: {e}")
            return f"[Error: {e}]"
            
    def set_target_language(self, language):
        self.target_language = language

    def update_model(self, model_name):
        self.model = model_name
        print(f"Model updated to {model_name}")

    @staticmethod
    def get_local_models():
        try:
            models_info = ollama.list()
            print(f"models_info: {models_info}")
            return [m['name'] for m in models_info['models']]
        except Exception as e:
            print(f"Error fetching models: {e}")
            return ["gemma3:1b"] # Fallback
