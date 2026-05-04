import ollama

class Translator:
    def __init__(self, model="gemma3:4b", target_language="Persian"):
        self.model = model
        self.target_language = target_language
        self.context_history = []
        print(f"Translator initialized with model '{model}' targeting '{target_language}'.")

    def translate(self, text):
        if not text:
            return ""
        
        context_str = ""
        if self.context_history:
            context_str = "Previous context sentences (for reference only, do not translate):\n"
            for prev in self.context_history:
                context_str += f"- {prev}\n"
            context_str += "\n"
            
        prompt = f"{context_str}Translate the following text to {self.target_language}. Use English for technical terms. Provide only the translation for the text below, no extra text:\n\n{text}"
        
        try:
            response = ollama.generate(model=self.model, prompt=prompt)
            translation = response['response'].strip()
            
            # Update context history with the current text
            self.context_history.append(text)
            if len(self.context_history) > 2:
                self.context_history.pop(0)
                
            return translation
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
            return [m.model for m in models_info.models]
        except Exception as e:
            print(f"Error fetching models: {e}")
            return ["gemma3:4b"] # Fallback
