
from src.config import settings
from src.extract_entities import LLMEntityExtractor
import sys

def main():
    with open("debug_output.txt", "w") as f:
        f.write("Starting Debug Script\n")
        
        # Check Settings
        use_llm = settings.get('use_llm', False)
        f.write(f"Settings use_llm: {use_llm}\n")
        f.write(f"LLM Config: {settings.llm_config}\n")
        
        if not use_llm:
            f.write("use_llm is False. Exiting.\n")
            return

        # Initialize Extractor
        try:
            extractor = LLMEntityExtractor()
            f.write(f"Extractor initialized. Model: {extractor.llm_model}, Base: {extractor.api_base}\n")
        except Exception as e:
            f.write(f"Error initializing extractor: {e}\n")
            return

        # Invoke LLM
        try:
            f.write("Invoking LLM...\n")
            response = extractor.llm.invoke("Say hello")
            f.write(f"LLM Response: {response}\n")
        except Exception as e:
            f.write(f"Error invoking LLM: {e}\n")

if __name__ == "__main__":
    main()
