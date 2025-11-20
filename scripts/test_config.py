
from src.config import settings
import os

def main():
    print("Testing Configuration Loading...")
    
    # Test LLM config
    llm_config = settings.llm_config
    print(f"LLM Config: {llm_config}")
    
    model = llm_config.get('model')
    api_base = llm_config.get('api_base')
    
    print(f"Model: {model}")
    print(f"API Base: {api_base}")
    
    # Test use_llm flag (might not be in settings.yaml yet)
    use_llm = settings.get('use_llm')
    print(f"Use LLM: {use_llm}")
    
    if model == "mistral" and api_base:
        print("SUCCESS: Configuration loaded correctly.")
    else:
        print("FAILURE: Configuration mismatch.")

if __name__ == "__main__":
    main()
