
import yaml
from pathlib import Path
from typing import Dict, Any

class Settings:
    _instance = None
    _config: Dict[str, Any] = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Settings, cls).__new__(cls)
            cls._instance._load_config()
        return cls._instance

    def _load_config(self):
        """Load configuration from settings.yaml"""
        # Look for settings.yaml in the project root (parent of src)
        # Assuming this file is in src/config.py
        root_dir = Path(__file__).parent.parent
        settings_path = root_dir / "settings.yaml"
        
        if settings_path.exists():
            try:
                with open(settings_path, 'r') as f:
                    self._config = yaml.safe_load(f) or {}
            except Exception as e:
                print(f"Error loading settings.yaml: {e}")
                self._config = {}
        else:
            print(f"Warning: settings.yaml not found at {settings_path}")
            self._config = {}

    @property
    def llm_config(self) -> Dict[str, Any]:
        return self._config.get('llm', {})

    @property
    def embeddings_config(self) -> Dict[str, Any]:
        return self._config.get('embeddings', {})
        
    @property
    def graphrag_config(self) -> Dict[str, Any]:
        return self._config.get('graphrag', {})

    def get(self, key: str, default: Any = None) -> Any:
        return self._config.get(key, default)

# Global settings instance
settings = Settings()
