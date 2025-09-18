"""
Model Provider Abstraction for SpatialAnalysisAgent
Supports multiple AI model providers including OpenAI, local models, and open-source alternatives
"""

import os
import sys
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
import configparser

# Add current directory to path
current_script_dir = os.path.dirname(os.path.abspath(__file__))
if current_script_dir not in sys.path:
    sys.path.append(current_script_dir)


class ModelProvider(ABC):
    """Abstract base class for AI model providers"""
    
    @abstractmethod
    def create_client(self, config: Dict[str, Any]):
        """Create and return a client for the model provider"""
        pass
    
    @abstractmethod
    def generate_completion(self, client, model: str, messages: List[Dict], **kwargs):
        """Generate completion using the provider's API"""
        pass
    
    @abstractmethod
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """Validate configuration for this provider"""
        pass


class OpenAIProvider(ModelProvider):
    """OpenAI API provider for GPT models"""
    
    def create_client(self, config: Dict[str, Any]):
        from openai import OpenAI
        return OpenAI(api_key=config.get('api_key'))
    
    def generate_completion(self, client, model: str, messages: List[Dict], **kwargs):
        return client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=kwargs.get('temperature', 0),
            stream=kwargs.get('stream', False)
        )
    
    def validate_config(self, config: Dict[str, Any]) -> bool:
        return 'api_key' in config and config['api_key'].strip() != ''


class OllamaProvider(ModelProvider):
    """Ollama local provider for gpt-oss-20b and other models"""
    
    def create_client(self, config: Dict[str, Any]):
        # Use OpenAI SDK with Ollama's compatible endpoint (exact match to LLM_SERVER_TESTING.py)
        from openai import OpenAI
        base_url = config.get('base_url', 'http://128.118.54.16:11434/v1')
        api_key = config.get('api_key', 'no-api')
        
        # Debug logging
        # print(f"[DEBUG] OllamaProvider creating client with:")
        # print(f"[DEBUG] - base_url: {base_url}")
        # print(f"[DEBUG] - api_key: {api_key}")
        # print(f"[DEBUG] - config received: {config}")
        
        return OpenAI(
            base_url=base_url,
            api_key=api_key
        )
    
    def generate_completion(self, client, model: str, messages: List[Dict], **kwargs):
        return client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=kwargs.get('temperature', 0),
            stream=kwargs.get('stream', False)
        )
    
    def validate_config(self, config: Dict[str, Any]) -> bool:
        # For Ollama, we just need the base URL to be reachable
        return True  # Simplified validation


# Removed HuggingFace provider - not needed for gpt-oss-20b

class GPT5Provider(ModelProvider):
    """Specialized provider for GPT-5 with different API structure"""

    def create_client(self, config: Dict[str, Any]):
        from openai import OpenAI
        return OpenAI(api_key=config.get('api_key'))

    def generate_completion(self, client, model: str, messages: List[Dict], **kwargs):
        # Convert messages to GPT-5 input format
        input_data = []
        for msg in messages:
            role = 'developer' if msg['role'] == 'system' else msg['role']
            input_data.append({'role': role, 'content': msg['content']})

        # Get reasoning effort from kwargs, default to medium
        effort_level = kwargs.get('reasoning_effort', 'medium')
        reasoning = {"effort": effort_level}

        return client.responses.create(
            model=model,
            input=input_data,
            reasoning=reasoning
        )

    def validate_config(self, config: Dict[str, Any]) -> bool:
        return 'api_key' in config and config['api_key'].strip() != ''







class ModelProviderFactory:
    """Factory to create appropriate model providers"""
    
    _providers = {
        'openai': OpenAIProvider(),
        'ollama': OllamaProvider(),
        'gpt5': GPT5Provider(),
    }
    
    # Model to provider mapping
    _model_providers = {
        'gpt-4': 'openai',
        'gpt-4o': 'openai',
        'gpt-4o-mini': 'openai',
        'gpt-5': 'gpt5',
        'o1': 'openai',
        'o1-mini': 'openai',
        'o3-mini': 'openai',
        'gpt-oss-20b': 'ollama',  # Default to Ollama for local inference
        # Local server models
        'llama3.1:70b': 'ollama',
        'llama4:latest': 'ollama',
        'qwen3:32b': 'ollama',
        'deepseek-r1:70b': 'ollama',
        'gpt-oss:120b': 'ollama',
        'gpt-oss:20b': 'ollama',
        'mistral:latest': 'ollama',
        'llama2:latest': 'ollama',
        'llama3.2:1b': 'ollama',
    }
    
    @classmethod
    def get_provider(cls, model: str) -> ModelProvider:
        """Get the appropriate provider for a model"""
        provider_name = cls._model_providers.get(model, 'openai')  # Default to OpenAI
        return cls._providers[provider_name]
    
    @classmethod
    def register_model(cls, model: str, provider: str):
        """Register a model with a specific provider"""
        cls._model_providers[model] = provider
    
    @classmethod
    def get_available_providers(cls) -> List[str]:
        """Get list of available providers"""
        return list(cls._providers.keys())


def load_model_config():
    """Load configuration for all model providers"""
    config = configparser.ConfigParser()
    config_path = os.path.join(current_script_dir, 'config.ini')
    config.read(config_path)
    
    model_config = {}
    
    # OpenAI config
    if 'API_Key' in config and 'OpenAI_key' in config['API_Key']:
        model_config['openai'] = {
            'api_key': config['API_Key']['OpenAI_key']
        }
    
    # GPT-5 config (uses same OpenAI key)
    if 'API_Key' in config and 'OpenAI_key' in config['API_Key']:
        model_config['gpt5'] = {
            'api_key': config['API_Key']['OpenAI_key']
        }
    
    # Ollama config (local) - Force to use your server
    model_config['ollama'] = {
        'base_url': 'http://128.118.54.16:11434/v1',  # Force your server URL
        'api_key': 'no-api'  # Match what works in LLM_SERVER_TESTING.py
    }
    
    # Debug logging
    # print(f"[DEBUG] Ollama config loaded: {model_config['ollama']}")
    
    # Removed HuggingFace config - not needed for gpt-oss-20b
    
    return model_config


def create_unified_client(model: str):
    """Create a unified client that can handle multiple providers"""
    provider = ModelProviderFactory.get_provider(model)
    config = load_model_config()
    
    # Get provider-specific config
    provider_name = ModelProviderFactory._model_providers.get(model, 'openai')
    provider_config = config.get(provider_name, {})
    
    if not provider.validate_config(provider_config):
        raise ValueError(f"Invalid configuration for {provider_name} provider")
    
    return provider.create_client(provider_config), provider


def generate_unified_completion(model: str, messages: List[Dict], **kwargs):
    """Generate completion using the appropriate provider for the model"""
    client, provider = create_unified_client(model)
    return provider.generate_completion(client, model, messages, **kwargs)