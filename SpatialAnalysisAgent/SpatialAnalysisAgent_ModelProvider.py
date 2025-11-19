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
    def generate_completion(self, request_id, client, model: str, messages: List[Dict], **kwargs):
        """Generate completion using the provider's API"""
        pass
    
    @abstractmethod
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """Validate configuration for this provider"""
        pass


class OpenAIProvider(ModelProvider):
    """OpenAI API provider for GPT models"""

    def __init__(self):
        self.api_key = None
        self.base_url = None


    def create_client(self, config: Dict[str, Any]):
        from openai import OpenAI
        self.api_key = (config.get('api_key') or '').strip()
        self.base_url = None
        if 'gibd-services' in self.api_key:
            self.base_url = f"https://www.gibd.online/api/openai/{self.api_key}"
            client = None
        else:
            client = OpenAI(api_key=self.api_key)

        return client

        # return OpenAI(api_key=config.get('api_key'))
    
    def generate_completion(self, request_id, client, model: str, messages: List[Dict], **kwargs):
        import requests, json

        stream = kwargs.get("stream", False)

        # --- If using proxy key -------
        if 'gibd-services' in (self.api_key or ''):
            url = self.base_url
            payload = {"service_name":"GIS Copilot",
                       "question_id": request_id,
                       "model": model,
                       "messages": messages,
                       "stream":stream}
            response = requests.post(url, json=payload, stream=stream)

            # if not payload["stream"]:
            if not stream:
                # ---- Non-streaming ----
                try:
                    data = response.json()

                    # Check for error in response
                    if "error" in data:
                        error_msg = data.get("error", "Unknown error")
                        print(f"Error: {error_msg}")
                        raise Exception(f"API Error: {error_msg}")

                    class DummyChoice:
                        pass
                    class DummyMessage:
                        pass

                    message = DummyMessage()
                    message.content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
                    choice = DummyChoice()
                    choice.message = message
                    response.choices = [choice]
                except json.JSONDecodeError as e:
                    print(f"Error: Failed to parse response - {str(e)}")
                    raise Exception(f"Failed to parse API response: {str(e)}")
                except Exception as e:
                    # Re-raise if it's already our custom error
                    if "API Error:" in str(e):
                        raise
                    print(f"Error: {str(e)}")
                    response.choices = []

                return response

            else:
                # ---- Streaming mode ----
                def stream_generator():
                    if response.status_code != 200:
                        # Try to parse error message from response
                        try:
                            error_data = response.json()
                            if "error" in error_data:
                                error_msg = error_data.get("error", "Unknown error")
                                print(f"Error: {error_msg}")
                                yield f"[ERROR] {error_msg}"
                                return
                        except:
                            pass
                        yield f"[ERROR] {response.text}"
                        return

                    for line in response.iter_lines(decode_unicode=True):
                        if not line:
                            continue
                        if line.startswith("data: "):
                            data_str = line[6:]
                            if data_str == "[DONE]":
                                break
                            try:
                                chunk = json.loads(data_str)
                                # Check for error in streaming chunk
                                if 'error' in chunk:
                                    error_msg = chunk.get("error", "Unknown error")
                                    print(f"Error: {error_msg}")
                                    yield f"[ERROR] {error_msg}"
                                    return
                                if 'choices' in chunk and len(chunk['choices']) > 0:
                                    delta = chunk['choices'][0].get('delta', {})
                                    content = delta.get('content', '')
                                    if content:
                                        yield content  # token-by-token streaming
                            except json.JSONDecodeError:
                                continue

                # return a generator so user can iterate over streamed tokens
                return stream_generator()

        else:
            return client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=kwargs.get('temperature', 0),
                stream=kwargs.get('stream', stream)
            )
    
    def validate_config(self, config: Dict[str, Any]) -> bool:
        return 'api_key' in config and config['api_key'].strip() != ''


class OllamaProvider(ModelProvider):
    """Ollama local provider for gpt-oss-20b and other models"""
    
    def create_client(self, config: Dict[str, Any]):
        # Use OpenAI SDK with Ollama's compatible endpoint (exact match to LLM_SERVER_TESTING_v1.py)
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
    
    def generate_completion(self, request_id, client, model: str, messages: List[Dict], **kwargs):
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
    """Specialized provider for GPT-5 and GPT-5.1 with different API structure"""

    def __init__(self):
        self.api_key = None
        self.base_url = None

    def create_client(self, config: Dict[str, Any]):
        from openai import OpenAI
        self.api_key = (config.get('api_key') or '').strip()
        self.base_url = None
        if 'gibd-services' in self.api_key:
            self.base_url = f"https://www.gibd.online/api/openai/{self.api_key}"
            client = None
        else:
            client = OpenAI(api_key=self.api_key)
        return client

    def generate_completion(self, request_id, client, model: str, messages: List[Dict], **kwargs):
        import requests, json

        stream = kwargs.get("stream", False)

        # --- If using proxy key (GIBD service) -------
        if 'gibd-services' in (self.api_key or ''):
            url = self.base_url

            # Get reasoning effort from kwargs
            effort_level = kwargs.get('reasoning_effort', 'medium')

            # Map reasoning effort based on model
            if model == 'gpt-5.1':
                # GPT-5.1 only supports: none, low, high
                effort_mapping = {
                    'none': 'none',
                    'low': 'low',
                    'minimal': 'low',
                    'medium': 'low',  # Map medium to low
                    'high': 'high'
                }
                effort_level = effort_mapping.get(effort_level, 'low')

            payload = {
                "service_name": "GIS Copilot",
                "question_id": request_id,
                "model": model,
                "messages": messages,
                "stream": stream,
                "reasoning_effort": effort_level,  # Explicitly pass reasoning effort
                **{k: v for k, v in kwargs.items() if k not in ['reasoning_effort', 'stream']}
            }

            response = requests.post(url, json=payload, stream=stream)

            if not stream:
                # ---- Non-streaming ----
                try:
                    data = response.json()

                    # Check for error in response
                    if "error" in data:
                        error_msg = data.get("error", "Unknown error")
                        print(f"Error: {error_msg}")
                        raise Exception(f"API Error: {error_msg}")

                    class DummyChoice:
                        pass
                    class DummyMessage:
                        pass

                    message = DummyMessage()
                    message.content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
                    choice = DummyChoice()
                    choice.message = message
                    response.choices = [choice]
                except json.JSONDecodeError as e:
                    print(f"Error: Failed to parse response - {str(e)}")
                    raise Exception(f"Failed to parse API response: {str(e)}")
                except Exception as e:
                    # Re-raise if it's already our custom error
                    if "API Error:" in str(e):
                        raise
                    print(f"Error: {str(e)}")
                    response.choices = []

                return response
            else:
                # ---- Streaming mode ----
                def stream_generator():
                    if response.status_code != 200:
                        # Try to parse error message from response
                        try:
                            error_data = response.json()
                            if "error" in error_data:
                                error_msg = error_data.get("error", "Unknown error")
                                print(f"Error: {error_msg}")
                                yield f"[ERROR] {error_msg}"
                                return
                        except:
                            pass
                        yield f"[ERROR] {response.text}"
                        return

                    for line in response.iter_lines(decode_unicode=True):
                        if not line:
                            continue
                        if line.startswith("data: "):
                            data_str = line[6:]
                            if data_str == "[DONE]":
                                break
                            try:
                                chunk = json.loads(data_str)
                                # Check for error in streaming chunk
                                if 'error' in chunk:
                                    error_msg = chunk.get("error", "Unknown error")
                                    print(f"Error: {error_msg}")
                                    yield f"[ERROR] {error_msg}"
                                    return
                                if 'choices' in chunk and len(chunk['choices']) > 0:
                                    delta = chunk['choices'][0].get('delta', {})
                                    content = delta.get('content', '')
                                    if content:
                                        yield content  # token-by-token streaming
                            except json.JSONDecodeError:
                                continue

                # return a generator so user can iterate over streamed tokens
                return stream_generator()

        # --- If using direct OpenAI key -------
        else:
            try:
                # First try the specialized GPT-5/GPT-5.1 API format if available
                # Convert messages to GPT-5/5.1 input format
                input_data = []
                for msg in messages:
                    role = 'developer' if msg['role'] == 'system' else msg['role']
                    input_data.append({'role': role, 'content': msg['content']})

                # Get reasoning effort from kwargs
                effort_level = kwargs.get('reasoning_effort', 'medium')

                # Map reasoning effort based on model
                if model == 'gpt-5.1':
                    # GPT-5.1 only supports: none, low, high
                    effort_mapping = {
                        'none': 'none',
                        'low': 'low',
                        'minimal': 'low',
                        'medium': 'low',  # Map medium to low
                        'high': 'high'
                    }
                    effort_level = effort_mapping.get(effort_level, 'low')
                else:
                    # GPT-5 supports: minimal, low, medium, high
                    # Keep original effort level for GPT-5
                    pass

                reasoning = {"effort": effort_level}

                return client.responses.create(
                    model=model,
                    input=input_data,
                    reasoning=reasoning,
                    **{k: v for k, v in kwargs.items() if k not in ['reasoning_effort']}
                )
            except AttributeError:
                # Fallback to standard OpenAI chat completions API if responses.create doesn't exist
                print(f"{model} specialized API not available, falling back to standard chat completions...")

                # Add reasoning effort to the system message if provided
                effort_level = kwargs.get('reasoning_effort', 'medium')
                default_effort = 'low' if model == 'gpt-5.1' else 'medium'

                if effort_level and effort_level != default_effort:
                    # Enhance system message with reasoning instructions
                    for msg in messages:
                        if msg['role'] == 'system':
                            msg['content'] += f" Please use {effort_level} reasoning effort for this task."
                            break

                return client.chat.completions.create(
                    model=model,
                    messages=messages,
                    **{k: v for k, v in kwargs.items() if k not in ['reasoning_effort']}
                )
            except Exception as e:
                print(f"{model} API error: {str(e)}")
                # Final fallback to standard format
                return client.chat.completions.create(
                    model=model,
                    messages=messages,
                    **{k: v for k, v in kwargs.items() if k not in ['reasoning_effort']}
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
        'gpt-5.1': 'gpt5',
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
        'api_key': 'no-api'  # Match what works in LLM_SERVER_TESTING_v1.py
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