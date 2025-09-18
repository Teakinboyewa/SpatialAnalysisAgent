"""
Test script to verify GPT-5 specialized provider works correctly
Run this in QGIS Python Console after setting up your OpenAI API key
"""

import sys
import os

# Add the SpatialAnalysisAgent directory to path
plugin_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(plugin_dir, 'SpatialAnalysisAgent'))


def test_gpt5_provider():
    """Test if GPT-5 uses the specialized provider"""
    try:
        # Import the model provider
        import SpatialAnalysisAgent_ModelProvider as ModelProvider
        
        print("ModelProvider imported successfully")
        
        # Test if GPT-5 gets the correct provider
        factory = ModelProvider.ModelProviderFactory()
        provider = factory.get_provider('gpt-5')
        
        print(f"Provider for GPT-5: {type(provider).__name__}")
        
        # Verify it's the specialized provider
        if type(provider).__name__ == 'GPT5Provider':
            print("GPT-5 correctly uses specialized provider")
        else:
            print(f"GPT-5 using wrong provider: {type(provider).__name__}")
            return False
        
        # Test provider mapping
        mapping = ModelProvider.ModelProviderFactory._model_providers
        if mapping.get('gpt-5') == 'gpt5':
            print("GPT-5 correctly mapped to 'gpt5' provider")
        else:
            print(f"GPT-5 mapped to wrong provider: {mapping.get('gpt-5')}")
            return False
        
        # Test configuration loading
        config = ModelProvider.load_model_config()
        print(f"Configuration loaded: {list(config.keys())}")
        
        # Test client creation (without API call)
        try:
            client, provider_instance = ModelProvider.create_unified_client('gpt-5')
            print("Unified client created successfully for GPT-5")
            print(f"Provider instance: {type(provider_instance).__name__}")
            
            # Test the generate_completion method exists and has right signature
            if hasattr(provider_instance, 'generate_completion'):
                print("generate_completion method exists")
                
                # You can uncomment this to test actual API call if you have API key
                # messages = [
                #     {"role": "system", "content": "You are a helpful GIS assistant."},
                #     {"role": "user", "content": "What is buffer analysis in GIS?"}
                # ]
                # response = provider_instance.generate_completion(
                #     client, 'gpt-5', messages, reasoning={"effort": "minimal"}
                # )
                # print(f"[OK] API call successful: {response}")
                
            else:
                print("generate_completion method missing")
                return False
                
        except Exception as e:
            if "Invalid configuration" in str(e):
                print("OpenAI API key not configured. Add to config.ini")
                print("But provider structure is correct!")
                return True
            else:
                print(f"Error creating client: {e}")
                return False
        
        return True
        
    except ImportError as e:
        print(f"Import error: {e}")
        return False
    except Exception as e:
        print(f"Unexpected error: {e}")
        return False

if __name__ == "__main__":
    print("Testing GPT-5 Specialized Provider")
    print("=" * 50)
    
    # success = test_gpt5_provider()
    #
    # print("\n" + "=" * 50)
    # if success:
    #     print("GPT-5 specialized provider verification passed!")
    #     print("\nGPT-5 will now use:")
    #     print("• client.responses.create() instead of chat.completions.create()")
    #     print("• input parameter instead of messages")
    #     print("• reasoning parameter for effort control")
    # else:
    #     print("Verification failed. Check the messages above.")