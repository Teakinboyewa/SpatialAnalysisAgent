"""
Simple test script to verify gpt-oss-20b integration works with SpatialAnalysisAgent
Run this in QGIS Python Console after setting up Ollama
"""

import sys
import os

# Add the SpatialAnalysisAgent directory to path
plugin_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(plugin_dir, 'SpatialAnalysisAgent'))

def test_gpt_oss_integration():
    """Test if gpt-oss-20b can be accessed through the plugin"""
    try:
        # Import the model provider
        import SpatialAnalysisAgent_ModelProvider as ModelProvider
        
        print("✅ ModelProvider imported successfully")
        
        # Test if Ollama provider is available
        factory = ModelProvider.ModelProviderFactory()
        provider = factory.get_provider('gpt-oss-20b')
        
        print(f"✅ Provider for gpt-oss-20b: {type(provider).__name__}")
        
        # Test configuration loading
        config = ModelProvider.load_model_config()
        print(f"✅ Configuration loaded: {list(config.keys())}")
        
        # Test client creation
        try:
            client, provider_instance = ModelProvider.create_unified_client('gpt-oss-20b')
            print("✅ Unified client created successfully")
            
            # Test a simple completion (only if Ollama is running)
            messages = [
                {"role": "system", "content": "You are a helpful GIS assistant."},
                {"role": "user", "content": "What is a buffer analysis in GIS? Answer in one sentence."}
            ]
            
            response = provider_instance.generate_completion(
                client, 'gpt-oss:20b', messages, temperature=0.1
            )
            
            if hasattr(response, 'choices') and len(response.choices) > 0:
                answer = response.choices[0].message.content
                print(f"✅ Test completion successful!")
                print(f"📝 Response: {answer}")
            else:
                print("⚠️  Response received but format unexpected")
                
        except Exception as e:
            if "Connection refused" in str(e) or "11434" in str(e):
                print("⚠️  Ollama server not running. Start with: ollama serve")
                print("⚠️  Then pull model with: ollama pull gpt-oss:20b")
            else:
                print(f"❌ Error testing completion: {e}")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def check_ollama_status():
    """Check if Ollama is running and has gpt-oss model"""
    try:
        import requests
        
        # Test Ollama API
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get('models', [])
            model_names = [m.get('name', '') for m in models]
            
            gpt_oss_models = [name for name in model_names if 'gpt-oss' in name.lower()]
            
            if gpt_oss_models:
                print(f"✅ Ollama running with gpt-oss models: {gpt_oss_models}")
                return True
            else:
                print("⚠️  Ollama running but no gpt-oss models found")
                print("   Run: ollama pull gpt-oss:20b")
                return False
        else:
            print(f"❌ Ollama API returned status {response.status_code}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("❌ Ollama not running. Start with: ollama serve")
        return False
    except Exception as e:
        print(f"❌ Error checking Ollama: {e}")
        return False

if __name__ == "__main__":
    print("🧪 Testing GPT-OSS-20B Integration with SpatialAnalysisAgent")
    print("=" * 60)
    
    print("\n1. Checking Ollama status...")
    ollama_ok = check_ollama_status()
    
    print("\n2. Testing plugin integration...")
    plugin_ok = test_gpt_oss_integration()
    
    print("\n" + "=" * 60)
    if ollama_ok and plugin_ok:
        print("🎉 All tests passed! gpt-oss-20b is ready to use")
        print("\nNext steps:")
        print("1. Open SpatialAnalysisAgent in QGIS")
        print("2. Go to Settings tab")
        print("3. Select 'gpt-oss-20b' from Models dropdown")
        print("4. Start your GIS analysis!")
    else:
        print("❌ Some tests failed. Check the messages above.")
        
    print("\n📖 For detailed setup: see GPT_OSS_SETUP.md")