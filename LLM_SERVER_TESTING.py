from openai import OpenAI

client = OpenAI(
    base_url="http://128.118.54.16:11434/v1",
    api_key="no-api"
)

# Test connection and list models
models = client.models.list()
print("Available models:")
for model in models.data:
    print(f"  - {model.id}")
