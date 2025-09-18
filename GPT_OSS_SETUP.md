# GPT-OSS-20B Integration Setup Guide

This guide explains how to set up gpt-oss-20b with your SpatialAnalysisAgent plugin.

## Simple Setup with Ollama (Only Option Needed!)

### 1. Install Ollama

**Windows**:
```bash
# Download from https://ollama.ai/ 
# Or use winget:
winget install ollama
```

**Linux**:
```bash
curl -fsSL https://ollama.ai/install.sh | sh
```

**macOS**:
```bash
brew install ollama
```

### 2. Pull gpt-oss-20b model
```bash
# For 20B model (recommended - fits in 16GB RAM)
ollama pull gpt-oss:20b

# For 120B model (requires 60GB+ RAM)
ollama pull gpt-oss:120b
```

### 3. Test the model (Optional)
```bash
ollama run gpt-oss:20b "Explain what a buffer analysis is in GIS"
```

### 4. Configure SpatialAnalysisAgent
- **Restart QGIS** (important!)
- Open **SpatialAnalysisAgent → Settings tab**
- Select **"gpt-oss-20b"** from the Models dropdown  
- **That's it!** No API keys needed

### Alternative Setup Options

#### Option A: Together AI (Cloud)

1. **Get Together AI API Key**:
   - Visit https://api.together.xyz/
   - Sign up and get your API key

2. **Configure as OpenAI-compatible endpoint**:
   ```ini
   [TogetherAI]
   api_key = your_together_api_key
   base_url = https://api.together.xyz/v1
   ```

## Hardware Requirements

### Local Installation (Ollama)
- **Minimum**: 16GB RAM (with quantization)
- **Recommended**: 32GB RAM for optimal performance
- **GPU**: Not required but improves speed
- **Storage**: ~12GB for model weights

### Cloud Options
- No local hardware requirements
- Pay per token/request

## Troubleshooting

### Common Issues

1. **"Connection refused" error**:
   - Ensure Ollama is running: `ollama serve`
   - Check if port 11434 is available

2. **Model not found**:
   - Verify model is pulled: `ollama list`
   - Pull again if missing: `ollama pull gpt-oss:20b`

3. **Out of memory errors**:
   - Use quantized version: `ollama pull gpt-oss:20b-q4`
   - Close other applications to free RAM

4. **Slow responses**:
   - Consider using GPU acceleration
   - Use smaller model variants
   - Increase timeout settings

### Verification

Test the integration with these steps:

1. **Open QGIS** and load the SpatialAnalysisAgent plugin
2. **Go to Settings tab**
3. **Select "gpt-oss-20b"** from Models dropdown
4. **Enter a simple request**: "Create a buffer of 100 meters around points"
5. **Check output** for proper response

## Performance Comparison

| Model | Speed | Quality | Memory | Cost |
|-------|-------|---------|---------|------|
| gpt-oss-20b (Local) | Medium | High | 16GB+ | Free |
| gpt-4o (OpenAI) | Fast | High | N/A | $$ |
| gpt-4 (OpenAI) | Medium | High | N/A | $$$ |

## Advanced Configuration

### Custom Ollama Settings

Edit `config.ini` to customize Ollama connection:

```ini
[Ollama]
base_url = http://localhost:11434/v1
timeout = 120
max_tokens = 2000
temperature = 0.7
```

### Model-Specific Prompts

The gpt-oss-20b model works best with:
- Clear, specific instructions
- Step-by-step task breakdown
- Structured output requests
- Reasoning level hints: "Reasoning: high"

## Next Steps

After setup, you can:
1. Fine-tune the model for GIS-specific tasks
2. Experiment with different reasoning levels
3. Compare performance with OpenAI models
4. Contribute improvements back to the community

For issues, check the [GitHub repository](https://github.com/Teakinboyewa/SpatialAnalysisAgent) or create an issue.