# Model Selection Guide

## Overview

SpatialAnalysisAgent supports multiple AI models from OpenAI including GPT-4, GPT-4o, GPT-5, and other OpenAI models. Choosing the right model depends on your priorities: speed vs. reasoning capability.

**Default Model**: GPT-5 is set as the default model, offering the best reasoning capabilities for complex spatial analysis tasks.

---

## GPT-5 vs GPT-4o: Understanding the Tradeoff

### GPT-5: Deep Reasoning for Complex Tasks
**GPT-5** is designed for tasks that require advanced reasoning and problem-solving capabilities. It excels at:
- Breaking down complex spatial analysis tasks into logical steps
- Generating sophisticated Python code with proper error handling
- Understanding nuanced GIS operations and their dependencies

**Tradeoff**: GPT-5 requires **more processing time** and has **longer wait times** compared to GPT-4o. Use GPT-5 when the task complexity justifies the wait.

### GPT-4o: Fast Responses for Simpler Tasks
**GPT-4o** offers a balanced approach with:
- Faster response times
- Efficient processing for straightforward GIS operations
- Good performance on well-defined tasks

**Recommendation**: Use GPT-4o when you need quick results for simpler spatial analysis tasks or when working iteratively.

---

## Smart Model Selection Strategy

### When GPT-5 is Selected

If you select **GPT-5**, the system uses a **hybrid approach** to optimize both quality and speed:

**High-Reasoning Operations (Use GPT-5):**
1. **Task Breakdown** - Converting your natural language request into structured GIS operations
2. **Code Generation** - Writing Python code to execute the spatial analysis
3. **Code Debugging** - Fixing errors in generated code (if applicable)

**Fast Operations (Use GPT-4o):**
1. **Task Name Generation** - Creating a concise name for your task
2. **Data Overview** - Analyzing input data characteristics
3. **Tool Selection** - Choosing appropriate QGIS tools for the task
4. **Geoprocessing Workflow** - Creating the execution graph/workflow

This hybrid strategy ensures that complex reasoning tasks benefit from GPT-5's capabilities while routine operations complete quickly with GPT-4o.

### When Other Models are Selected

If you select any model **other than GPT-5** (e.g., GPT-4o, GPT-4o-mini), **all API calls use the selected model** consistently throughout the entire workflow.

---

## API Calls and Their Purpose

SpatialAnalysisAgent makes several API calls during task execution. Understanding each call helps you appreciate how the system works:

### 1. Task Name Generation
- **Purpose**: Creates a short, descriptive name for your task (1-2 words)
- **Model Used**:
  - GPT-4o (if GPT-5 is selected)
  - Selected model (for all other models)
- **Example**: "Buffer_Analysis" for a task involving buffer operations

### 2. Task Breakdown
- **Purpose**: Converts your natural language request into a structured, step-by-step GIS task description
- **Model Used**: The model you selected (including GPT-5 if applicable)
- **Example Input**: "Find all counties in Pennsylvania with rainfall greater than 2.5 inches"
- **Example Output**: Structured task with labeled operations (Attribute Query, Calculate Area, Count Features)
- **Complexity**: High reasoning required

### 3. Data Overview
- **Purpose**: Analyzes your input data (shapefiles, rasters, etc.) to understand:
  - Geometry types
  - Coordinate reference systems (CRS)
  - Attribute fields and data types
  - Spatial extent
- **Model Used**:
  - GPT-4o (if GPT-5 is selected)
  - Selected model (for all other models)

### 4. Tool Selection
- **Purpose**: Selects the most appropriate QGIS tools and Python libraries for your task
- **Model Used**:
  - GPT-4o (if GPT-5 is selected)
  - Selected model (for all other models)
- **Example**: Selects "native:buffer" for buffer operations, or recommends seaborn for chart creation

### 5. Geoprocessing Workflow Generation
- **Purpose**: Creates a directed graph showing the workflow and dependencies between operations
- **Model Used**:
  - GPT-4o (if GPT-5 is selected)
  - Selected model (for all other models)
- **Output**: A GraphML file and HTML visualization of the workflow

### 6. Code Generation
- **Purpose**: Writes Python code to execute the spatial analysis task
- **Model Used**: The model you selected (including GPT-5 if applicable)
- **Complexity**: High reasoning required for:
  - Proper parameter handling
  - Error handling
  - Data type conversions
  - QGIS API usage

### 7. Code Debugging (if applicable)
- **Purpose**: Automatically fixes errors in generated code if execution fails
- **Model Used**: The model you selected (including GPT-5 if applicable)
- **Attempts**: Up to 5 debugging iterations
- **Complexity**: High reasoning required to understand error messages and apply fixes

---

## Supported Models

SpatialAnalysisAgent supports the following OpenAI models:

- **gpt-5** - Latest reasoning model with advanced capabilities
- **gpt-4o** - Balanced performance and speed
- **gpt-4o-mini** - Lighter, faster version of GPT-4o
- **gpt-4** - Previous generation with strong performance
- **o1** - Specialized reasoning model
- **o1-mini** - Lighter reasoning model
- **o3-mini** - Compact reasoning model

---

## OpenAI Model Details and Capabilities

For comprehensive information about OpenAI models, including:
- Detailed specifications
- Token limits
- Pricing
- Best use cases
- Performance benchmarks

**Visit the official OpenAI documentation:**
[https://platform.openai.com/docs/models](https://platform.openai.com/docs/models)

This resource provides up-to-date information about:
- Model capabilities and limitations
- Context windows and token limits
- Training data cutoff dates
- API usage guidelines
- Rate limits and quotas

---

## Best Practices for Model Selection

### Choose GPT-5 when:
- Working with complex, multi-step spatial analysis tasks
- Generating code for intricate GIS workflows
- Dealing with ambiguous or poorly defined requirements
- Code quality and correctness are more important than speed

### Choose GPT-4o when:
- Performing routine spatial analysis operations
- Working with well-defined, straightforward tasks
- Speed is a priority
- Iterating rapidly on multiple tasks

---

## Configuration

All configuration is done through the **SpatialAnalysisAgent Settings Tab** in the QGIS interface:

### 1. API Key Configuration
Navigate to the Settings tab and input your API key. You have two options:

**Option A: OpenAI API Key**
- Get your API key from [OpenAI Platform](https://platform.openai.com/api-keys)
- Input the key directly in the API Key field
- This gives you access to all OpenAI models (GPT-4, GPT-4o, GPT-5, etc.)

**Option B: GIBD-Service Key**
- Alternative API service provider: [https://www.gibd.online/](https://www.gibd.online/)
- Input your GIBD-Service key (format: `gibd-services-...`)
- The system automatically detects GIBD-Service keys and routes requests accordingly

### 2. Model Selection
Choose your preferred model from the dropdown menu in the Settings tab:
- Available models: gpt-5, gpt-4o, gpt-4o-mini, gpt-4, o1, o1-mini, o3-mini

### 3. Reasoning Effort (GPT-5 only)
When using GPT-5, adjust the reasoning effort level:
- **Minimal**: Fastest responses, basic reasoning
- **Low**: Faster responses, less thorough reasoning
- **Medium**: Balanced approach (default)
- **High**: Maximum reasoning depth, slower responses

### 4. Additional Settings
- **Code Review**: Enable/disable automatic code review
- **Use RAG**: Enable/disable Retrieval-Augmented Generation for tool selection

---

## Performance Considerations

### Response Time Factors:
1. **Model Selection**: GPT-5 > GPT-4o > GPT-4o-mini
2. **Task Complexity**: More complex tasks take longer
3. **Code Review**: Enabling code review adds one additional API call
4. **Debugging**: Each debugging iteration adds time (up to 5 attempts)
5. **Network Latency**: API calls depend on internet connection speed

### Optimization Tips:
- Disable code review for simple tasks to save time
- Use GPT-4o for exploratory analysis and GPT-5 for production code
- Provide clear, specific task descriptions to minimize iterations
- Ensure input data is properly formatted to avoid debugging cycles

---

## Troubleshooting

### Model Not Working?
1. Check your API key in the Settings tab
2. Verify internet connectivity
3. Ensure sufficient API credits (for OpenAI/GIBD-Service)
4. Check model availability (some models may be in beta/preview)
5. For GIBD-Service keys, ensure the format is correct (gibd-services-...)

### Slow Performance?
1. Try GPT-4o instead of GPT-5
2. Disable code review for faster iteration
3. Simplify your task description
4. Check network connectivity

### Poor Quality Results?
1. Try GPT-5 for better reasoning
2. Enable code review
3. Provide more detailed task descriptions
4. Include example data or expected output

---

## Additional Resources

- [OpenAI Platform Documentation](https://platform.openai.com/docs/models)
- [QGIS Python API Documentation](https://qgis.org/pyqgis/)
- [SpatialAnalysisAgent User Manual](./User_Manual.md)

---

**Last Updated**: Based on SpatialAnalysisAgent implementation analysis