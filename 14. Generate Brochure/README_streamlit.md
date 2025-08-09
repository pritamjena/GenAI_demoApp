# Streamlit Company Brochure Generator

A beautiful web interface for generating company brochures using Ollama LLM with real-time streaming capabilities.

## Features

- **🎨 Beautiful Web Interface**: Clean, modern UI built with Streamlit
- **📝 Easy Input**: Simple form for company name and URL
- **⚡ Real-time Streaming**: Watch brochure generation in real-time
- **🔧 Model Selection**: Choose from different Ollama models
- **🧹 Clean Output**: Automatic removal of `<think></think>` tags
- **📊 Live Status**: Real-time connection and URL status checks
- **📱 Responsive Design**: Works on desktop and mobile

## Screenshots

The app features:
- **Left Panel**: Input form with company details and streaming options
- **Right Panel**: Live status updates and model information
- **Sidebar**: Configuration options and instructions

## Prerequisites

1. **Ollama**: Make sure you have Ollama installed and running
2. **DeepSeek Model**: Pull the required model:
   ```bash
   ollama pull deepseek-r1:8b
   ```

## Installation

1. Install the required dependencies:
   ```bash
   pip install -r requirements_streamlit.txt
   ```

2. Make sure Ollama is running:
   ```bash
   ollama serve
   ```

## Usage

### Running the Streamlit App

1. Navigate to the directory:
   ```bash
   cd "GenAI_demoApp/14. Generate Brochure"
   ```

2. Run the Streamlit app:
   ```bash
   streamlit run streamlit_brochure_app.py
   ```

3. Open your browser to the URL shown (usually `http://localhost:8501`)

### Using the App

1. **Enter Company Details**:
   - Company Name: e.g., "HuggingFace", "OpenAI", "Microsoft"
   - Website URL: e.g., "https://huggingface.co"

2. **Choose Options**:
   - **Streaming Mode**: Enable for real-time generation
   - **Model Selection**: Choose from available models in sidebar

3. **Generate Brochure**:
   - Click "🚀 Generate Brochure" button
   - Watch the magic happen!

## Key Features

### 🎯 **Think Tag Removal**
The app automatically removes `<think></think>` tags from LLM outputs using regex:
```python
def remove_think_tags(text):
    # Remove <think> tags and their content
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    # Also remove any remaining <think> or </think> tags
    text = re.sub(r'</?think>', '', text)
    return text.strip()
```

### ⚡ **Streaming Mode**
- **Enabled**: Real-time output as content is generated
- **Disabled**: Complete brochure returned at once
- **Visual Feedback**: Progress indicators and status updates

### 🔧 **Model Selection**
Choose from popular models:
- `deepseek-r1:8b` (default)
- `llama3.2`
- `mistral`
- `codellama`

### 📊 **Live Status Monitoring**
- **Ollama Connection**: Real-time connection status
- **URL Reachability**: Check if website is accessible
- **Model Information**: Current model and settings

## Configuration

### Model Selection
Change the default model in the script:
```python
MODEL = "deepseek-r1:8b"  # Change to your preferred model
```

### Available Models
Add more models to the dropdown:
```python
model_options = ["deepseek-r1:8b", "llama3.2", "mistral", "codellama", "your-model"]
```

### Customization
- **Page Title**: Modify `page_title` in `st.set_page_config()`
- **Layout**: Change `layout="wide"` for different layouts
- **Styling**: Customize CSS in the footer section

## Error Handling

The app includes comprehensive error handling:
- **Network Issues**: Graceful handling of connection problems
- **Invalid URLs**: Clear error messages for unreachable sites
- **Model Errors**: Helpful messages for missing models
- **Streaming Errors**: Fallback to non-streaming mode

## Troubleshooting

### Common Issues

1. **"Ollama connection failed"**
   - Make sure Ollama is running: `ollama serve`
   - Check if the model is pulled: `ollama list`

2. **"URL is not reachable"**
   - Check the URL format (include `https://`)
   - Verify the website is accessible
   - Try a different website

3. **"Model not found"**
   - Pull the required model: `ollama pull deepseek-r1:8b`
   - Check available models: `ollama list`

4. **Streamlit not starting**
   - Check if Streamlit is installed: `pip install streamlit`
   - Verify Python version compatibility

### Performance Tips

1. **Faster Generation**: Use smaller models for quicker results
2. **Better Quality**: Use larger models for more detailed brochures
3. **Streaming**: Enable for long generations to see progress
4. **Caching**: Streamlit automatically caches results

## File Structure

```
14. Generate Brochure/
├── streamlit_brochure_app.py      # Main Streamlit app
├── company_brochure_generator.py  # Command-line version
├── requirements_streamlit.txt      # Streamlit dependencies
├── requirements.txt               # CLI dependencies
├── README_streamlit.md           # This file
└── README.md                    # CLI documentation
```

## Dependencies

- `streamlit`: Web framework for the interface
- `requests`: HTTP requests for web scraping
- `beautifulsoup4`: HTML parsing
- `python-dotenv`: Environment variable management
- `openai`: OpenAI client for Ollama interactions

## Development

### Adding New Features

1. **New Models**: Add to `model_options` list
2. **Custom Prompts**: Modify system prompts in functions
3. **UI Enhancements**: Add new Streamlit components
4. **Output Formats**: Add export options (PDF, HTML, etc.)

### Testing

1. **Local Testing**: Run with `streamlit run streamlit_brochure_app.py`
2. **Model Testing**: Test with different models
3. **URL Testing**: Try various company websites
4. **Error Testing**: Test with invalid inputs

## License

This Streamlit app is part of the GenAI Demo App collection.

---

**Happy Brochure Generating! 🚀**
