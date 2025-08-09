# Company Brochure Generator

A Python script that automatically scrapes company websites and generates professional brochures using Ollama LLM.

## Features

- **Web Scraping**: Automatically extracts content from company websites
- **Smart Link Detection**: Identifies relevant pages (About, Careers, etc.) for brochure content
- **LLM-Powered**: Uses Ollama with Llama 3.2 model to generate professional brochures
- **Streaming Output**: Option to stream brochure generation in real-time
- **Error Handling**: Robust error handling for network issues and parsing problems

## Prerequisites

1. **Ollama**: Make sure you have Ollama installed and running
2. **DeepSeek Model**: Pull the required model:
   ```bash
   ollama pull deepseek-r1:8b
   ```

## Installation

1. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Set up environment variables (optional):
   Create a `.env` file in the same directory:
   ```
   # Add any environment variables if needed
   ```

## Usage

### Basic Usage

Run the script directly:
```bash
python company_brochure_generator.py
```

This will generate a brochure for HuggingFace as an example.

### Custom Usage

You can also import and use the functions in your own code:

```python
from company_brochure_generator import create_brochure, stream_brochure

# Generate a brochure
brochure = create_brochure("Your Company", "https://yourcompany.com")
if brochure:
    print(brochure)

# Stream brochure generation
stream_brochure("Your Company", "https://yourcompany.com")
```

## Functions

### `create_brochure(company_name, url)`
- Generates a complete brochure for the specified company
- Returns the brochure content as a string
- Handles errors gracefully

### `stream_brochure(company_name, url)`
- Streams brochure generation in real-time
- Prints content as it's generated
- Useful for long-running generations

### `get_links(url)`
- Extracts relevant links from a website
- Uses LLM to identify important pages (About, Careers, etc.)
- Returns structured JSON with link information

### `Website(url)`
- Class for scraping and parsing website content
- Removes irrelevant elements (scripts, styles, images)
- Extracts clean text content and links

## Configuration

### Model Selection
Change the model in the script:
```python
MODEL = "deepseek-r1:8b"  # Change to your preferred model
```

### Brochure Tone
Modify the system prompt in `create_brochure()` function:
- **Professional**: Default tone for business brochures
- **Humorous**: Uncomment the humorous prompt for entertaining brochures

### Content Length
Adjust the content truncation in `get_brochure_user_prompt()`:
```python
web_content = web_content[:5000]  # Change 5000 to your preferred limit
```

## Error Handling

The script includes comprehensive error handling for:
- Network connectivity issues
- Invalid URLs
- JSON parsing errors
- LLM response errors
- Web scraping failures

## Example Output

The script generates markdown-formatted brochures with:
- Company overview
- Key services/products
- Company culture information
- Career opportunities
- Contact information

## Troubleshooting

1. **Ollama not running**: Make sure Ollama is installed and running
2. **Model not found**: Pull the required model with `ollama pull deepseek-r1:8b`
3. **Network errors**: Check your internet connection and URL accessibility
4. **JSON parsing errors**: The script includes fallback mechanisms for malformed responses

## Dependencies

- `requests`: HTTP requests for web scraping
- `beautifulsoup4`: HTML parsing
- `python-dotenv`: Environment variable management
- `openai`: OpenAI API client (for potential future use)
- `openai`: OpenAI client for LLM interactions (used with Ollama)

## License

This script is part of the GenAI Demo App collection.
