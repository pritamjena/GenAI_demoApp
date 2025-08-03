# Blog Generation Agentic AI System

A sophisticated blog generation system built with LangGraph and FastAPI that creates high-quality, research-driven blog content in multiple languages.

## Features

- **Multi-language Output**: Generate blogs in various languages while keeping input topics in English
- **AI-Powered Content**: Uses advanced LLM models for creating authoritative, research-backed content
- **Structured Workflow**: Implements a two-stage process (title creation + content generation) using LangGraph
- **RESTful API**: FastAPI-based endpoint for easy integration
- **Professional Quality**: Generates comprehensive, educational content with proper formatting
- **Smart Translation**: Automatically translates English topics into the target language for content generation

## Installation

1. Clone the repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
# Create a .env file with your API keys
LANGCHAIN_API_KEY=your_langchain_api_key_here
```

## Usage

### Starting the Server

```bash
python app.py
```

The server will start on `http://localhost:8000`

### API Endpoints

#### POST /blogs

Generate a blog post with the specified topic (in English) and output language.

**Request Body:**
```json
{
    "topic": "Artificial Intelligence in Healthcare",
    "language": "Spanish"
}
```

**Parameters:**
- `topic` (required): The main topic for the blog post (always in English)
- `language` (optional): The output language for blog generation (defaults to "English")

**Response:**
```json
{
    "data": {
        "topic": "Artificial Intelligence in Healthcare",
        "language": "Spanish",
        "blog": {
            "title": "Análisis Integral: Aplicaciones de IA en la Atención Médica Moderna",
            "content": "# Resumen Ejecutivo\n\nEste análisis integral explora..."
        }
    }
}
```

### Example Requests

#### Spanish Blog (English Topic)
```bash
curl -X POST "http://localhost:8000/blogs" \
     -H "Content-Type: application/json" \
     -d '{"topic": "Machine Learning Basics", "language": "Spanish"}'
```

#### French Blog (English Topic)
```bash
curl -X POST "http://localhost:8000/blogs" \
     -H "Content-Type: application/json" \
     -d '{"topic": "Artificial Intelligence", "language": "French"}'
```

#### Hindi Blog (English Topic)
```bash
curl -X POST "http://localhost:8000/blogs" \
     -H "Content-Type: application/json" \
     -d '{"topic": "Data Science Fundamentals", "language": "Hindi"}'
```

#### Chinese Blog (English Topic)
```bash
curl -X POST "http://localhost:8000/blogs" \
     -H "Content-Type: application/json" \
     -d '{"topic": "Blockchain Technology", "language": "Chinese"}'
```

## How It Works

1. **Input**: You provide a topic in English (e.g., "Machine Learning Basics")
2. **Language Selection**: You specify the desired output language (e.g., "Spanish")
3. **Translation**: The system automatically translates the English topic into the target language
4. **Content Generation**: Creates comprehensive blog content in the specified language
5. **Output**: Returns both title and content in the target language

## Supported Languages

The system supports blog generation in various languages including:

- English
- Spanish
- French
- German
- Hindi
- Chinese
- And many more...

## Architecture

### Components

1. **BlogState** (`src/states/blogstate.py`): Defines the state structure with topic, language, and blog content
2. **BlogNode** (`src/nodes/blog_node.py`): Contains the logic for title creation and content generation with translation
3. **GraphBuilder** (`src/graphs/graph_builder.py`): Builds the LangGraph workflow
4. **GroqLLM** (`src/llms/groqllm.py`): LLM integration for content generation

### Workflow

1. **Input Processing**: Receives English topic and target language parameters
2. **Topic Translation**: Translates the English topic into the target language
3. **Title Generation**: Creates an authoritative, SEO-optimized title in the target language
4. **Content Generation**: Generates comprehensive, research-backed content in the target language
5. **Output**: Returns structured blog with title and content in the specified language

## Testing

Run the test suite to verify functionality:

```bash
python test_api.py
```

The test suite includes:
- Basic blog generation
- Multi-language testing with English topics
- Error handling
- API endpoint validation

## Configuration

### Environment Variables

- `LANGCHAIN_API_KEY`: Your LangChain API key for LLM access

### Customization

You can customize the blog generation by modifying:
- Prompt templates in `src/nodes/blog_node.py`
- Language-specific instructions and cultural considerations
- Content structure and formatting
- Output length and style preferences

## Key Benefits

- **Consistent Input**: Always use English for topics, regardless of output language
- **Automatic Translation**: No need to translate topics manually
- **Cultural Adaptation**: Content is adapted for the target language and culture
- **Professional Quality**: Maintains high standards across all languages
- **Easy Integration**: Simple API interface for multi-language content generation

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## License

This project is licensed under the MIT License.
