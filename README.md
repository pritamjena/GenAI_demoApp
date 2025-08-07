# 🚀 GenAI Demo Applications

A comprehensive collection of **12 exciting Generative AI applications** showcasing various AI capabilities, from news agents to data analysis, chatbots, and more. Each application demonstrates different aspects of modern AI technologies including RAG, agents, function calling, and multi-modal AI.

![Python](https://img.shields.io/badge/Python-100%25-blue)
![Status](https://img.shields.io/badge/Status-Active-green)
![License](https://img.shields.io/badge/License-MIT-orange)

---

## 📋 Project Structure

### 01. 🌐 AI News Agent with Streamlit & Ollama  
**Streamlit-powered news search agent using local LLM**
- Features DuckDuckGo search integration with Ollama (DeepSeek-R1)
- Interactive web interface for real-time news queries
- Uses LangChain's ReAct agent pattern for intelligent search

### 02. 💰 Financial Data with Agno  
**Multi-agent financial analysis system**  
- YFinance integration for stock data and analyst recommendations
- Web search capabilities for latest financial news
- Team-based AI agents working together for comprehensive analysis

### 03. ⚡ Function Calling  
**Weather information and deal finder using API integration**
- Real-time weather data with RapidAPI integration  
- Amazon deals scraper with function calling capabilities
- Demonstrates LLM integration with external APIs

### 04. 🌍 Website Assistance  
**Intelligent website content assistant with RAG**
- FAISS vector database for website content indexing
- Comprehensive web scraping with tab content extraction
- Q&A system powered by vector similarity search

### 05. 💬 Chat with Webpage  
**Interactive webpage chatbot using RAG**  
- Real-time webpage content ingestion and processing
- Streamlit interface for seamless user interaction
- ChromaDB integration for efficient document retrieval

### 06. 📊 Data Analysis Agent  
**Advanced data analysis with AI-powered SQL generation**
- CSV/Excel file upload and processing capabilities
- DuckDB integration for complex data queries
- Interactive Streamlit dashboard for data exploration

### 07. 🧮 Income Tax Calculator  
**AI-powered tax consultation system**
- Vector-based knowledge retrieval from tax websites
- Intelligent tax calculation assistance
- Multi-interface support (CLI and Streamlit)

### 08. 🔌 FastAPI Demo  
**RESTful API server for Ollama model interaction**
- Complete API endpoints for chat and text generation
- Health monitoring and model listing capabilities
- Production-ready FastAPI implementation

### 09. 🤖 Agentic AI - Basic Chatbot  
**LangGraph-powered conversational AI**
- Modular architecture with separate components
- State management for conversation context
- Advanced agent orchestration using LangGraph

### 10. 🛠️ Agentic AI - Chatbot with Tools  
**Enhanced chatbot with external tool integration**
- Extended tool capabilities for complex tasks
- Multi-modal interaction support
- Advanced agent reasoning and decision-making

### 11. 📰 Agentic AI - AI News Agent  
**Sophisticated news processing and analysis system**
- End-to-end agentic AI pipeline for news content
- Advanced content curation and summarization
- Multi-source news aggregation capabilities

### 12. ✍️ Agentic AI - Blog Generation  
**Multi-language blog content generation system**
- FastAPI-based blog creation with language support
- LangGraph workflow for structured content generation
- Automatic translation and cultural adaptation

---

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Ollama installed and running locally
- Required API keys (Groq, RapidAPI, etc.)

### Installation
```bash
# Clone the repository
git clone https://github.com/pritamjena/GenAI_demoApp.git
cd GenAI_demoApp

# Navigate to any application folder
cd "01. AI News Agent with Streamlit & Ollama"

# Install dependencies (varies by application)
pip install -r requirements.txt  # if available
# or install manually based on imports

# Run the application
streamlit run news_agent.py
```

### Environment Setup
Create a `.env` file in the root directory:
```env
GROQ_API_KEY=your_groq_api_key
RAPIDAPI_KEY=your_rapidapi_key
LANGCHAIN_API_KEY=your_langchain_api_key
LANGCHAIN_ENDPOINT=your_langchain_endpoint
LANGCHAIN_PROJECT=your_project_name
```

---

## 🔧 Technology Stack

| Technology | Applications Used | Purpose |
|------------|-------------------|---------|
| **LangChain** | 1, 3, 4, 5, 7 | Agent orchestration, RAG implementation |
| **Streamlit** | 1, 5, 6, 7 | Interactive web interfaces |
| **Ollama** | 1, 3, 4, 5, 7, 8, 9 | Local LLM inference |
| **FastAPI** | 8, 12 | RESTful API development |
| **LangGraph** | 9, 10, 11, 12 | Advanced agent workflows |
| **Agno/Phi** | 2, 6 | Multi-agent systems |
| **FAISS/ChromaDB** | 4, 5, 7 | Vector storage and retrieval |
| **DuckDB** | 6 | In-memory data analysis |

---

## ✨ Features Highlights

- 🤖 **12 Unique AI Applications** - Each solving different real-world problems
- 🔍 **RAG Implementation** - Multiple applications demonstrate Retrieval-Augmented Generation
- 🧠 **Agent Orchestration** - Advanced multi-agent systems and workflows
- 🌐 **Web Integration** - Real-time data fetching and web scraping capabilities
- 📊 **Data Analysis** - Intelligent data processing and visualization
- 🌍 **Multi-language Support** - International content generation capabilities
- ⚡ **Local LLM Support** - Privacy-focused local model integration
- 🔌 **API Integration** - External service connectivity and function calling

---

## 🎯 Use Cases

**For Developers:**
- Learn modern AI application development patterns
- Understand RAG implementation strategies
- Explore agent-based architecture designs

**For Businesses:**
- News monitoring and analysis automation
- Financial data processing and insights
- Customer support chatbot implementation
- Content generation and localization

**For Researchers:**
- Multi-agent system experimentation
- RAG performance optimization
- Conversational AI development

---

## 🤝 Contributing

We welcome contributions! Here's how you can help:

1. **Fork the repository**
2. **Create a feature branch** (`git checkout -b feature/AmazingFeature`)
3. **Add your improvements** (new applications, bug fixes, documentation)
4. **Commit your changes** (`git commit -m 'Add some AmazingFeature'`)
5. **Push to the branch** (`git push origin feature/AmazingFeature`)
6. **Open a Pull Request**

### Contribution Ideas:
- Add new AI application demos
- Improve existing application UIs
- Add comprehensive testing
- Enhance documentation
- Optimize performance

---

## 💬 Support

- **Issues**: Report bugs or request features via [GitHub Issues](https://github.com/pritamjena/GenAI_demoApp/issues)
- **Discussions**: Join conversations in [GitHub Discussions](https://github.com/pritamjena/GenAI_demoApp/discussions)
- **Documentation**: Check individual folder READMEs for specific application details

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## ⭐ Star History

If you find this project helpful, please consider giving it a star ⭐ to help others discover it!

---

**Made with ❤️ by the GenAI Community**

*Explore the future of AI applications - one demo at a time!*