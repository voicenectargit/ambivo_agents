# Ambivo Agents 🤖

**A Comprehensive Multi-Agent System for AI-Powered Automation**

```
Author: Hemant Gosain 'Sunny'
Company: Ambivo
Email: sgosain@ambivo.com
License: MIT Open Source License
Version: 1.2.0
GitHub: https://github.com/yourusername/ambivo-agents
Company: https://www.ambivo.com
```

## Overview

Ambivo Agents is a powerful, production-ready multi-agent system that provides AI-powered automation capabilities including media processing, knowledge base operations, web scraping, code execution, and intelligent routing. Built with enterprise-grade features like Redis memory management, multi-LLM provider support, and Docker-based security.

## 🚀 Key Features

### 🎥 **Media Processing Agent**
- Extract audio from video files using FFmpeg
- Convert between audio/video formats (MP4, AVI, MP3, WAV, etc.)
- Resize, trim, and process media files
- Create thumbnails and extract frames
- Adjust audio volume and levels
- Merge audio/video files

### 🧠 **Knowledge Base Agent** 
- Ingest documents (PDF, DOCX, TXT) and text into Qdrant vector database
- Intelligent document chunking and embedding
- Semantic search and retrieval
- Web content ingestion from URLs
- Custom metadata support

### 🕷️ **Web Scraping Agent**
- Real-time web scraping with Playwright and requests
- Proxy support for enterprise scraping
- Docker-based secure execution
- Rate limiting and error handling
- Link and image extraction

### 🔍 **Web Search Agent**
- Multi-provider search (Brave, AVES APIs)
- News and academic search capabilities
- Automatic provider rotation
- Rate limiting and quota management

### 🐍 **Code Execution Agent**
- Secure Python and Bash execution in Docker containers
- Isolated environment with memory limits
- File handling and output capture
- Error handling and timeout management

### 🎯 **Intelligent Routing**
- Automatic agent selection based on content analysis
- Multi-turn conversation support
- Context preservation across sessions
- Message routing and delegation

## 📋 Prerequisites

### Required Dependencies
```bash
# Core dependencies
pip install redis docker-py
pip install langchain langchain-openai langchain-anthropic langchain-aws
pip install llama-index qdrant-client
pip install requests beautifulsoup4 playwright
pip install papaparse lz4 cachetools

# Media processing
pip install ffmpeg-python

# Optional: Install Playwright browsers
playwright install
```

### System Requirements
- **Docker**: Required for secure code execution and media processing
- **Redis**: Required for memory management and session persistence
- **FFmpeg**: Required for media processing (can be installed in Docker)
- **Python 3.8+**: Required runtime

### External Services
- **LLM Providers**: OpenAI, Anthropic, or AWS Bedrock API keys
- **Search APIs**: Brave Search API, AVES API (optional)
- **Vector Database**: Qdrant instance for knowledge base
- **Proxy Services**: ScraperAPI or similar (optional)

## ⚙️ Configuration

Create `agent_config.yaml` in your project root:

```yaml
# Redis Configuration
redis:
  host: "localhost"
  port: 6379
  db: 0
  password: null

# LLM Configuration
llm:
  preferred_provider: "openai"
  temperature: 0.7
  openai_api_key: "your-openai-api-key"
  anthropic_api_key: "your-anthropic-api-key"
  aws_access_key_id: "your-aws-key"
  aws_secret_access_key: "your-aws-secret"
  aws_region: "us-east-1"

# Knowledge Base Configuration
knowledge_base:
  qdrant_url: "http://localhost:6333"
  qdrant_api_key: null
  chunk_size: 1024
  chunk_overlap: 20
  similarity_top_k: 5
  default_collection_prefix: "kb"

# Web Scraping Configuration
web_scraping:
  proxy_enabled: false
  proxy_config:
    http_proxy: "http://username:password@proxy.example.com:8080"
  docker_image: "sgosain/amb-ubuntu-python-public-pod"
  timeout: 60
  rate_limit_seconds: 1.0
  max_links_per_page: 100
  max_images_per_page: 50
  default_headers:
    User-Agent: "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"

# Web Search Configuration
web_search:
  brave_api_key: "your-brave-api-key"
  avesapi_api_key: "your-aves-api-key"

# Media Processing Configuration
media_processing:
  docker_image: "sgosain/amb-ubuntu-python-public-pod"
  timeout: 300
  memory_limit: "2g"
  input_dir: "./media_input"
  output_dir: "./media_output"
  work_dir: "/opt/ambivo/work_dir"

# Docker Configuration
docker:
  images: ["sgosain/amb-ubuntu-python-public-pod"]
  timeout: 60
  memory_limit: "512m"
  work_dir: "/opt/ambivo/work_dir"

# Service Configuration
service:
  max_sessions: 100
  session_timeout: 3600
  log_level: "INFO"
  log_to_file: false

# Memory Management
memory_management:
  compression:
    enabled: true
    algorithm: "lz4"
    compression_level: 1
  cache:
    enabled: true
    max_size: 1000
    ttl_seconds: 300

# Agent Capabilities
capabilities:
  code_execution: true
  web_scraping: true
  knowledge_base: true
  web_search: true
  media_processing: true
  proxy: true
```

## 🚀 Quick Start

### Basic Usage

```python
import asyncio
from ambivo_agents.services import create_agent_service

async def main():
    # Create agent service
    agent_service = create_agent_service()
    
    # Create session
    session_id = agent_service.create_session()
    
    # Process message
    result = await agent_service.process_message(
        message="Hello! Can you help me extract audio from a video file?",
        session_id=session_id,
        user_id="user123"
    )
    
    print(f"Agent Response: {result['response']}")

if __name__ == "__main__":
    asyncio.run(main())
```

### Health Check

```python
from ambivo_agents.services import create_agent_service

# Check system health
agent_service = create_agent_service()
health = agent_service.health_check()

print(f"System Status: {health['service_available']}")
print(f"Redis: {health['redis_available']}")
print(f"LLM: {health['llm_service_available']}")
print(f"Available Agents: {health['available_agent_types']}")
```

## 📖 Detailed Examples

### 1. Media Audio Extraction
See `examples/media_audio_extraction.py` for complete working example.

### 2. Knowledge Base Operations  
See `examples/knowledge_base_operations.py` for text ingestion and querying.

### 3. Web Scraping Apartments.com
See `examples/apartments_scraping.py` for real estate data extraction.

## 🏗️ Architecture

```
ambivo_agents/
├── agents/           # Specialized agent implementations
│   ├── assistant.py      # General purpose assistant
│   ├── code_executor.py  # Code execution agent
│   ├── knowledge_base.py # Knowledge base operations
│   ├── media_editor.py   # Media processing agent
│   ├── web_scraper.py    # Web scraping agent
│   └── web_search.py     # Web search agent
├── core/             # Core framework components
│   ├── base.py          # Base classes and interfaces
│   ├── memory.py        # Redis memory management
│   └── llm.py           # Multi-provider LLM service
├── services/         # Service layer
│   ├── agent_service.py # Main service orchestrator
│   └── factory.py       # Agent factory and routing
├── executors/        # Execution engines
│   ├── docker_executor.py  # Docker code execution
│   └── media_executor.py   # FFmpeg media processing
└── config/           # Configuration management
    └── loader.py        # YAML configuration loader
```

## 🔧 Agent Types

| Agent Type | Purpose | Key Capabilities |
|------------|---------|------------------|
| **Assistant** | General conversation | Chat, Q&A, general assistance |
| **Code Executor** | Code execution | Python/Bash in Docker containers |
| **Knowledge Base** | Document operations | Ingest, search, retrieve documents |
| **Media Editor** | Media processing | Audio/video conversion, extraction |
| **Web Scraper** | Web data extraction | Scrape websites with proxy support |
| **Web Search** | Information retrieval | Search web, news, academic content |
| **Proxy** | Intelligent routing | Route messages to appropriate agents |

## 🛡️ Security Features

- **Docker Isolation**: All code execution in secure containers
- **Memory Limits**: Prevent resource exhaustion
- **Network Isolation**: Containers run without network access
- **Input Sanitization**: Safe handling of user inputs
- **Rate Limiting**: Prevent API abuse
- **Error Handling**: Graceful failure management

## 📊 Monitoring & Analytics

```python
# Get service statistics
stats = agent_service.get_service_stats()
print(f"Active Sessions: {stats['active_sessions']}")
print(f"Messages Processed: {stats['total_messages_processed']}")
print(f"Uptime: {stats['uptime_seconds']} seconds")

# Get session information
session_info = agent_service.get_session_info(session_id)
print(f"Message Count: {session_info['message_count']}")
print(f"Agent Types: {session_info['available_agents']}")
```

## 🔄 Multi-Provider LLM Support

The system automatically rotates between configured LLM providers:

- **OpenAI**: GPT-4, GPT-3.5-turbo
- **Anthropic**: Claude-3.5-Sonnet
- **AWS Bedrock**: Cohere Command, Titan

Provider rotation handles:
- Rate limiting and quota management
- Error recovery and fallback
- Performance optimization
- Cost management

## 🚨 Error Handling

```python
try:
    result = await agent_service.process_message(
        message="Process this data",
        session_id=session_id,
        user_id="user123"
    )
    
    if result['success']:
        print(f"Success: {result['response']}")
    else:
        print(f"Error: {result['error']}")
        
except Exception as e:
    print(f"System Error: {e}")
```

## 📝 Best Practices

### Session Management
- Use meaningful session IDs for tracking
- Clean up expired sessions regularly
- Monitor session memory usage

### Memory Management
- Use conversation IDs for context isolation
- Enable compression for large data
- Configure appropriate cache sizes

### Agent Selection
- Let the proxy agent handle routing automatically
- Use specific agents for specialized tasks
- Monitor agent performance metrics

### Resource Management
- Set appropriate Docker memory limits
- Configure reasonable timeouts
- Monitor Redis memory usage

## 🧪 Testing

Run the comprehensive test suite:

```bash
# Run specific examples
python examples/media_audio_extraction.py
python examples/knowledge_base_operations.py
python examples/apartments_scraping.py

# Run full system test
python examples/comprehensive_example.py
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## 📄 License

```
MIT License

Copyright (c) 2025 Hemant Gosain / Ambivo

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## 📞 Support

- **Email**: sgosain@ambivo.com
- **Company**: https://www.ambivo.com
- **Documentation**: Coming soon
- **Issues**: GitHub Issues

## 🗺️ Roadmap

- [ ] Advanced media processing capabilities
- [ ] Enhanced knowledge base features
- [ ] Additional LLM provider support
- [ ] Advanced analytics and reporting

---

**Built with 🛡️ by by the Ambivo Team***