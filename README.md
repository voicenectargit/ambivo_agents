# Ambivo Agents - Multi-Agent AI System

A comprehensive toolkit for AI-powered automation including media processing, knowledge base operations, web scraping, YouTube downloads, and more.

## 🚀 Features

### Core Capabilities
- **Multi-Agent Architecture**: Specialized agents for different tasks with intelligent routing
- **Docker-Based Execution**: Secure, isolated execution environment for code and media processing
- **Redis Memory Management**: Persistent conversation memory with compression and caching
- **Multi-Provider LLM Support**: Automatic failover between OpenAI, Anthropic, and AWS Bedrock
- **Configuration-Driven**: All features controlled via `agent_config.yaml`

### Available Agents

#### 🤖 Assistant Agent
- General purpose conversational AI
- Context-aware responses
- Multi-turn conversations

#### 💻 Code Executor Agent  
- Secure Python and Bash execution in Docker
- Isolated environment with resource limits
- Real-time output streaming

#### 🔍 Web Search Agent
- Multi-provider search (Brave, AVES APIs)
- News and academic search capabilities
- Automatic provider failover

#### 🕷️ Web Scraper Agent
- Proxy-enabled scraping (ScraperAPI compatible)
- Playwright and requests-based scraping
- Batch URL processing with rate limiting

#### 📚 Knowledge Base Agent
- Document ingestion (PDF, DOCX, TXT, web URLs)
- Vector similarity search with Qdrant
- Semantic question answering

#### 🎥 Media Editor Agent
- Audio/video processing with FFmpeg
- Format conversion, resizing, trimming
- Audio extraction and volume adjustment

#### 🎬 YouTube Download Agent *(New!)*
- Download videos and audio from YouTube
- Docker-based execution with pytubefix
- Automatic title sanitization and metadata extraction

## 📋 Prerequisites

### Required
- **Python 3.8+**
- **Docker** (for code execution, media processing, YouTube downloads)
- **Redis** (for memory management)

### API Keys (Optional - based on enabled features)
- **OpenAI API Key** (for GPT models)
- **Anthropic API Key** (for Claude models)
- **AWS Credentials** (for Bedrock models)
- **Brave Search API Key** (for web search)
- **AVES API Key** (for web search)
- **ScraperAPI/Proxy credentials** (for web scraping)

## 🛠️ Installation

### 1. Install Dependencies
```bash
# Core dependencies
pip install redis python-dotenv pyyaml click

# LLM providers (choose based on your needs)
pip install openai anthropic boto3 langchain-openai langchain-anthropic langchain-aws

# Knowledge base (if using)
pip install qdrant-client llama-index langchain-unstructured

# Web capabilities (if using)
pip install requests beautifulsoup4 playwright

# Media processing (if using) 
pip install docker

# YouTube downloads (if using)
pip install pytubefix pydantic

# Optional: Install all at once
pip install -r requirements.txt
```

### 2. Setup Docker Images
```bash
# Pull the multi-purpose container image
docker pull sgosain/amb-ubuntu-python-public-pod
```

### 3. Setup Redis
```bash
# Using Docker
docker run -d --name redis -p 6379:6379 redis:latest

# Or install locally
# sudo apt-get install redis-server  # Ubuntu/Debian
# brew install redis                 # macOS
```

## ⚙️ Configuration

Create `agent_config.yaml` in your project root:

```yaml
# Redis Configuration (Required)
redis:
  host: "localhost"
  port: 6379
  db: 0
  password: null  # Set if using Redis AUTH

# LLM Configuration (Required - at least one provider)
llm:
  preferred_provider: "openai"  # openai, anthropic, or bedrock
  temperature: 0.7
  
  # Provider API Keys
  openai_api_key: "your-openai-key"
  anthropic_api_key: "your-anthropic-key"
  
  # AWS Bedrock (optional)
  aws_access_key_id: "your-aws-key"
  aws_secret_access_key: "your-aws-secret"
  aws_region: "us-east-1"

# Agent Capabilities (Enable/disable features)
agent_capabilities:
  enable_web_search: true
  enable_web_scraping: true  
  enable_knowledge_base: true
  enable_media_editor: true
  enable_youtube_download: true
  enable_code_execution: true
  enable_proxy_mode: true

# Web Search Configuration (if enabled)
web_search:
  brave_api_key: "your-brave-api-key"
  avesapi_api_key: "your-aves-api-key"

# Web Scraping Configuration (if enabled)
web_scraping:
  proxy_enabled: true
  proxy_config:
    http_proxy: "http://scraperapi:your-key@proxy-server.scraperapi.com:8001"
  default_headers:
    User-Agent: "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
  timeout: 60
  max_links_per_page: 100

# Knowledge Base Configuration (if enabled)
knowledge_base:
  qdrant_url: "http://localhost:6333"
  qdrant_api_key: null  # Set if using Qdrant Cloud
  chunk_size: 1024
  chunk_overlap: 20
  similarity_top_k: 5

# Media Editor Configuration (if enabled)
media_editor:
  docker_image: "sgosain/amb-ubuntu-python-public-pod"
  input_dir: "./media_input"
  output_dir: "./media_output" 
  timeout: 300
  memory_limit: "2g"

# YouTube Download Configuration (if enabled)
youtube_download:
  docker_image: "sgosain/amb-ubuntu-python-public-pod"
  download_dir: "./youtube_downloads"
  timeout: 600
  memory_limit: "1g"
  default_audio_only: true

# Docker Configuration
docker:
  timeout: 60
  memory_limit: "512m"
  images: ["sgosain/amb-ubuntu-python-public-pod"]

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
  cache:
    enabled: true
    max_size: 1000
    ttl_seconds: 300
```

## 🚀 Quick Start

### Command Line Interface

```bash
# Install the CLI
pip install ambivo-agents

# Health check
ambivo-agents health

# Interactive chat mode
ambivo-agents interactive

# Single message
ambivo-agents chat "Hello, how can you help me?"

# Media processing
ambivo-agents media extract-audio input.mp4 --output-format mp3

# Knowledge base operations  
ambivo-agents kb ingest-file document.pdf --kb-name my_docs

# Web scraping
ambivo-agents scrape url https://example.com --output results.json
```

### Python API

```python
from ambivo_agents.services import create_agent_service
import asyncio

async def main():
    # Create agent service
    service = create_agent_service()
    
    # Create session
    session_id = service.create_session()
    
    # Process message
    result = await service.process_message(
        message="Download the audio from https://youtube.com/watch?v=example",
        session_id=session_id,
        user_id="user123"
    )
    
    print(result['response'])

# Run
asyncio.run(main())
```

## 📖 Usage Examples

### YouTube Downloads
```bash
# Download audio from YouTube
ambivo-agents chat "Download audio from https://youtube.com/watch?v=dQw4w9WgXcQ"

# Download video 
ambivo-agents chat "Download video from https://youtube.com/watch?v=dQw4w9WgXcQ in highest quality"
```

### Media Processing
```bash
# Extract audio from video
ambivo-agents chat "Extract audio from /path/to/video.mp4 as high quality mp3"

# Convert video format
ambivo-agents chat "Convert /path/to/video.avi to mp4 with h264 codec"

# Create thumbnail
ambivo-agents chat "Create thumbnail from /path/to/video.mp4 at 00:05:00"
```

### Knowledge Base Operations
```bash
# Ingest documents
ambivo-agents chat "Ingest document /path/to/manual.pdf into knowledge base 'company_docs'"

# Query knowledge base
ambivo-agents chat "Query knowledge base 'company_docs': What is the return policy?"
```

### Web Search & Scraping
```bash
# Web search
ambivo-agents chat "search for latest AI trends 2024"

# Web scraping
ambivo-agents chat "scrape https://example.com and extract all links"
```

### Code Execution
```bash
# Python code
ambivo-agents chat "```python\nprint('Hello World')\nimport math\nprint(math.pi)\n```"

# Bash commands  
ambivo-agents chat "```bash\nls -la\ndf -h\n```"
```

## 🔧 Architecture

### Agent Routing
The system uses a **Proxy Agent** that intelligently routes messages to specialized agents based on content analysis:

- **YouTube keywords** → YouTube Download Agent
- **Media keywords** → Media Editor Agent  
- **Search keywords** → Web Search Agent
- **Scraping keywords** → Web Scraper Agent
- **Knowledge keywords** → Knowledge Base Agent
- **Code blocks** → Code Executor Agent
- **General queries** → Assistant Agent

### Memory System
- **Redis-based persistence** with compression and caching
- **Conversation-aware context** with TTL management
- **Multi-session support** with automatic cleanup

### LLM Provider Management
- **Automatic failover** between OpenAI, Anthropic, AWS Bedrock
- **Rate limiting** and error handling
- **Provider rotation** based on availability and performance

## 🐳 Docker Setup

### Custom Docker Image
If you need additional dependencies, extend the base image:

```dockerfile
FROM sgosain/amb-ubuntu-python-public-pod

# Install additional packages
RUN pip install your-additional-packages

# Add custom scripts
COPY your-scripts/ /opt/scripts/
```

### Volume Mounting
The agents automatically handle volume mounting for:
- Media input/output directories
- YouTube download directories  
- Code execution workspaces

## 🔍 Troubleshooting

### Common Issues

1. **Redis Connection Failed**
   ```bash
   # Check Redis is running
   redis-cli ping
   # Should return "PONG"
   ```

2. **Docker Not Available**
   ```bash
   # Check Docker is running
   docker ps
   # Install if missing: https://docs.docker.com/get-docker/
   ```

3. **Agent Not Found Errors**
   - Verify the feature is enabled in `agent_capabilities`
   - Check that required configuration sections exist
   - Ensure API keys are properly set

4. **Module Import Errors**
   ```bash
   # Install missing dependencies
   pip install missing-package
   ```

### Debug Mode
Enable verbose logging:
```yaml
service:
  log_level: "DEBUG"
  log_to_file: true
```

## 🔐 Security Considerations

- **Docker Isolation**: All code execution happens in isolated containers
- **Network Restrictions**: Containers run with `network_disabled=True` by default
- **Resource Limits**: Memory and CPU limits prevent resource exhaustion  
- **API Key Management**: Store sensitive keys in environment variables
- **Input Sanitization**: All user inputs are validated and sanitized

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Development Setup
```bash
# Clone repository
git clone https://github.com/ambivo/ambivo-agents.git
cd ambivo-agents

# Install in development mode
pip install -e .

# Run tests
pytest tests/

# Type checking
mypy ambivo_agents/
```

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

## 👨‍💻 Author

**Hemant Gosain 'Sunny'**
- Company: [Ambivo](https://www.ambivo.com)
- Email: sgosain@ambivo.com

## 🆘 Support

- 📧 Email: sgosain@ambivo.com
- 🌐 Website: https://www.ambivo.com
- 📖 Documentation: [Coming Soon]
- 🐛 Issues: [GitHub Issues](https://github.com/ambivo/ambivo-agents/issues)

---

*Built with 🛡 by the Ambivo team*