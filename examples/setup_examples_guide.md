# Ambivo Agents - Setup and Examples Guide

This guide will help you get started with the Ambivo Agents library and run the provided examples.

## 🚀 Quick Setup

### 1. Install Dependencies

```bash
# Core dependencies
pip install redis docker-py
pip install langchain langchain-openai langchain-anthropic langchain-aws
pip install llama-index qdrant-client
pip install requests beautifulsoup4 playwright
pip install papaparse lz4 cachetools ffmpeg-python

# Install Playwright browsers
playwright install
```

### 2. Setup External Services

#### Redis Server
```bash
# Using Docker
docker run -d --name redis -p 6379:6379 redis:7-alpine

# Or install locally
# Ubuntu/Debian: sudo apt install redis-server
# macOS: brew install redis
```

#### Qdrant Vector Database
```bash
# Using Docker
docker run -d --name qdrant -p 6333:6333 qdrant/qdrant:latest
```

#### Docker for Code Execution
```bash
# Pull the required image
docker pull sgosain/amb-ubuntu-python-public-pod
```

### 3. Configuration

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

# Knowledge Base Configuration
knowledge_base:
  qdrant_url: "http://localhost:6333"
  qdrant_api_key: null
  chunk_size: 1024
  chunk_overlap: 20

# Web Scraping Configuration
web_scraping:
  proxy_enabled: false
  timeout: 60
  rate_limit_seconds: 1.0
  default_headers:
    User-Agent: "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"

# Media Processing Configuration
media_editor:
  docker_image: "sgosain/amb-ubuntu-python-public-pod"
  timeout: 300
  memory_limit: "2g"
  input_dir: "./media_input"
  output_dir: "./media_output"

# Docker Configuration
docker:
  images: ["sgosain/amb-ubuntu-python-public-pod"]
  timeout: 60
  memory_limit: "512m"

# Service Configuration
service:
  max_sessions: 100
  session_timeout: 3600
  log_level: "INFO"

# Agent Capabilities
capabilities:
  code_execution: true
  web_scraping: true
  knowledge_base: true
  web_search: false  # Set to true if you have search API keys
  media_editor: true
  proxy: true
```

### 4. API Keys

Get API keys and add them to your configuration:

- **OpenAI**: https://platform.openai.com/api-keys
- **Anthropic**: https://console.anthropic.com/
- **Brave Search** (optional): https://api.search.brave.com/
- **AVES API** (optional): https://avesapi.com/

## 📖 Running the Examples

### Example 1: Media Audio Extraction

Extracts audio from video files using FFmpeg.

```bash
# Run the full demo
python examples/media_audio_extraction.py

# Process a specific video file
python examples/media_audio_extraction.py --input-file /path/to/video.mp4 --output-format mp3 --quality high

# Create sample video and extract audio
python examples/media_audio_extraction.py
```

**What it demonstrates:**
- Audio extraction from video files
- Multiple format support (MP3, WAV, AAC, FLAC)
- Quality settings and bitrate control
- Batch processing capabilities
- Media file information retrieval
- Volume adjustment and audio conversion

**Prerequisites:**
- Docker running
- FFmpeg available in Docker container
- Video files in `media_input/` directory (or demo creates sample)

### Example 2: Knowledge Base Operations

Ingests documents and performs semantic search queries.

```bash
# Run the full demo
python examples/knowledge_base_operations.py

# Ingest a specific document
python examples/knowledge_base_operations.py --ingest-file /path/to/document.pdf

# Ingest web content
python examples/knowledge_base_operations.py --ingest-url https://example.com/article

# Query existing knowledge base
python examples/knowledge_base_operations.py --query-only "What are the company's core values?"

# Use custom knowledge base name
python examples/knowledge_base_operations.py --kb-name my_custom_kb
```

**What it demonstrates:**
- Document ingestion (PDF, DOCX, TXT)
- Text content ingestion with metadata
- Web content ingestion from URLs
- Semantic search and retrieval
- Knowledge base querying with different question types
- Source attribution and relevance scoring

**Prerequisites:**
- Qdrant running on localhost:6333
- LLM provider configured (OpenAI recommended)
- Documents in `demo_documents/` directory (or demo creates them)

### Example 3: Apartments.com Scraping

Scrapes real estate listings from apartments.com.

```bash
# Run the full demo
python examples/apartments_scraping.py

# Search specific city
python examples/apartments_scraping.py --city "Austin" --state "TX" --max-price 2000

# Scrape specific URL
python examples/apartments_scraping.py --url "https://www.apartments.com/san-francisco-ca/"

# Advanced search with criteria
python examples/apartments_scraping.py --city "Seattle" --state "WA" --max-price 3000 --min-bedrooms 2 --max-pages 3
```

**What it demonstrates:**
- Real-time web scraping with Playwright
- Dynamic content extraction from JavaScript-heavy sites
- Data parsing and structure extraction
- Rate limiting and ethical scraping practices
- Multi-city batch processing
- CSV and JSON data export
- URL accessibility checking

**Prerequisites:**
- Playwright browsers installed
- Web scraping enabled in configuration
- Good internet connection
- Consider proxy configuration for production use

## 🔍 Understanding the Output

### Media Processing Output
```
media_output/
├── extracted_audio_timestamp.mp3
├── converted_audio_timestamp.flac
├── volume_adjusted_timestamp.wav
└── sample_test_video.mp4
```

### Knowledge Base Output
```
demo_documents/
├── company_overview.txt
├── products_services.txt
├── technical_specifications.txt
└── deployment_guide.txt
```

### Web Scraping Output
```
scraped_apartments/
├── apartments_20250624_143022.json
├── apartments_20250624_143022.csv
├── filtered_San_Francisco_CA_timestamp.json
└── all_cities_consolidated_timestamp.json
```

## 🛠️ Troubleshooting

### Common Issues

#### 1. Redis Connection Error
```bash
# Check if Redis is running
docker ps | grep redis

# Start Redis if not running
docker start redis

# Or restart
docker restart redis
```

#### 2. Qdrant Connection Error
```bash
# Check Qdrant status
curl http://localhost:6333/health

# Restart Qdrant
docker restart qdrant
```

#### 3. Docker Permission Issues
```bash
# Add user to docker group (Linux)
sudo usermod -aG docker $USER
# Then logout/login

# Or run with sudo
sudo python examples/media_audio_extraction.py
```

#### 4. Missing API Keys
```yaml
# Update agent_config.yaml with your keys
llm:
  openai_api_key: "sk-your-actual-key-here"
  anthropic_api_key: "your-anthropic-key-here"
```

#### 5. Playwright Installation Issues
```bash
# Reinstall Playwright and browsers
pip uninstall playwright
pip install playwright
playwright install

# Install system dependencies (Linux)
playwright install-deps
```

### Web Scraping Considerations

1. **Rate Limiting**: apartments.com may block rapid requests
   - Increase delays between requests
   - Use proxy configuration
   - Consider residential proxies for production

2. **Content Changes**: Website structure may change
   - Update parsing logic if needed
   - Monitor for anti-bot measures

3. **Legal Compliance**: Always respect robots.txt and terms of service
   - Check `https://apartments.com/robots.txt`
   - Use data responsibly
   - Consider contacting site owners for permission

## 📊 Performance Tips

### Memory Optimization
```yaml
memory_management:
  compression:
    enabled: true
    algorithm: "lz4"
  cache:
    enabled: true
    max_size: 1000
    ttl_seconds: 300
```

### Docker Resource Limits
```yaml
docker:
  memory_limit: "1g"  # Increase if needed
  timeout: 120        # Increase for complex operations
```

### Concurrent Processing
```python
# Process multiple files concurrently
tasks = []
for file in files:
    task = agent_service.process_message(...)
    tasks.append(task)

results = await asyncio.gather(*tasks)
```

## 🔐 Security Best Practices

1. **Environment Variables**: Store API keys in environment variables
2. **Docker Security**: Run containers with limited privileges
3. **Network Security**: Use firewalls and VPNs for production
4. **Data Encryption**: Enable Redis AUTH and TLS
5. **Input Validation**: Sanitize all user inputs
6. **Monitoring**: Set up logging and alerting

## 📈 Scaling for Production

1. **Redis Cluster**: Use Redis cluster for high availability
2. **Load Balancing**: Deploy multiple agent service instances
3. **Monitoring**: Use Prometheus + Grafana
4. **Kubernetes**: Deploy with Kubernetes for orchestration
5. **Backup Strategy**: Regular backups of Redis and Qdrant data

## 🤝 Next Steps

1. **Customize Configuration**: Adapt settings for your use case
2. **Develop Custom Agents**: Create specialized agents for your domain
3. **Integration**: Integrate with your existing systems
4. **Monitoring**: Set up comprehensive monitoring and alerting
5. **Scaling**: Plan for horizontal scaling as usage grows

## 📞 Support

- **Email**: sgosain@ambivo.com
- **Company**: https://www.ambivo.com
- **Documentation**: Coming soon
- **GitHub Issues**: Report bugs and feature requests

---

**Happy automating with Ambivo Agents! 🚀**