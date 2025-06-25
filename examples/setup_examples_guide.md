# Ambivo Agents - Setup and Examples Guide (Updated)

This guide reflects the latest fixes for consistent capability checking and routing.

## 🚀 Quick Setup

### 1. Install Dependencies

```bash
# Core dependencies
pip install redis docker-py
pip install langchain langchain-openai langchain-anthropic langchain-aws
pip install llama-index qdrant-client
pip install requests beautifulsoup4 playwright
pip install papaparse lz4 cachetools

# Media processing
pip install ffmpeg-python

# Install Playwright browsers
playwright install
```

### 2. Setup External Services

#### Redis Server
```bash
# Using Docker (recommended)
docker run -d --name redis -p 6379:6379 redis:7-alpine

# Or install locally
# Ubuntu/Debian: sudo apt install redis-server
# macOS: brew install redis
```

#### Qdrant Vector Database (for Knowledge Base)
```bash
# Using Docker
docker run -d --name qdrant -p 6333:6333 qdrant/qdrant:latest

# Or use Qdrant Cloud (recommended for production)
# Sign up at: https://cloud.qdrant.io/
```

#### Docker for Secure Execution
```bash
# Pull the required image for code/media processing
docker pull sgosain/amb-ubuntu-python-public-pod
```

### 3. Updated Configuration (agent_config.yaml)

**⚠️ Important: Use consistent capability names**

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
  # Optional AWS Bedrock
  aws_access_key_id: "your-aws-key"
  aws_secret_access_key: "your-aws-secret"
  aws_region: "us-east-1"

# Knowledge Base Configuration
knowledge_base:
  qdrant_url: "http://localhost:6333"
  qdrant_api_key: null  # Set if using Qdrant Cloud
  chunk_size: 1024
  chunk_overlap: 20
  similarity_top_k: 5
  default_collection_prefix: "kb"

# Web Search Configuration
web_search:
  brave_api_key: "your-brave-api-key"
  avesapi_api_key: "your-aves-api-key"
  default_max_results: 10
  search_timeout_seconds: 15

# Web Scraping Configuration
web_scraping:
  proxy_enabled: false
  proxy_config:
    http_proxy: "http://username:password@proxy.example.com:8080"
  docker_image: "sgosain/amb-ubuntu-python-public-pod"
  timeout: 60
  rate_limit_seconds: 2.0
  max_links_per_page: 100
  default_headers:
    User-Agent: "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"

# Media Processing Configuration - UPDATED NAME
media_editor:
  docker_image: "sgosain/amb-ubuntu-python-public-pod"
  timeout: 300
  memory_limit: "2g"
  input_dir: "./examples/media_input"      # Updated path
  output_dir: "./examples/media_output"    # Updated path
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
  cache:
    enabled: true
    max_size: 1000
    ttl_seconds: 300

# Agent Capabilities - UPDATED CONSISTENT NAMES
agent_capabilities:
  enable_code_execution: true
  enable_web_scraping: true
  enable_knowledge_base: true
  enable_web_search: true
  enable_media_editor: true        # Updated name
  enable_proxy_mode: true
```

### 4. API Keys Setup

Get API keys and add them to your configuration:

- **OpenAI**: https://platform.openai.com/api-keys
- **Anthropic**: https://console.anthropic.com/
- **Brave Search**: https://api.search.brave.com/
- **AVES API**: https://avesapi.com/
- **Qdrant Cloud**: https://cloud.qdrant.io/ (optional)

## 🎯 Creating Targeted Agents

### Simple Targeted Agent Example

Here's how to create a focused, single-purpose agent:

```python
#!/usr/bin/env python3
"""
Simple targeted agent example - Media Audio Extractor
Focused on one specific task with clear functionality
"""

import asyncio
from pathlib import Path
from ambivo_agents.core.memory import create_redis_memory_manager
from ambivo_agents.core.llm import create_multi_provider_llm_service
from ambivo_agents.agents.media_editor import MediaEditorAgent
from ambivo_agents.config.loader import load_config, get_config_section

class SimpleAudioExtractor:
    """Targeted agent wrapper for audio extraction only"""
    
    def __init__(self):
        # Load configuration
        config = load_config()
        redis_config = get_config_section('redis', config)
        llm_config = get_config_section('llm', config)
        
        # Create components
        memory = create_redis_memory_manager("audio_extractor", redis_config)
        llm_service = create_multi_provider_llm_service(llm_config)
        
        # Create targeted media agent
        self.media_agent = MediaEditorAgent(
            agent_id="simple_audio_extractor",
            memory_manager=memory,
            llm_service=llm_service
        )
        
        print("🎵 Simple Audio Extractor ready!")
    
    async def extract_audio(self, video_file: str, output_format: str = "mp3"):
        """Extract audio from video - single focused function"""
        
        print(f"🎬 Extracting audio from: {video_file}")
        print(f"🎵 Output format: {output_format}")
        
        try:
            # Call the specific tool directly
            result = await self.media_agent.execute_tool(
                "extract_audio_from_video",
                {
                    "input_video": video_file,
                    "output_format": output_format,
                    "audio_quality": "high"
                }
            )
            
            if result.get('success'):
                tool_result = result.get('result', {})
                if tool_result.get('success'):
                    print(f"✅ Audio extraction completed!")
                    print(f"📁 Output: {tool_result.get('output_file', {})}")
                    return tool_result
                else:
                    print(f"❌ Extraction failed: {tool_result.get('error')}")
                    return None
            else:
                print(f"❌ Tool execution failed: {result.get('error')}")
                return None
                
        except Exception as e:
            print(f"❌ Exception: {e}")
            return None

# Usage example
async def main():
    extractor = SimpleAudioExtractor()
    
    # Extract audio from a video file
    result = await extractor.extract_audio(
        video_file="examples/media_input/test_video.mov",
        output_format="mp3"
    )
    
    if result:
        print(f"🎉 Success! Audio extracted to: {result.get('final_path')}")
    else:
        print("❌ Extraction failed")

if __name__ == "__main__":
    asyncio.run(main())
```

### Another Targeted Agent - Simple Knowledge Base Query

```python
#!/usr/bin/env python3
"""
Simple Knowledge Base Query Agent
Focused only on querying existing knowledge bases
"""

import asyncio
from ambivo_agents.core.memory import create_redis_memory_manager
from ambivo_agents.core.llm import create_multi_provider_llm_service
from ambivo_agents.agents.knowledge_base import KnowledgeBaseAgent
from ambivo_agents.config.loader import load_config, get_config_section

class SimpleKBQuery:
    """Targeted agent for knowledge base queries only"""
    
    def __init__(self, kb_name: str = "default"):
        config = load_config()
        redis_config = get_config_section('redis', config)
        llm_config = get_config_section('llm', config)
        
        memory = create_redis_memory_manager("kb_query", redis_config)
        llm_service = create_multi_provider_llm_service(llm_config)
        
        self.kb_agent = KnowledgeBaseAgent(
            agent_id="simple_kb_query",
            memory_manager=memory,
            llm_service=llm_service
        )
        
        self.kb_name = kb_name
        print(f"🧠 Simple KB Query ready for: {kb_name}")
    
    async def ask_question(self, question: str):
        """Ask a question to the knowledge base"""
        
        print(f"❓ Question: {question}")
        print(f"📚 Searching knowledge base: {self.kb_name}")
        
        try:
            result = await self.kb_agent.execute_tool(
                "query_knowledge_base",
                {
                    "kb_name": self.kb_name,
                    "query": question,
                    "question_type": "free-text"
                }
            )
            
            if result.get('success'):
                answer_data = result.get('result', {})
                if answer_data.get('success'):
                    print(f"✅ Answer found!")
                    print(f"💬 Answer: {answer_data.get('answer')}")
                    
                    sources = answer_data.get('source_details', [])
                    if sources:
                        print(f"📋 Sources: {len(sources)} found")
                    
                    return answer_data.get('answer')
                else:
                    print(f"❌ Query failed: {answer_data.get('error')}")
                    return None
            else:
                print(f"❌ Tool failed: {result.get('error')}")
                return None
                
        except Exception as e:
            print(f"❌ Exception: {e}")
            return None

# Usage
async def main():
    kb = SimpleKBQuery("company_docs")
    
    questions = [
        "What is our company mission?",
        "How do I deploy the system?", 
        "What are the technical requirements?"
    ]
    
    for question in questions:
        answer = await kb.ask_question(question)
        print("-" * 50)

if __name__ == "__main__":
    asyncio.run(main())
```

## 📖 Updated Examples

### Fixed Media Audio Extraction

```bash
# Use the fixed version that properly detects .mov files
python examples/fixed_media_demo.py

# Test with specific file
python examples/fixed_media_demo.py --file test_video.mov --format mp3

# Get media info only
python examples/fixed_media_demo.py --info
```

### Simple Web Search (Provider-Specific)

```bash
# Use the simple search agent that shows provider info
python examples/simple_search_example.py --query "what is ambivo"

# Check provider status
python examples/simple_search_example.py --status

# Integrated with service
python examples/integrated_simple_search_example.py --ambivo
```

### Knowledge Base Operations

```bash
# Fixed routing ensures KB messages go to KnowledgeBaseAgent
python examples/knowledge_base_operations.py

# Ingest specific document
python examples/knowledge_base_operations.py --ingest-file document.pdf

# Query only
python examples/knowledge_base_operations.py --query-only "What is AI?"
```

## 🔧 Troubleshooting Updated Issues

### 1. Routing Problems (FIXED)

**Problem**: Messages going to wrong agents (e.g., web search going to Knowledge Base)

**Solution**: Use updated agent service with fixed routing:

```python
# Check agent availability with consistent naming
from ambivo_agents.services import create_agent_service

service = create_agent_service()
health = service.health_check()

print("Available Agents:", health['available_agent_types'])
print("Enabled Capabilities:", health['enabled_capabilities'])

# Should show consistent naming:
# {'web_search': True, 'media_editor': True, ...}
```

### 2. Capability Inconsistencies (FIXED)

**Problem**: Different parts of system reporting different capabilities

**Solution**: All components now use centralized capability checking:

```python
from ambivo_agents.config.loader import (
    validate_agent_capabilities,
    get_available_agent_types,
    get_enabled_capabilities
)

# All these return consistent results
capabilities = validate_agent_capabilities()
agent_types = get_available_agent_types()
enabled = get_enabled_capabilities()

print("Capabilities:", capabilities)
print("Agent Types:", agent_types)
print("Enabled:", enabled)
```

### 3. Media Processing Path Issues (FIXED)

**Problem**: Media files not found even when they exist

**Solution**: Use correct path configuration:

```yaml
# In agent_config.yaml - use examples subdirectory
media_editor:
  input_dir: "./examples/media_input"
  output_dir: "./examples/media_output"
```

### 4. Docker Execution Issues

**Problem**: Tools return help text instead of executing

**Solution**: Use direct tool execution:

```python
# Instead of message-based approach, call tools directly
result = await media_agent.execute_tool(
    "extract_audio_from_video",
    {"input_video": "path/to/video.mov", "output_format": "mp3"}
)
```

## 🎯 Best Practices for Targeted Agents

### 1. Single Responsibility Principle
```python
# Good - focused on one task
class AudioExtractor:
    async def extract_audio(self, video_file, format):
        # Only audio extraction logic

# Bad - multiple responsibilities  
class MediaProcessor:
    async def extract_audio(self): pass
    async def scrape_web(self): pass
    async def query_knowledge(self): pass
```

### 2. Direct Tool Access
```python
# Use execute_tool for direct access
result = await agent.execute_tool("tool_name", parameters)

# Instead of message routing (which can fail)
result = await agent_service.process_message("Please extract audio...")
```

### 3. Clear Error Handling
```python
try:
    result = await agent.execute_tool("extract_audio", params)
    if result.get('success'):
        tool_result = result.get('result', {})
        if tool_result.get('success'):
            return tool_result
        else:
            print(f"Tool failed: {tool_result.get('error')}")
    else:
        print(f"Execution failed: {result.get('error')}")
except Exception as e:
    print(f"Exception: {e}")
```

### 4. Configuration Validation
```python
# Always validate configuration at startup
from ambivo_agents.config.loader import validate_agent_capabilities

try:
    capabilities = validate_agent_capabilities()
    if not capabilities.get('media_editor'):
        raise ValueError("Media editor not enabled")
except Exception as e:
    print(f"Configuration error: {e}")
    exit(1)
```

## 🔍 Testing Your Setup

### Capability Consistency Test
```bash
python examples/test_capability_consistency.py
```

Expected output:
```
✅ ALL METHODS CONSISTENT
media_editor: ✅ ENABLED
web_search: ✅ AVAILABLE
```

### Agent Routing Test
```bash
python examples/test_routing_fix.py
```

Expected output:
```
✅ "search the web" → WebSearchAgent
✅ "extract audio" → MediaEditorAgent  
✅ "query knowledge base" → KnowledgeBaseAgent
```

## 🚀 Production Deployment

### Environment Variables
```bash
# Set API keys as environment variables
export OPENAI_API_KEY="your-key"
export ANTHROPIC_API_KEY="your-key"
export BRAVE_API_KEY="your-key"
export QDRANT_API_KEY="your-key"
```

