#!/usr/bin/env python3
"""
Ambivo Agents CLI Interface - Updated with  .create() Paradigm

Author: Hemant Gosain 'Sunny'
Company: Ambivo
Email: sgosain@ambivo.com
License: MIT
"""

import asyncio
import click
import json
import sys
import time
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any

# 🌟 : Import agents directly using clean imports
from ambivo_agents import (
    KnowledgeBaseAgent,
    YouTubeDownloadAgent,
    MediaEditorAgent,
    WebSearchAgent,
    WebScraperAgent,
    CodeExecutorAgent,
    AgentSession
)

# Fallback to service for complex routing if needed
try:
    from ambivo_agents.services import create_agent_service

    SERVICE_AVAILABLE = True
except ImportError:
    SERVICE_AVAILABLE = False


class AmbivoAgentsCLI:
    """Command-line interface for Ambivo Agents with  .create() paradigm"""

    def __init__(self):
        self.user_id = "cli_user"
        self.tenant_id = "cli_tenant"
        self.session_metadata = {"cli_session": True, "version": "1.3.0"}

    async def create_agent(self, agent_class, additional_metadata: Dict[str, Any] = None):
        """Create agent using the  .create() pattern"""
        metadata = {**self.session_metadata}
        if additional_metadata:
            metadata.update(additional_metadata)

        agent, context = agent_class.create(
            user_id=self.user_id,
            tenant_id=self.tenant_id,
            session_metadata=metadata
        )

        click.echo(f"✅ Created {agent_class.__name__} (Session: {context.session_id[:8]}...)")
        return agent, context

    async def smart_message_routing(self, message: str) -> str:
        """
        Smart routing to appropriate agent based on message content
        Uses the  direct agent approach
        """
        message_lower = message.lower()

        # 🎬 YouTube Download Detection
        if any(keyword in message_lower for keyword in ['youtube', 'download', 'youtu.be']) and (
                'http' in message or 'www.' in message):
            agent, context = await self.create_agent(YouTubeDownloadAgent, {"operation": "youtube_download"})

            try:
                # Extract YouTube URLs
                import re
                youtube_patterns = [
                    r'https?://(?:www\.)?youtube\.com/watch\?v=[\w-]+',
                    r'https?://(?:www\.)?youtu\.be/[\w-]+',
                ]

                urls = []
                for pattern in youtube_patterns:
                    urls.extend(re.findall(pattern, message))

                if urls:
                    url = urls[0]  # Use first URL found

                    # Determine if user wants audio or video
                    wants_video = any(keyword in message_lower for keyword in ['video', 'mp4', 'watch', 'visual'])
                    audio_only = not wants_video

                    if 'info' in message_lower or 'information' in message_lower:
                        result = await agent._get_youtube_info(url)
                    else:
                        result = await agent._download_youtube(url, audio_only=audio_only)

                    await agent.cleanup_session()

                    if result['success']:
                        return f"✅ YouTube operation completed!\n{result.get('message', '')}\nSession: {context.session_id}"
                    else:
                        return f"❌ YouTube operation failed: {result['error']}"
                else:
                    await agent.cleanup_session()
                    return "❌ No valid YouTube URLs found in message"

            except Exception as e:
                await agent.cleanup_session()
                return f"❌ YouTube operation error: {e}"

        # 🎵 Media Processing Detection
        elif any(keyword in message_lower for keyword in
                 ['extract audio', 'convert video', 'media', 'ffmpeg', '.mp4', '.avi', '.mov']):
            agent, context = await self.create_agent(MediaEditorAgent, {"operation": "media_processing"})

            try:
                # Simple routing based on keywords
                if 'extract audio' in message_lower:
                    return "🎵 Media Editor Agent ready for audio extraction!\nPlease provide the input file path for processing."
                elif 'convert' in message_lower:
                    return "🎥 Media Editor Agent ready for video conversion!\nPlease provide the input file path and target format."
                else:
                    return "🎬 Media Editor Agent ready!\nI can extract audio, convert videos, resize, trim, and more."

            finally:
                await agent.cleanup_session()

        # 📚 Knowledge Base Detection
        elif any(keyword in message_lower for keyword in ['knowledge base', 'ingest', 'query', 'document', 'kb ']):
            agent, context = await self.create_agent(KnowledgeBaseAgent, {"operation": "knowledge_base"})

            try:
                if 'ingest' in message_lower:
                    return "📄 Knowledge Base Agent ready for document ingestion!\nPlease provide the document path and knowledge base name."
                elif 'query' in message_lower:
                    return "🔍 Knowledge Base Agent ready for queries!\nPlease provide your question and knowledge base name."
                else:
                    return "📚 Knowledge Base Agent ready!\nI can ingest documents and answer questions based on your knowledge bases."

            finally:
                await agent.cleanup_session()

        # 🔍 Web Search Detection
        elif any(keyword in message_lower for keyword in ['search', 'find', 'look up', 'google']):
            agent, context = await self.create_agent(WebSearchAgent, {"operation": "web_search"})

            try:
                # Extract search query
                search_query = message
                for prefix in ['search for', 'find', 'look up', 'google']:
                    if prefix in message_lower:
                        search_query = message[message_lower.find(prefix) + len(prefix):].strip()
                        break

                result = await agent._search_web(search_query, max_results=5)
                await agent.cleanup_session()

                if result['success']:
                    response = f"🔍 Search Results for '{search_query}':\n\n"
                    for i, res in enumerate(result['results'][:3], 1):
                        response += f"{i}. **{res.get('title', 'No title')}**\n"
                        response += f"   {res.get('url', 'No URL')}\n"
                        response += f"   {res.get('snippet', 'No snippet')[:150]}...\n\n"
                    response += f"Session: {context.session_id}"
                    return response
                else:
                    return f"❌ Search failed: {result['error']}"

            except Exception as e:
                await agent.cleanup_session()
                return f"❌ Search error: {e}"

        # 🕷️ Web Scraping Detection
        elif any(keyword in message_lower for keyword in ['scrape', 'extract', 'crawl']) and (
                'http' in message or 'www.' in message):
            agent, context = await self.create_agent(WebScraperAgent, {"operation": "web_scraping"})

            try:
                # Extract URLs
                import re
                url_pattern = r'https?://[^\s]+'
                urls = re.findall(url_pattern, message)

                if urls:
                    url = urls[0]
                    return f"🕷️ Web Scraper Agent ready to scrape: {url}\nSession: {context.session_id}\n(Note: Actual scraping requires proper configuration)"
                else:
                    return "❌ No valid URLs found for scraping"

            finally:
                await agent.cleanup_session()

        # 💻 Code Execution Detection
        elif '```' in message:
            agent, context = await self.create_agent(CodeExecutorAgent, {"operation": "code_execution"})

            try:
                return f"💻 Code Executor Agent ready!\nSession: {context.session_id}\n(Note: Code execution requires Docker configuration)"

            finally:
                await agent.cleanup_session()

        # 🤖 General Assistant (fallback)
        else:
            # For general queries, provide helpful guidance
            return f"""🤖 Ambivo Agents CLI - How can I help?

I can assist with:

🎬 **YouTube Downloads**
   'download audio from https://youtube.com/watch?v=example'
   'download video from https://youtube.com/watch?v=example'

🎵 **Media Processing**
   'extract audio from video.mp4'
   'convert video.avi to mp4'

📚 **Knowledge Base**
   'ingest document file.pdf into knowledge base'
   'query knowledge base: what is our policy?'

🔍 **Web Search**
   'search for artificial intelligence trends'
   'find latest s about space exploration'

🕷️ **Web Scraping**
   'scrape https://example.com'

💻 **Code Execution**
   ```python
   print('Hello World')
   ```

Type 'ambivo-agents interactive' for interactive mode!
"""


# Initialize CLI instance
cli_instance = AmbivoAgentsCLI()


@click.group()
@click.version_option(version="1.3.0", prog_name="Ambivo Agents")
@click.option('--config', '-c', help='Configuration file path')
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose output')
def cli(config: Optional[str], verbose: bool):
    """
    Ambivo Agents - Multi-Agent AI System CLI

    🌟 : Uses the .create() paradigm for direct agent creation
    Each agent is created with explicit context management.

    Features:
    - YouTube Downloads with pytubefix
    - Media Processing with FFmpeg
    - Knowledge Base Operations with Qdrant
    - Web Search with multiple providers
    - Web Scraping with proxy support
    - Code Execution in Docker containers

    Author: Hemant Gosain 'Sunny'
    Company: Ambivo
    Email: sgosain@ambivo.com
    """
    if verbose:
        click.echo("🤖 Ambivo Agents CLI v1.3.0 -  .create() Paradigm")
        click.echo("📧 Contact: sgosain@ambivo.com")
        click.echo("🏢 Company: https://www.ambivo.com")
        click.echo("🌟 : Direct agent creation with explicit context")

    if config:
        click.echo(f"📋 Using config file: {config}")


@cli.command()
def health():
    """Check system health using direct agent creation"""
    click.echo("🏥 Checking Ambivo Agents Health ( .create() approach)...")

    agents_to_test = [
        ("YouTube Download", YouTubeDownloadAgent),
        ("Media Editor", MediaEditorAgent),
        ("Knowledge Base", KnowledgeBaseAgent),
        ("Web Search", WebSearchAgent),
        ("Web Scraper", WebScraperAgent),
        ("Code Executor", CodeExecutorAgent),
    ]

    async def health_check():
        results = {}

        for agent_name, agent_class in agents_to_test:
            try:
                agent, context = await cli_instance.create_agent(agent_class)
                await agent.cleanup_session()
                results[agent_name] = "✅ Available"
            except Exception as e:
                results[agent_name] = f"❌ Error: {str(e)[:50]}..."

        click.echo("\n📊 Agent Health Status:")
        for agent_name, status in results.items():
            click.echo(f"  {agent_name}: {status}")

        available_count = len([s for s in results.values() if "✅" in s])
        total_count = len(results)

        click.echo(f"\n📈 Summary: {available_count}/{total_count} agents available")

        if available_count == 0:
            click.echo("❌ No agents available - check configuration")
            sys.exit(1)
        elif available_count < total_count:
            click.echo("⚠️  Some agents unavailable - partial functionality")
        else:
            click.echo("🎉 All agents healthy!")

    asyncio.run(health_check())


@cli.command()
@click.argument('message')
@click.option('--format', '-f', type=click.Choice(['text', 'json']), default='text', help='Output format')
def chat(message: str, format: str):
    """Send a message using smart agent routing ( .create() approach)"""
    click.echo(f"💬 Processing: {message}")

    async def process():
        start_time = time.time()
        response = await cli_instance.smart_message_routing(message)
        processing_time = time.time() - start_time

        if format == 'json':
            result = {
                'success': True,
                'response': response,
                'processing_time': processing_time,
                'message': message,
                'paradigm': 'direct_agent_creation'
            }
            click.echo(json.dumps(result, indent=2))
        else:
            click.echo(f"\n🤖 Response:\n{response}")
            click.echo(f"\n⏱️  Processing time: {processing_time:.2f}s")
            click.echo(f"🌟 Using  .create() paradigm")

    asyncio.run(process())


@cli.group()
def youtube():
    """YouTube download commands using direct agent creation"""
    pass


@youtube.command()
@click.argument('url')
@click.option('--audio-only', '-a', is_flag=True, default=True, help='Download audio only (default)')
@click.option('--video', '-v', is_flag=True, help='Download video (overrides --audio-only)')
@click.option('--output-name', '-n', help='Custom output filename (without extension)')
def download(url: str, audio_only: bool, video: bool, output_name: Optional[str]):
    """Download from YouTube using direct YouTubeDownloadAgent.create()"""

    if video:
        audio_only = False

    content_type = "video" if not audio_only else "audio"
    click.echo(f"🎬 Downloading {content_type} from: {url}")

    async def download_process():
        try:
            # 🌟 : Direct agent creation
            agent, context = await cli_instance.create_agent(
                YouTubeDownloadAgent,
                {"operation": "youtube_download", "url": url}
            )

            click.echo(f"📋 Session ID: {context.session_id}")
            click.echo(f"👤 User: {context.user_id}")

            # Use the agent directly
            if audio_only:
                result = await agent._download_youtube_audio(url, output_name)
            else:
                result = await agent._download_youtube_video(url, output_name)

            if result['success']:
                click.echo(f"✅ Download successful!")
                click.echo(f"📁 File: {result.get('filename', 'Unknown')}")
                click.echo(f"📍 Path: {result.get('file_path', 'Unknown')}")
                click.echo(f"📊 Size: {result.get('file_size_bytes', 0):,} bytes")
                click.echo(f"⏱️  Time: {result.get('execution_time', 0):.2f}s")
            else:
                click.echo(f"❌ Download failed: {result['error']}")

            # Cleanup
            await agent.cleanup_session()
            click.echo(f"🧹 Session cleaned up")

        except Exception as e:
            click.echo(f"❌ Error: {e}")

    asyncio.run(download_process())


@youtube.command()
@click.argument('url')
def info(url: str):
    """Get YouTube video info using direct agent creation"""
    click.echo(f"📹 Getting video info: {url}")

    async def info_process():
        try:
            # 🌟 : Direct agent creation
            agent, context = await cli_instance.create_agent(
                YouTubeDownloadAgent,
                {"operation": "youtube_info", "url": url}
            )

            result = await agent._get_youtube_info(url)

            if result['success']:
                video_info = result['video_info']
                click.echo(f"✅ Video Information:")
                click.echo(f"📹 Title: {video_info.get('title', 'Unknown')}")
                click.echo(f"👤 Author: {video_info.get('author', 'Unknown')}")
                click.echo(f"⏱️  Duration: {video_info.get('duration', 0)} seconds")
                click.echo(f"👀 Views: {video_info.get('views', 0):,}")
                click.echo(f"📋 Session: {context.session_id}")
            else:
                click.echo(f"❌ Failed to get video info: {result['error']}")

            await agent.cleanup_session()

        except Exception as e:
            click.echo(f"❌ Error: {e}")

    asyncio.run(info_process())


@cli.command()
def interactive():
    """Interactive chat mode using the  .create() paradigm"""
    click.echo("🤖 Ambivo Agents Interactive Mode ( .create() Paradigm)")
    click.echo("🌟 Each message creates agents directly with explicit context")
    click.echo("Type 'quit', 'exit', or 'bye' to exit")
    click.echo("Type 'help' for available commands")
    click.echo("-" * 60)

    async def interactive_loop():
        conversation_history = []

        while True:
            try:
                user_input = click.prompt("\n🗣️  You", type=str)

                if user_input.lower() in ['quit', 'exit', 'bye']:
                    click.echo("👋 Goodbye!")
                    break
                elif user_input.lower() == 'help':
                    click.echo("""
🌟 Ambivo Agents -  .create() Paradigm Help

📝 **Direct Agent Creation Examples:**

🎬 **YouTube Downloads:**
   'download audio from https://youtube.com/watch?v=example'
   'download video from https://youtube.com/watch?v=example'
   'get info about https://youtube.com/watch?v=example'

🎵 **Media Processing:**
   'extract audio from video.mp4'
   'convert video.avi to mp4'

📚 **Knowledge Base:**
   'ingest document file.pdf into knowledge base'
   'query knowledge base: what is our policy?'

🔍 **Web Search:**
   'search for artificial intelligence trends'
   'find latest s about space exploration'

🕷️ **Web Scraping:**
   'scrape https://example.com'

💻 **Code Execution:**
   ```python
   print('Hello World')
   import datetime
   print(datetime.datetime.now())
   ```

🌟 **Key Features:**
   - Each operation creates agents directly with .create()
   - Explicit context management with session IDs
   - Automatic cleanup after operations
   - Direct agent-to-tool communication
                    """)
                    continue

                # Store user input
                conversation_history.append(f"User: {user_input}")

                # Process with smart routing
                response = await cli_instance.smart_message_routing(user_input)
                conversation_history.append(f"Agent: {response[:100]}...")

                click.echo(f"🤖 Agent: {response}")
                click.echo(f"💡 Used  .create() paradigm for direct agent creation")

            except KeyboardInterrupt:
                click.echo("\n👋 Goodbye!")
                break
            except EOFError:
                click.echo("\n👋 Goodbye!")
                break

    asyncio.run(interactive_loop())


@cli.command()
def demo():
    """Demo showcasing the  .create() paradigm"""
    click.echo("🎪 Ambivo Agents Demo -  .create() Paradigm")
    click.echo("=" * 60)
    click.echo("🌟 This demo shows direct agent creation with explicit context")

    demos = [
        ("🎬 YouTube Agent", "get info about https://youtube.com/watch?v=dQw4w9WgXcQ"),
        ("🔍 Search Agent", "search for latest AI developments"),
        ("🤖 General Routing", "How can you help me with automation tasks?"),
    ]

    async def run_demos():
        for title, demo_message in demos:
            click.echo(f"\n{title}")
            click.echo("-" * 40)
            click.echo(f"Input: {demo_message}")
            click.echo("🌟 Using .create() paradigm...")

            try:
                start_time = time.time()
                response = await cli_instance.smart_message_routing(demo_message)
                processing_time = time.time() - start_time

                # Truncate long responses for demo
                if len(response) > 300:
                    response = response[:300] + "..."

                click.echo(f"Output: {response}")
                click.echo(f"Time: {processing_time:.2f}s")
                click.echo(f"✅ Agent created and cleaned up automatically")

            except Exception as e:
                click.echo(f"Demo error: {e}")

            # Pause between demos
            time.sleep(1)

    asyncio.run(run_demos())

    click.echo(f"\n🎉 Demo completed!")
    click.echo(f"🌟 All operations used the  .create() paradigm")
    click.echo(f"💡 Each agent was created directly with explicit context")
    click.echo(f"🧹 All sessions were automatically cleaned up")


@cli.command()
def examples():
    """Show usage examples for the  .create() paradigm"""
    click.echo("📚 Ambivo Agents -  .create() Paradigm Examples")
    click.echo("=" * 60)

    click.echo(f"""
🌟 ** PARADIGM: Direct Agent Creation**

The  approach creates agents directly with explicit context:

```python
# 🌟  WAY: Direct agent creation
from ambivo_agents import YouTubeDownloadAgent

agent, context = YouTubeDownloadAgent.create(user_id="john")
print(f"Session: {{context.session_id}}")
print(f"User: {{context.user_id}}")

result = await agent._download_youtube_audio(url)
await agent.cleanup_session()
```

vs. service-based approach:

```python
# Service-based
service = create_agent_service()
result = await service.process_message(...)
```

🎯 **CLI Examples using  paradigm:**

🎬 **YouTube Downloads:**
   ambivo-agents youtube download 'https://youtube.com/watch?v=example'
   ambivo-agents youtube info 'https://youtube.com/watch?v=example'

💬 **Smart Chat Routing:**
   ambivo-agents chat "download audio from https://youtube.com/watch?v=example"
   ambivo-agents chat "search for artificial intelligence trends"
   ambivo-agents chat "extract audio from video.mp4"

🔄 **Interactive Mode:**
   ambivo-agents interactive

🏥 **Health Check:**
   ambivo-agents health

🎪 **Demo:**
   ambivo-agents demo

💡 **Key Benefits:**
   ✅ Explicit context management
   ✅ Direct agent-to-tool communication  
   ✅ Clear session lifecycle
   ✅ Better error handling
   ✅ More predictable behavior
""")


if __name__ == '__main__':
    cli()