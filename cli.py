#!/usr/bin/env python3
"""
Ambivo Agents CLI Interface - Updated with YouTube Download Support

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
from typing import Optional

# Import from the package
from ambivo_agents.services import create_agent_service


class AmbivoAgentsCLI:
    """Command-line interface for Ambivo Agents"""

    def __init__(self):
        self.agent_service = None
        self.session_id = None
        self.user_id = "cli_user"

    def initialize_service(self):
        """Initialize the agent service"""
        if not self.agent_service:
            try:
                self.agent_service = create_agent_service()
                self.session_id = self.agent_service.create_session()
                click.echo(f"✅ Initialized Ambivo Agents (Session: {self.session_id[:8]}...)")
            except Exception as e:
                click.echo(f"❌ Failed to initialize Ambivo Agents: {e}", err=True)
                sys.exit(1)

    async def process_message(self, message: str, conversation_id: str = "cli") -> dict:
        """Process a message through the agent system"""
        self.initialize_service()

        return await self.agent_service.process_message(
            message=message,
            session_id=self.session_id,
            user_id=self.user_id,
            conversation_id=conversation_id
        )


# Initialize CLI instance
cli_instance = AmbivoAgentsCLI()


@click.group()
@click.version_option(version="1.3.0", prog_name="Ambivo Agents")
@click.option('--config', '-c', help='Configuration file path')
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose output')
def cli(config: Optional[str], verbose: bool):
    """
    Ambivo Agents - Multi-Agent AI System CLI

    A comprehensive toolkit for AI-powered automation including
    media processing, knowledge base operations, web scraping,
    YouTube downloads, and more.

    Author: Hemant Gosain 'Sunny'
    Company: Ambivo
    Email: sgosain@ambivo.com
    """
    if verbose:
        click.echo("🤖 Ambivo Agents CLI v1.3.0")
        click.echo("📧 Contact: sgosain@ambivo.com")
        click.echo("🏢 Company: https://www.ambivo.com")
        click.echo("🎬 NEW: YouTube Download Support")

    if config:
        click.echo(f"📋 Using config file: {config}")


@cli.command()
def health():
    """Check system health and status"""
    click.echo("🏥 Checking Ambivo Agents Health...")

    try:
        cli_instance.initialize_service()
        health_status = cli_instance.agent_service.health_check()

        click.echo("\n📊 Health Status:")
        click.echo(f"  Service Available: {'✅' if health_status['service_available'] else '❌'}")
        click.echo(f"  Redis Available: {'✅' if health_status.get('redis_available') else '❌'}")
        click.echo(f"  LLM Available: {'✅' if health_status.get('llm_service_available') else '❌'}")
        click.echo(f"  Active Sessions: {health_status.get('active_sessions', 0)}")

        if health_status.get('available_agent_types'):
            click.echo("\n🤖 Available Agents:")
            for agent_type, available in health_status['available_agent_types'].items():
                status = "✅" if available else "❌"
                click.echo(f"  {agent_type}: {status}")

        if not health_status['service_available']:
            sys.exit(1)

    except Exception as e:
        click.echo(f"❌ Health check failed: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.argument('message')
@click.option('--conversation', '-conv', default='cli', help='Conversation ID')
@click.option('--format', '-f', type=click.Choice(['text', 'json']), default='text', help='Output format')
def chat(message: str, conversation: str, format: str):
    """Send a message to the agent system"""
    click.echo(f"💬 Processing message: {message}")

    async def process():
        result = await cli_instance.process_message(message, conversation)

        if format == 'json':
            click.echo(json.dumps(result, indent=2))
        else:
            if result['success']:
                click.echo(f"🤖 Response: {result['response']}")
                click.echo(f"⏱️  Time: {result['processing_time']:.2f}s")
                click.echo(f"🔧 Agent: {result['agent_id']}")
            else:
                click.echo(f"❌ Error: {result['error']}", err=True)

    asyncio.run(process())


@cli.group()
def media():
    """Media processing commands"""
    pass


@media.command()
@click.argument('input_file')
@click.option('--output-format', '-f', default='mp3', type=click.Choice(['mp3', 'wav', 'aac', 'flac']),
              help='Output format')
@click.option('--quality', '-q', default='medium', type=click.Choice(['low', 'medium', 'high']), help='Audio quality')
def extract_audio(input_file: str, output_format: str, quality: str):
    """Extract audio from video file"""
    if not Path(input_file).exists():
        click.echo(f"❌ File not found: {input_file}", err=True)
        sys.exit(1)

    click.echo(f"🎵 Extracting audio from: {input_file}")

    message = f"""Extract audio from the video file at path: {input_file}

Please use the following settings:
- Output format: {output_format}
- Audio quality: {quality}

Use the extract_audio_from_video tool to process this file."""

    async def process():
        result = await cli_instance.process_message(message, "media_extract")

        if result['success']:
            click.echo(f"✅ {result['response']}")
        else:
            click.echo(f"❌ Error: {result['error']}", err=True)

    asyncio.run(process())


@media.command()
@click.argument('input_file')
@click.option('--output-format', '-f', default='mp4', type=click.Choice(['mp4', 'avi', 'mov', 'mkv']),
              help='Output format')
@click.option('--codec', '-c', default='h264', type=click.Choice(['h264', 'h265', 'vp9']), help='Video codec')
def convert_video(input_file: str, output_format: str, codec: str):
    """Convert video to different format"""
    if not Path(input_file).exists():
        click.echo(f"❌ File not found: {input_file}", err=True)
        sys.exit(1)

    click.echo(f"🎥 Converting video: {input_file}")

    message = f"""Convert the video file at path: {input_file}

Please use the following settings:
- Output format: {output_format}
- Video codec: {codec}

Use the convert_video_format tool to process this file."""

    async def process():
        result = await cli_instance.process_message(message, "media_convert")

        if result['success']:
            click.echo(f"✅ {result['response']}")
        else:
            click.echo(f"❌ Error: {result['error']}", err=True)

    asyncio.run(process())


@cli.group()
def kb():
    """Knowledge base commands"""
    pass


@kb.command()
@click.argument('file_path')
@click.option('--kb-name', '-k', default='default_kb', help='Knowledge base name')
def ingest_file(file_path: str, kb_name: str):
    """Ingest a document file into knowledge base"""
    if not Path(file_path).exists():
        click.echo(f"❌ File not found: {file_path}", err=True)
        sys.exit(1)

    click.echo(f"📄 Ingesting {file_path} into {kb_name}")

    message = f"""Ingest the document file at path: {file_path}

Please use the ingest_document tool to process this file into the knowledge base "{kb_name}" with appropriate metadata."""

    async def process():
        result = await cli_instance.process_message(message, "kb_ingest")

        if result['success']:
            click.echo(f"✅ {result['response']}")
        else:
            click.echo(f"❌ Error: {result['error']}", err=True)

    asyncio.run(process())


@kb.command()
@click.argument('query')
@click.option('--kb-name', '-k', default='default_kb', help='Knowledge base name')
def query(query: str, kb_name: str):
    """Query a knowledge base"""
    click.echo(f"🔍 Querying {kb_name}: {query}")

    message = f"""Query the knowledge base "{kb_name}" with this question: {query}

Please use the query_knowledge_base tool to find relevant information."""

    async def process():
        result = await cli_instance.process_message(message, "kb_query")

        if result['success']:
            click.echo(f"✅ {result['response']}")
        else:
            click.echo(f"❌ Error: {result['error']}", err=True)

    asyncio.run(process())


@cli.group()
def scrape():
    """Web scraping commands"""
    pass


@scrape.command()
@click.argument('url')
@click.option('--output', '-o', help='Output file path')
def url(url: str, output: Optional[str]):
    """Scrape a single URL"""
    click.echo(f"🕷️ Scraping: {url}")

    message = f"""Scrape the web page at this URL: {url}

Please extract:
- Page content
- All links
- Images
- Any structured data

Use the scrape_url tool to fetch this content."""

    async def process():
        result = await cli_instance.process_message(message, "scrape_url")

        if result['success']:
            if output:
                output_path = Path(output)
                response_data = {
                    'url': url,
                    'scraped_at': datetime.now().isoformat(),
                    'content': result['response']
                }

                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(response_data, f, indent=2)
                click.echo(f"💾 Saved to: {output_path}")
            else:
                click.echo(result['response'])
        else:
            click.echo(f"❌ Error: {result['error']}", err=True)

    asyncio.run(process())


@scrape.command()
@click.argument('urls', nargs=-1, required=True)
@click.option('--output-dir', '-o', default='./scraped_content', help='Output directory')
def batch(urls, output_dir: str):
    """Batch scrape multiple URLs"""
    click.echo(f"📦 Batch scraping {len(urls)} URLs")

    url_list = " ".join(urls)
    message = f"""Batch scrape these URLs: {url_list}

Please extract content from all URLs and provide a summary."""

    async def process():
        result = await cli_instance.process_message(message, "scrape_batch")

        if result['success']:
            output_path = Path(output_dir)
            output_path.mkdir(exist_ok=True)

            output_file = output_path / f"batch_scrape_{int(time.time())}.json"
            response_data = {
                'urls': list(urls),
                'scraped_at': datetime.now().isoformat(),
                'results': result['response']
            }

            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(response_data, f, indent=2)

            click.echo(f"✅ {result['response']}")
            click.echo(f"💾 Saved to: {output_file}")
        else:
            click.echo(f"❌ Error: {result['error']}", err=True)

    asyncio.run(process())


@cli.group()
def youtube():
    """YouTube download commands"""
    pass


@youtube.command()
@click.argument('url')
@click.option('--audio-only', '-a', is_flag=True, default=True, help='Download audio only (default)')
@click.option('--video', '-v', is_flag=True, help='Download video (overrides --audio-only)')
@click.option('--output-name', '-n', help='Custom output filename (without extension)')
def download(url: str, audio_only: bool, video: bool, output_name: Optional[str]):
    """Download video or audio from YouTube"""

    # If --video flag is used, override audio_only
    if video:
        audio_only = False

    content_type = "video" if not audio_only else "audio"
    click.echo(f"🎬 Downloading {content_type} from: {url}")

    if output_name:
        message = f"""Download {'audio' if audio_only else 'video'} from {url} with custom filename "{output_name}" """
    else:
        message = f"""Download {'audio' if audio_only else 'video'} from {url}"""

    async def process():
        result = await cli_instance.process_message(message, "youtube_download")

        if result['success']:
            click.echo(f"✅ {result['response']}")
        else:
            click.echo(f"❌ Error: {result['error']}", err=True)

    asyncio.run(process())


@youtube.command()
@click.argument('url')
def info(url: str):
    """Get information about a YouTube video"""
    click.echo(f"📹 Getting video info: {url}")

    message = f"""Get information about this YouTube video: {url}

Please provide details like title, duration, views, and available streams."""

    async def process():
        result = await cli_instance.process_message(message, "youtube_info")

        if result['success']:
            click.echo(result['response'])
        else:
            click.echo(f"❌ Error: {result['error']}", err=True)

    asyncio.run(process())


@youtube.command()
@click.argument('urls', nargs=-1, required=True)
@click.option('--audio-only', '-a', is_flag=True, default=True, help='Download audio only (default)')
@click.option('--video', '-v', is_flag=True, help='Download video (overrides --audio-only)')
def batch(urls, audio_only: bool, video: bool):
    """Batch download multiple YouTube videos"""

    # If --video flag is used, override audio_only
    if video:
        audio_only = False

    content_type = "video" if not audio_only else "audio"
    click.echo(f"📦 Batch downloading {content_type} from {len(urls)} URLs")

    url_list = " ".join(urls)
    message = f"""Batch download {'audio' if audio_only else 'video'} from these YouTube URLs: {url_list}"""

    async def process():
        result = await cli_instance.process_message(message, "youtube_batch")

        if result['success']:
            click.echo(f"✅ {result['response']}")
        else:
            click.echo(f"❌ Error: {result['error']}", err=True)

    asyncio.run(process())


@cli.group()
def search():
    """Web search commands"""
    pass


@search.command()
@click.argument('query')
@click.option('--max-results', '-n', default=5, help='Maximum number of results')
def web(query: str, max_results: int):
    """Search the web"""
    click.echo(f"🔍 Searching: {query}")

    message = f"""Search the web for: {query}

Please find the top {max_results} most relevant results."""

    async def process():
        result = await cli_instance.process_message(message, "web_search")

        if result['success']:
            click.echo(f"✅ {result['response']}")
        else:
            click.echo(f"❌ Error: {result['error']}", err=True)

    asyncio.run(process())


@search.command()
@click.argument('query')
@click.option('--max-results', '-n', default=5, help='Maximum number of results')
def news(query: str, max_results: int):
    """Search for news"""
    click.echo(f"📰 Searching news: {query}")

    message = f"""Search for recent news about: {query}

Please find the top {max_results} most recent and relevant news articles."""

    async def process():
        result = await cli_instance.process_message(message, "news_search")

        if result['success']:
            click.echo(f"✅ {result['response']}")
        else:
            click.echo(f"❌ Error: {result['error']}", err=True)

    asyncio.run(process())


@cli.command()
def interactive():
    """Start interactive chat mode"""
    click.echo("🤖 Starting Ambivo Agents Interactive Mode")
    click.echo("Type 'quit', 'exit', or 'bye' to exit")
    click.echo("Type 'help' for available commands")
    click.echo("-" * 50)

    conversation_id = f"interactive_{int(time.time())}"

    async def interactive_loop():
        while True:
            try:
                user_input = click.prompt("\n🗣️  You", type=str)

                if user_input.lower() in ['quit', 'exit', 'bye']:
                    click.echo("👋 Goodbye!")
                    break
                elif user_input.lower() == 'help':
                    click.echo("""
🤖 Available Commands & Examples:

📝 General:
- Ask any question or give instructions
- 'what is artificial intelligence?'
- 'explain quantum computing'

🎥 Media Processing:
- 'extract audio from /path/to/video.mp4'
- 'convert /path/to/video.avi to mp4'
- 'resize video /path/to/video.mp4 to 720p'

📚 Knowledge Base:
- 'ingest document /path/to/doc.pdf into knowledge base'
- 'query knowledge base: what is our return policy?'
- 'search documents for machine learning'

🕷️ Web Scraping:
- 'scrape https://example.com'
- 'extract content from https://news.site.com'

🎬 YouTube Downloads:
- 'download audio from https://youtube.com/watch?v=example'
- 'download video from https://youtube.com/watch?v=example'
- 'get info about https://youtube.com/watch?v=example'
- 'download https://youtube.com/watch?v=url1 and https://youtube.com/watch?v=url2'

🔍 Web Search:
- 'search for latest AI trends 2024'
- 'find news about space exploration'
- 'search web for Python tutorials'

💻 Code Execution:
- ```python
  print('Hello World')
  import math
  print(math.pi)
  ```
- ```bash
  ls -la
  df -h
  ```

🚪 Exit:
- 'quit' or 'exit' to leave
                    """)
                    continue

                result = await cli_instance.process_message(user_input, conversation_id)

                if result['success']:
                    click.echo(f"🤖 Agent: {result['response']}")
                else:
                    click.echo(f"❌ Error: {result['error']}")

            except KeyboardInterrupt:
                click.echo("\n👋 Goodbye!")
                break
            except EOFError:
                click.echo("\n👋 Goodbye!")
                break

    asyncio.run(interactive_loop())


@cli.command()
def demo():
    """Run a demo showcasing various capabilities"""
    click.echo("🎪 Ambivo Agents Demo")
    click.echo("=" * 50)

    demos = [
        ("💬 General Chat", "Hello! What can you help me with today?"),
        ("🔍 Web Search", "search for latest developments in artificial intelligence"),
        ("📊 Knowledge", "What are the key principles of machine learning?"),
        ("💻 Code",
         "```python\nprint('Demo: Hello from Python!')\nimport datetime\nprint(f'Current time: {datetime.datetime.now()}')\n```"),
    ]

    async def run_demos():
        conversation_id = "demo_session"

        for title, demo_message in demos:
            click.echo(f"\n{title}")
            click.echo("-" * 30)
            click.echo(f"Input: {demo_message}")

            try:
                result = await cli_instance.process_message(demo_message, conversation_id)

                if result['success']:
                    # Truncate long responses for demo
                    response = result['response']
                    if len(response) > 200:
                        response = response[:200] + "..."

                    click.echo(f"Output: {response}")
                    click.echo(f"Agent: {result.get('agent_id', 'unknown')}")
                    click.echo(f"Time: {result.get('processing_time', 0):.2f}s")
                else:
                    click.echo(f"Error: {result.get('error', 'Unknown error')}")

            except Exception as e:
                click.echo(f"Demo error: {e}")

            # Pause between demos
            time.sleep(1)

    asyncio.run(run_demos())

    click.echo(f"\n🎉 Demo completed! Try 'ambivo-agents interactive' for hands-on experience.")


@cli.command()
def examples():
    """Show usage examples for different features"""
    click.echo("📚 Ambivo Agents - Usage Examples")
    click.echo("=" * 50)

    examples = {
        "🎬 YouTube Downloads": [
            "ambivo-agents youtube download 'https://youtube.com/watch?v=dQw4w9WgXcQ'",
            "ambivo-agents youtube download 'https://youtube.com/watch?v=dQw4w9WgXcQ' --video",
            "ambivo-agents youtube info 'https://youtube.com/watch?v=dQw4w9WgXcQ'",
            "ambivo-agents youtube batch 'url1' 'url2' 'url3' --audio-only"
        ],
        "🎵 Media Processing": [
            "ambivo-agents media extract-audio video.mp4 -f mp3 -q high",
            "ambivo-agents media convert-video video.avi -f mp4 -c h264",
        ],
        "📚 Knowledge Base": [
            "ambivo-agents kb ingest-file document.pdf -k company_docs",
            "ambivo-agents kb query 'what is our return policy?' -k company_docs",
        ],
        "🕷️ Web Scraping": [
            "ambivo-agents scrape url https://example.com -o scraped_data.json",
            "ambivo-agents scrape batch 'url1' 'url2' 'url3' -o ./scraped/",
        ],
        "🔍 Web Search": [
            "ambivo-agents search web 'artificial intelligence trends 2024' -n 10",
            "ambivo-agents search news 'space exploration' -n 5",
        ],
        "💬 Interactive Chat": [
            "ambivo-agents interactive",
            "ambivo-agents chat 'Hello, how can you help me?'",
        ]
    }

    for category, example_list in examples.items():
        click.echo(f"\n{category}")
        click.echo("-" * 30)
        for example in example_list:
            click.echo(f"  {example}")

    click.echo(f"\n💡 Tip: Use 'ambivo-agents [command] --help' for detailed options")


if __name__ == '__main__':
    cli()