#!/usr/bin/env python3
"""
Ambivo Agents CLI Interface

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
@click.version_option(version="1.2.0", prog_name="Ambivo Agents")
@click.option('--config', '-c', help='Configuration file path')
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose output')
def cli(config: Optional[str], verbose: bool):
    """
    Ambivo Agents - Multi-Agent AI System CLI

    A comprehensive toolkit for AI-powered automation including
    media processing, knowledge base operations, web scraping, and more.

    Author: Hemant Gosain 'Sunny'
    Company: Ambivo
    Email: sgosain@ambivo.com
    """
    if verbose:
        click.echo("🤖 Ambivo Agents CLI v1.2.0")
        click.echo("📧 Contact: sgosain@ambivo.com")
        click.echo("🏢 Company: https://www.ambivo.com")

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
Available commands:
- Ask any question or give instructions
- 'extract audio from /path/to/video.mp4'
- 'ingest document /path/to/doc.pdf into knowledge base'
- 'query knowledge base: what is...'
- 'scrape https://example.com'
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


if __name__ == '__main__':
    cli()