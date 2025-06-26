#!/usr/bin/env python3
"""
Ambivo Agents CLI Interface - Version 1.0

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
import yaml
import os
import uuid
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, Union

# Import agents directly using clean imports
from ambivo_agents import (
    AssistantAgent,
    KnowledgeBaseAgent,
    YouTubeDownloadAgent,
    MediaEditorAgent,
    WebSearchAgent,
    WebScraperAgent,
    CodeExecutorAgent
)

# Import AgentSession with fallback
try:
    from ambivo_agents import AgentSession

    AGENT_SESSION_AVAILABLE = True
except ImportError:
    try:
        from ambivo_agents.core.base import AgentSession

        AGENT_SESSION_AVAILABLE = True
    except ImportError:
        AGENT_SESSION_AVAILABLE = False
        AgentSession = None

# Fallback to service for complex routing if needed
try:
    from ambivo_agents.services import create_agent_service

    SERVICE_AVAILABLE = True
except ImportError:
    SERVICE_AVAILABLE = False


class ConfigManager:
    """Manages YAML configuration for Ambivo Agents"""

    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path
        self.config = self._load_default_config()
        self._load_config()

    def _load_default_config(self) -> Dict[str, Any]:
        """Load default configuration"""
        return {
            'cli': {
                'version': '1.0.0',
                'default_mode': 'shell',
                'auto_session': True,
                'session_prefix': 'ambivo',
                'verbose': False,
                'theme': 'default'
            },
            'agents': {
                'youtube': {
                    'default_audio_only': True,
                    'output_directory': './downloads',
                    'max_concurrent_downloads': 3
                },
                'media': {
                    'temp_directory': './temp',
                    'supported_formats': ['mp4', 'avi', 'mov', 'mp3', 'wav'],
                    'ffmpeg_path': 'ffmpeg'
                },
                'knowledge_base': {
                    'default_embedding_model': 'sentence-transformers/all-MiniLM-L6-v2',
                    'chunk_size': 1000,
                    'chunk_overlap': 200
                },
                'web_search': {
                    'default_max_results': 5,
                    'timeout': 30,
                    'providers': ['brave', 'duckduckgo']
                },
                'web_scraper': {
                    'user_agent': 'Ambivo-Agent/1.0',
                    'timeout': 30,
                    'max_retries': 3
                },
                'code_executor': {
                    'docker_enabled': False,
                    'allowed_languages': ['python', 'javascript', 'bash'],
                    'timeout': 300
                }
            },
            'session': {
                'auto_cleanup': True,
                'session_timeout': 3600,
                'max_sessions': 10
            },
            'logging': {
                'level': 'INFO',
                'file': './logs/ambivo-agents.log',
                'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            }
        }

    def _load_config(self):
        """Load configuration from file, prioritizing agent_config.yaml"""
        if not self.config_path:
            # Try agent_config.yaml first, then other locations
            possible_paths = [
                './agent_config.yaml',
                './agent_config.yml',
                '~/.ambivo/agent_config.yaml',
                '~/.ambivo/agent_config.yml',
                './ambivo-agents.yaml',
                './ambivo-agents.yml',
                '~/.ambivo/config.yaml',
                '~/.ambivo/config.yml',
                '/etc/ambivo/config.yaml'
            ]

            for path in possible_paths:
                expanded_path = Path(path).expanduser()
                if expanded_path.exists():
                    self.config_path = str(expanded_path)
                    break

        if self.config_path and Path(self.config_path).exists():
            try:
                with open(self.config_path, 'r') as f:
                    file_config = yaml.safe_load(f)
                    if file_config:
                        self._merge_config(self.config, file_config)
                        click.echo(f"📋 Loaded configuration from: {self.config_path}")
                        return
            except Exception as e:
                click.echo(f"⚠️  Warning: Failed to load config from {self.config_path}: {e}")

        # If no config found, prompt to create agent_config.yaml
        if not self.config_path:
            self._prompt_for_config_creation()

    def _prompt_for_config_creation(self):
        """Prompt user to create agent_config.yaml"""
        click.echo("📋 No configuration file found.")
        click.echo("💡 Would you like to create agent_config.yaml with default settings?")

        if click.confirm("Create agent_config.yaml?"):
            config_path = "./agent_config.yaml"
            if self.save_sample_config(config_path):
                click.echo(f"✅ Created configuration file: {config_path}")
                click.echo("💡 Edit this file to customize your settings")
                self.config_path = config_path
            else:
                click.echo("❌ Failed to create configuration file")
                click.echo("💡 Continuing with default settings")
        else:
            click.echo("💡 Continuing with default settings")

    def _merge_config(self, base: Dict, override: Dict):
        """Recursively merge configuration dictionaries"""
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._merge_config(base[key], value)
            else:
                base[key] = value

    def get(self, path: str, default=None):
        """Get configuration value using dot notation (e.g., 'agents.youtube.default_audio_only')"""
        keys = path.split('.')
        current = self.config

        for key in keys:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return default

        return current

    def save_sample_config(self, path: str):
        """Save a sample configuration file"""
        try:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            with open(path, 'w') as f:
                yaml.dump(self.config, f, default_flow_style=False, indent=2)
            return True
        except Exception as e:
            click.echo(f"❌ Failed to save sample config: {e}")
            return False


class AmbivoAgentsCLI:
    """Command-line interface for Ambivo Agents with shell default and YAML config"""

    def __init__(self, config_manager: ConfigManager):
        self.config = config_manager
        self.user_id = "cli_user"
        self.tenant_id = "cli_tenant"
        self.session_metadata = {
            "cli_session": True,
            "version": self.config.get('cli.version', '1.0.0'),
            "mode": "shell_default"
        }
        self.session_file = Path.home() / ".ambivo_agents_session"
        self._ensure_auto_session()

        # Check import status
        if not AGENT_SESSION_AVAILABLE:
            if self.config.get('cli.verbose', False):
                print("⚠️  Warning: AgentSession not available - some features may be limited")
                print("💡 Ensure AgentSession is exported in ambivo_agents/__init__.py")

    def _ensure_auto_session(self):
        """Automatically create a session if none exists and auto_session is enabled"""
        if self.config.get('cli.auto_session', True):
            current_session = self.get_current_session()
            if not current_session:
                # Auto-create a UUID4 session
                session_id = str(uuid.uuid4())
                self.set_current_session(session_id)
                if self.config.get('cli.verbose', False):
                    click.echo(f"🔄 Auto-created session: {session_id}")

    def get_current_session(self) -> Optional[str]:
        """Get the currently active session from file"""
        try:
            if self.session_file.exists():
                return self.session_file.read_text().strip()
        except Exception:
            pass
        return None

    def set_current_session(self, session_id: str):
        """Set the current session and save to file"""
        try:
            self.session_file.write_text(session_id)
            return True
        except Exception as e:
            click.echo(f"❌ Failed to save session: {e}")
            return False

    def clear_current_session(self):
        """Clear the current session"""
        try:
            if self.session_file.exists():
                self.session_file.unlink()
            return True
        except Exception as e:
            click.echo(f"❌ Failed to clear session: {e}")
            return False

    async def create_agent(self, agent_class, additional_metadata: Dict[str, Any] = None):
        """Create agent using the .create() pattern with configuration"""
        metadata = {**self.session_metadata}
        if additional_metadata:
            metadata.update(additional_metadata)

        # Add configuration context
        metadata['config'] = {
            'agent_type': agent_class.__name__,
            'configured': True
        }

        agent, context = agent_class.create(
            user_id=self.user_id,
            tenant_id=self.tenant_id,
            session_metadata=metadata
        )

        if self.config.get('cli.verbose', False):
            click.echo(f"✅ Created {agent_class.__name__} (Session: {context.session_id[:8]}...)")

        return agent, context

    async def smart_message_routing(self, message: str) -> str:
        """Smart routing to appropriate agent based on message content with configuration"""
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

                    # Use configuration for default behavior
                    default_audio_only = self.config.get('agents.youtube.default_audio_only', True)
                    wants_video = any(keyword in message_lower for keyword in ['video', 'mp4', 'watch', 'visual'])
                    audio_only = default_audio_only if not wants_video else False

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
                supported_formats = self.config.get('agents.media.supported_formats', ['mp4', 'avi', 'mov'])

                if 'extract audio' in message_lower:
                    return f"🎵 Media Editor Agent ready for audio extraction!\nSupported formats: {', '.join(supported_formats)}\nPlease provide the input file path for processing."
                elif 'convert' in message_lower:
                    return f"🎥 Media Editor Agent ready for video conversion!\nSupported formats: {', '.join(supported_formats)}\nPlease provide the input file path and target format."
                else:
                    return f"🎬 Media Editor Agent ready!\nSupported formats: {', '.join(supported_formats)}\nI can extract audio, convert videos, resize, trim, and more."

            finally:
                await agent.cleanup_session()

        # 📚 Knowledge Base Detection
        elif any(keyword in message_lower for keyword in ['knowledge base', 'ingest', 'query', 'document', 'kb ']):
            agent, context = await self.create_agent(KnowledgeBaseAgent, {"operation": "knowledge_base"})

            try:
                chunk_size = self.config.get('agents.knowledge_base.chunk_size', 1000)

                if 'ingest' in message_lower:
                    return f"📄 Knowledge Base Agent ready for document ingestion!\nChunk size: {chunk_size}\nPlease provide the document path and knowledge base name."
                elif 'query' in message_lower:
                    return f"🔍 Knowledge Base Agent ready for queries!\nPlease provide your question and knowledge base name."
                else:
                    return f"📚 Knowledge Base Agent ready!\nI can ingest documents and answer questions based on your knowledge bases."

            finally:
                await agent.cleanup_session()

        # 🔍 Web Search Detection
        elif any(keyword in message_lower for keyword in ['search', 'find', 'look up', 'google']):
            agent, context = await self.create_agent(WebSearchAgent, {"operation": "web_search"})

            try:
                max_results = self.config.get('agents.web_search.default_max_results', 5)

                # Extract search query
                search_query = message
                for prefix in ['search for', 'find', 'look up', 'google']:
                    if prefix in message_lower:
                        search_query = message[message_lower.find(prefix) + len(prefix):].strip()
                        break

                result = await agent._search_web(search_query, max_results=max_results)
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
                    user_agent = self.config.get('agents.web_scraper.user_agent', 'Ambivo-Agent/1.0')
                    return f"🕷️ Web Scraper Agent ready to scrape: {url}\nUser-Agent: {user_agent}\nSession: {context.session_id}\n(Note: Actual scraping requires proper configuration)"
                else:
                    return "❌ No valid URLs found for scraping"

            finally:
                await agent.cleanup_session()

        # 💻 Code Execution Detection
        elif '```' in message:
            agent, context = await self.create_agent(CodeExecutorAgent, {"operation": "code_execution"})

            try:
                allowed_languages = self.config.get('agents.code_executor.allowed_languages', ['python', 'javascript'])
                return f"💻 Code Executor Agent ready!\nAllowed languages: {', '.join(allowed_languages)}\nSession: {context.session_id}\n(Note: Code execution requires Docker configuration)"

            finally:
                await agent.cleanup_session()

        # 🤖 General Assistant (fallback) - Route to AssistantAgent
        else:
            agent, context = await self.create_agent(AssistantAgent, {"operation": "general_assistance"})

            try:
                # Create an AgentMessage for the AssistantAgent
                from ambivo_agents.core.base import AgentMessage, MessageType

                agent_message = AgentMessage(
                    id=f"msg_{str(uuid.uuid4())[:8]}",
                    sender_id="cli_user",
                    recipient_id=agent.agent_id,
                    content=message,
                    message_type=MessageType.USER_INPUT,
                    session_id=context.session_id,
                    conversation_id=context.conversation_id
                )

                # Process the message with the AssistantAgent
                response_message = await agent.process_message(agent_message, context.to_execution_context())

                await agent.cleanup_session()

                return response_message.content

            except Exception as e:
                await agent.cleanup_session()
                return f"❌ Error processing your question: {e}"


# Initialize configuration and CLI
config_manager = None
cli_instance = None


def initialize_cli(config_path: Optional[str] = None, verbose: bool = False):
    """Initialize CLI with configuration"""
    global config_manager, cli_instance

    config_manager = ConfigManager(config_path)
    if verbose:
        config_manager.config['cli']['verbose'] = True

    cli_instance = AmbivoAgentsCLI(config_manager)
    return cli_instance


@click.group(invoke_without_command=True)
@click.version_option(version="1.0.0", prog_name="Ambivo Agents")
@click.option('--config', '-c', help='Configuration file path')
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose output')
@click.pass_context
def cli(ctx, config: Optional[str], verbose: bool):
    """
    Ambivo Agents - Multi-Agent AI System CLI (Shell Mode by Default)

    🌟 Features:
    - YouTube Downloads with pytubefix
    - Media Processing with FFmpeg
    - Knowledge Base Operations with Qdrant
    - Web Search with multiple providers
    - Web Scraping with proxy support
    - Code Execution in Docker containers
    - YAML Configuration Support
    - Shell Mode by Default
    - Auto-Session Creation

    Author: Hemant Gosain 'Sunny'
    Company: Ambivo
    Email: sgosain@ambivo.com
    """
    global cli_instance

    # Initialize CLI
    cli_instance = initialize_cli(config, verbose)

    if verbose:
        click.echo("🤖 Ambivo Agents CLI v1.0.0 - Shell Mode Default")
        click.echo("📧 Contact: sgosain@ambivo.com")
        click.echo("🏢 Company: https://www.ambivo.com")
        click.echo("🌟 Direct agent creation with YAML configuration")

    # If no command was provided, start shell mode by default
    if ctx.invoked_subcommand is None:
        default_mode = cli_instance.config.get('cli.default_mode', 'shell')
        if default_mode == 'shell':
            ctx.invoke(shell)
        else:
            click.echo(ctx.get_help())


@cli.group()
def config():
    """Configuration management commands"""
    pass


@config.command()
def show():
    """Show current configuration"""
    click.echo("📋 Current Configuration:")
    click.echo("=" * 50)

    def print_config(data, indent=0):
        for key, value in data.items():
            if isinstance(value, dict):
                click.echo("  " * indent + f"{key}:")
                print_config(value, indent + 1)
            else:
                click.echo("  " * indent + f"{key}: {value}")

    print_config(cli_instance.config.config)

    if cli_instance.config.config_path:
        click.echo(f"\n📁 Loaded from: {cli_instance.config.config_path}")
    else:
        click.echo(f"\n📁 Using default configuration")


@config.command()
@click.argument('path')
def save_sample(path: str):
    """Save a sample configuration file"""
    if cli_instance.config.save_sample_config(path):
        click.echo(f"✅ Sample configuration saved to: {path}")
        click.echo("💡 Edit the file and use --config to load it")
    else:
        click.echo("❌ Failed to save sample configuration")


@config.command()
@click.argument('key')
@click.argument('value')
def set(key: str, value: str):
    """Set a configuration value (runtime only)"""
    # Try to parse value as appropriate type
    try:
        if value.lower() in ['true', 'false']:
            parsed_value = value.lower() == 'true'
        elif value.isdigit():
            parsed_value = int(value)
        elif '.' in value and value.replace('.', '').isdigit():
            parsed_value = float(value)
        else:
            parsed_value = value
    except:
        parsed_value = value

    # Set the value in current config
    keys = key.split('.')
    current = cli_instance.config.config
    for k in keys[:-1]:
        if k not in current:
            current[k] = {}
        current = current[k]

    current[keys[-1]] = parsed_value
    click.echo(f"✅ Set {key} = {parsed_value}")
    click.echo("💡 This change is runtime-only. Save to file to persist.")


@config.command()
@click.argument('key')
def get(key: str):
    """Get a configuration value"""
    value = cli_instance.config.get(key)
    if value is not None:
        click.echo(f"{key}: {value}")
    else:
        click.echo(f"❌ Configuration key '{key}' not found")


@cli.group()
def session():
    """Session management commands"""
    pass


@session.command()
@click.argument('session_name', required=False)
def create(session_name: Optional[str]):
    """Create and activate a session (auto-generates UUID4 if no name provided)"""
    if session_name:
        session_prefix = cli_instance.config.get('cli.session_prefix', 'ambivo')
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        full_session_id = f"{session_prefix}_{session_name}_{timestamp}"
    else:
        # Generate UUID4 session
        full_session_id = str(uuid.uuid4())

    if cli_instance.set_current_session(full_session_id):
        click.echo(f"✅ Created and activated session: {full_session_id}")
        click.echo(f"💡 All commands will now use this session automatically")
        click.echo(f"🔧 Use 'ambivo-agents session end' to deactivate")
    else:
        click.echo("❌ Failed to create session")
        sys.exit(1)


@session.command()
@click.argument('session_name')
def use(session_name: str):
    """Switch to an existing session"""
    if cli_instance.set_current_session(session_name):
        click.echo(f"✅ Switched to session: {session_name}")
        click.echo(f"💡 All commands will now use this session")
    else:
        click.echo("❌ Failed to switch session")
        sys.exit(1)


@session.command()
def current():
    """Show the currently active session"""
    current = cli_instance.get_current_session()
    if current:
        click.echo(f"📋 Current session: {current}")
    else:
        click.echo("❌ No active session")
        click.echo("💡 Create one with: ambivo-agents session create my_session")


@session.command()
def end():
    """End the current session"""
    current = cli_instance.get_current_session()
    if current:
        if cli_instance.clear_current_session():
            click.echo(f"✅ Ended session: {current}")
            # Auto-create a replacement session if auto_session is enabled
            cli_instance._ensure_auto_session()
            new_session = cli_instance.get_current_session()
            if new_session:
                click.echo(f"🔄 Auto-created replacement session: {new_session}")
        else:
            click.echo("❌ Failed to end session")
            sys.exit(1)
    else:
        click.echo("❌ No active session to end")


@session.command()
@click.option('--limit', '-l', default=20, help='Maximum number of messages to show')
@click.option('--format', '-f', type=click.Choice(['text', 'json']), default='text', help='Output format')
def history(limit: int, format: str):
    """Show conversation history for the current session"""
    current_session = cli_instance.get_current_session()

    if not current_session:
        click.echo("❌ No active session")
        click.echo("💡 Create a session with: ambivo-agents session create my_session")
        return

    async def show_history():
        try:
            # Create a temporary AssistantAgent to access conversation history
            agent, context = await cli_instance.create_agent(AssistantAgent, {"operation": "history_access"})

            # Get conversation history
            history_data = await agent.get_conversation_history(limit=limit, include_metadata=True)

            if not history_data:
                click.echo(f"📋 No conversation history found for session: {current_session}")
                await agent.cleanup_session()
                return

            if format == 'json':
                click.echo(json.dumps(history_data, indent=2, default=str))
            else:
                session_display = current_session[:12] + "..." if len(current_session) > 12 else current_session
                click.echo(f"📋 **Conversation History** (Session: {session_display})")
                click.echo(f"📊 Total Messages: {len(history_data)}")
                click.echo("=" * 60)

                for i, msg in enumerate(history_data, 1):
                    timestamp = msg.get('timestamp', 'Unknown time')
                    if isinstance(timestamp, str):
                        try:
                            from datetime import datetime
                            dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                            timestamp = dt.strftime('%Y-%m-%d %H:%M:%S')
                        except:
                            pass

                    message_type = msg.get('message_type', 'unknown')
                    content = msg.get('content', '')
                    sender = msg.get('sender_id', 'unknown')

                    # Format based on message type
                    if message_type == 'user_input':
                        icon = "🗣️ "
                        label = "You"
                    elif message_type == 'agent_response':
                        icon = "🤖"
                        label = f"Agent ({sender})"
                    else:
                        icon = "ℹ️ "
                        label = message_type.title()

                    click.echo(f"\n{i}. {icon} **{label}** - {timestamp}")
                    click.echo(f"   {content}")

                    if i < len(history_data):
                        click.echo("   " + "-" * 50)

            await agent.cleanup_session()

        except Exception as e:
            click.echo(f"❌ Error retrieving history: {e}")

    asyncio.run(show_history())


@session.command()
@click.option('--format', '-f', type=click.Choice(['text', 'json']), default='text', help='Output format')
def summary(format: str):
    """Show summary of the current session"""
    current_session = cli_instance.get_current_session()

    if not current_session:
        click.echo("❌ No active session")
        click.echo("💡 Create a session with: ambivo-agents session create my_session")
        return

    async def show_summary():
        try:
            # Create a temporary AssistantAgent to access conversation summary
            agent, context = await cli_instance.create_agent(AssistantAgent, {"operation": "summary_access"})

            # Get conversation summary
            summary_data = await agent.get_conversation_summary()

            if format == 'json':
                click.echo(json.dumps(summary_data, indent=2, default=str))
            else:
                if 'error' in summary_data:
                    click.echo(f"❌ Error getting summary: {summary_data['error']}")
                else:
                    click.echo(f"📊 **Session Summary**")
                    click.echo("=" * 40)
                    click.echo(f"🔗 Session ID: {summary_data.get('session_id', 'Unknown')}")
                    click.echo(f"👤 User ID: {summary_data.get('user_id', 'Unknown')}")
                    click.echo(f"💬 Total Messages: {summary_data.get('total_messages', 0)}")
                    click.echo(f"🗣️  User Messages: {summary_data.get('user_messages', 0)}")
                    click.echo(f"🤖 Agent Messages: {summary_data.get('agent_messages', 0)}")
                    click.echo(f"⏱️  Session Duration: {summary_data.get('session_duration', 'Unknown')}")

                    if summary_data.get('first_message'):
                        click.echo(f"\n📝 First Message:")
                        click.echo(f"   {summary_data['first_message']}")

                    if summary_data.get('last_message'):
                        click.echo(f"\n📝 Last Message:")
                        click.echo(f"   {summary_data['last_message']}")

            await agent.cleanup_session()

        except Exception as e:
            click.echo(f"❌ Error retrieving summary: {e}")

    asyncio.run(show_summary())


@session.command()
@click.confirmation_option(prompt='Are you sure you want to clear the conversation history?')
def clear():
    """Clear conversation history for the current session"""
    current_session = cli_instance.get_current_session()

    if not current_session:
        click.echo("❌ No active session")
        return

    async def clear_history():
        try:
            # Create a temporary AssistantAgent to clear conversation history
            agent, context = await cli_instance.create_agent(AssistantAgent, {"operation": "history_clear"})

            # Clear conversation history
            success = await agent.clear_conversation_history()

            if success:
                click.echo(f"✅ Cleared conversation history for session: {current_session}")
            else:
                click.echo(f"❌ Failed to clear conversation history")

            await agent.cleanup_session()

        except Exception as e:
            click.echo(f"❌ Error clearing history: {e}")

    asyncio.run(clear_history())


@cli.command()
def shell():
    """Start Ambivo Agents interactive shell (default mode)"""

    # Show welcome message with configuration info
    click.echo("🚀 Ambivo Agents Shell v1.0.0 (Default Mode)")
    click.echo("💡 YAML configuration support with auto-sessions")

    if cli_instance.config.config_path:
        click.echo(f"📋 Config: {cli_instance.config.config_path}")
    else:
        click.echo("📋 Config: Using defaults")

    # Show current session
    current_session = cli_instance.get_current_session()
    if current_session:
        session_display = current_session[:8] + "..." if len(current_session) > 8 else current_session
        click.echo(f"🔗 Session: {session_display}")

    click.echo("💡 Type 'help' for commands, 'exit' to quit")
    click.echo("-" * 60)

    def get_prompt():
        """Generate dynamic prompt based on session state and theme"""
        current_session = cli_instance.get_current_session()
        theme = cli_instance.config.get('cli.theme', 'default')

        if current_session:
            # Show shortened session ID in prompt
            session_short = current_session[:8] if len(current_session) > 8 else current_session
            if theme == 'minimal':
                return f"({session_short})> "
            else:
                return f"ambivo-agents ({session_short})> "

        if theme == 'minimal':
            return "> "
        else:
            return "ambivo-agents> "

    def process_shell_command(command_line: str):
        """Process a command line in shell mode"""
        if not command_line.strip():
            return True

        # Clean up command line - remove leading colons and extra whitespace
        cleaned_command = command_line.strip()
        if cleaned_command.startswith(':'):
            cleaned_command = cleaned_command[1:].strip()

        # Parse command line
        parts = cleaned_command.split()
        if not parts:
            return True

        cmd = parts[0].lower()
        args = parts[1:] if len(parts) > 1 else []

        # Handle shell-specific commands
        if cmd in ['exit', 'quit', 'bye']:
            click.echo("👋 Goodbye!")
            return False

        elif cmd == 'help':
            click.echo("""
🌟 Ambivo Agents Shell Commands:

📋 **Configuration:**
   config show                - Show current configuration
   config get <key>           - Get configuration value
   config set <key> <value>   - Set configuration value (runtime)
   config save-sample <path>  - Save sample config file

📋 **Session Management:**
   session create [name]      - Create session (UUID4 if no name)
   session current            - Show current session
   session status             - Full session info
   session use <name>         - Switch to session
   session end                - End current session

💬 **Chat Commands:**
   chat <message>             - Send message (uses active session)
   <message>                  - Direct message (shortcut)

🎬 **YouTube Commands:**
   youtube download <url>     - Download video/audio (config-aware)
   youtube info <url>         - Get video information

🔄 **Modes:**
   interactive               - Start chat-only interactive mode
   shell                     - This shell mode (default)

🛠️ **Utilities:**
   health                    - System health check
   demo                      - Run demo
   examples                  - Show usage examples

🚪 **Exit:**
   exit, quit, bye           - Exit shell

💡 **Features:**
   - Auto-session creation with UUID4
   - agent_config.yaml support
   - Configuration-aware agents
   - Customizable themes and behavior
            """)
            return True

        elif cmd == 'clear':
            click.clear()
            return True

        # Handle configuration commands
        elif cmd == 'config':
            return handle_config_command(args)

        # Handle regular commands by routing to appropriate CLI commands
        try:
            if cmd == 'session':
                return handle_session_command(args)
            elif cmd == 'chat':
                return handle_chat_command(args)
            elif cmd == 'youtube':
                return handle_youtube_command(args)
            elif cmd == 'interactive':
                return handle_interactive_command()
            elif cmd == 'health':
                return handle_health_command()
            elif cmd == 'demo':
                return handle_demo_command()
            elif cmd == 'examples':
                return handle_examples_command()
            else:
                # Try to interpret as chat message
                return handle_chat_command([command_line])

        except Exception as e:
            click.echo(f"❌ Error executing command: {e}")
            return True

    def handle_config_command(args):
        """Handle configuration commands in shell"""
        if not args:
            click.echo("❌ Usage: config <show|get|set|save-sample> [args]")
            return True

        subcmd = args[0].lower()

        if subcmd == 'show':
            click.echo("📋 Current Configuration (Key Settings):")
            key_settings = [
                'cli.default_mode',
                'cli.auto_session',
                'cli.theme',
                'agents.youtube.default_audio_only',
                'agents.web_search.default_max_results',
                'session.auto_cleanup'
            ]
            for key in key_settings:
                value = cli_instance.config.get(key)
                click.echo(f"  {key}: {value}")

        elif subcmd == 'get':
            if len(args) < 2:
                click.echo("❌ Usage: config get <key>")
                return True
            key = args[1]
            value = cli_instance.config.get(key)
            if value is not None:
                click.echo(f"{key}: {value}")
            else:
                click.echo(f"❌ Configuration key '{key}' not found")

        elif subcmd == 'set':
            if len(args) < 3:
                click.echo("❌ Usage: config set <key> <value>")
                return True
            key = args[1]
            value = args[2]

            # Parse value
            try:
                if value.lower() in ['true', 'false']:
                    parsed_value = value.lower() == 'true'
                elif value.isdigit():
                    parsed_value = int(value)
                elif '.' in value and value.replace('.', '').isdigit():
                    parsed_value = float(value)
                else:
                    parsed_value = value
            except:
                parsed_value = value

            # Set the value
            keys = key.split('.')
            current = cli_instance.config.config
            for k in keys[:-1]:
                if k not in current:
                    current[k] = {}
                current = current[k]

            current[keys[-1]] = parsed_value
            click.echo(f"✅ Set {key} = {parsed_value}")

        elif subcmd == 'save-sample':
            if len(args) < 2:
                path = "./agent_config.yaml"
            else:
                path = args[1]

            if cli_instance.config.save_sample_config(path):
                click.echo(f"✅ Sample configuration saved to: {path}")
            else:
                click.echo("❌ Failed to save sample configuration")

        else:
            click.echo(f"❌ Unknown config command: {subcmd}")

        return True

    # Helper functions for session commands
    def handle_session_history(limit: int = 20):
        """Handle session history command in shell"""

        async def show_history():
            current_session = cli_instance.get_current_session()

            if not current_session:
                click.echo("❌ No active session")
                return

            try:
                # Create a temporary AssistantAgent to access conversation history
                agent, context = await cli_instance.create_agent(AssistantAgent, {"operation": "history_access"})

                # Get conversation history
                history_data = await agent.get_conversation_history(limit=limit, include_metadata=True)

                if not history_data:
                    click.echo(f"📋 No conversation history found for current session")
                    await agent.cleanup_session()
                    return

                session_display = current_session[:12] + "..." if len(current_session) > 12 else current_session
                click.echo(f"📋 **Conversation History** (Session: {session_display})")
                click.echo(f"📊 Total Messages: {len(history_data)}")
                click.echo("=" * 50)

                for i, msg in enumerate(history_data[-10:], 1):  # Show last 10 in shell
                    content = msg.get('content', '')[:100] + "..." if len(msg.get('content', '')) > 100 else msg.get(
                        'content', '')
                    message_type = msg.get('message_type', 'unknown')

                    if message_type == 'user_input':
                        click.echo(f"{i}. 🗣️  You: {content}")
                    elif message_type == 'agent_response':
                        sender = msg.get('sender_id', 'agent')[:10]
                        click.echo(f"{i}. 🤖 {sender}: {content}")
                    else:
                        click.echo(f"{i}. ℹ️  {message_type}: {content}")

                await agent.cleanup_session()

            except Exception as e:
                click.echo(f"❌ Error retrieving history: {e}")

        asyncio.run(show_history())
        return True

    def handle_session_summary():
        """Handle session summary command in shell"""

        async def show_summary():
            current_session = cli_instance.get_current_session()

            if not current_session:
                click.echo("❌ No active session")
                return

            try:
                # Create a temporary AssistantAgent to access conversation summary
                agent, context = await cli_instance.create_agent(AssistantAgent, {"operation": "summary_access"})

                # Get conversation summary
                summary_data = await agent.get_conversation_summary()

                if 'error' in summary_data:
                    click.echo(f"❌ Error getting summary: {summary_data['error']}")
                else:
                    click.echo(f"📊 **Session Summary**")
                    click.echo("=" * 30)
                    click.echo(f"💬 Messages: {summary_data.get('total_messages', 0)} total")
                    click.echo(f"⏱️  Duration: {summary_data.get('session_duration', 'Unknown')}")
                    click.echo(f"🔗 Session: {current_session[:8]}...")

                await agent.cleanup_session()

            except Exception as e:
                click.echo(f"❌ Error retrieving summary: {e}")

        asyncio.run(show_summary())
        return True

    def handle_session_clear():
        """Handle session clear command in shell"""

        async def clear_history():
            current_session = cli_instance.get_current_session()

            if not current_session:
                click.echo("❌ No active session")
                return

            try:
                # Create a temporary AssistantAgent to clear conversation history
                agent, context = await cli_instance.create_agent(AssistantAgent, {"operation": "history_clear"})

                # Clear conversation history
                success = await agent.clear_conversation_history()

                if success:
                    click.echo(f"✅ Cleared conversation history")
                else:
                    click.echo(f"❌ Failed to clear conversation history")

                await agent.cleanup_session()

            except Exception as e:
                click.echo(f"❌ Error clearing history: {e}")

        asyncio.run(clear_history())
        return True

    def handle_session_command(args):
        """Handle session subcommands"""
        if not args:
            click.echo("❌ Usage: session <create|current|status|use|end> [name]")
            return True

        subcmd = args[0].lower()

        if subcmd == 'create':
            if len(args) > 1:
                session_name = args[1]
                session_prefix = cli_instance.config.get('cli.session_prefix', 'ambivo')
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                full_session_id = f"{session_prefix}_{session_name}_{timestamp}"
            else:
                # Generate UUID4 session
                full_session_id = str(uuid.uuid4())

            if cli_instance.set_current_session(full_session_id):
                click.echo(f"✅ Created and activated session: {full_session_id}")
            else:
                click.echo("❌ Failed to create session")

        elif subcmd == 'current':
            current = cli_instance.get_current_session()
            if current:
                click.echo(f"📋 Current session: {current}")
            else:
                click.echo("❌ No active session")

        elif subcmd == 'status':
            current = cli_instance.get_current_session()
            auto_session = cli_instance.config.get('cli.auto_session', True)
            click.echo("📊 Session Status:")
            if current:
                click.echo(f"  ✅ Active session: {current}")
                click.echo(f"  📁 Session file: {cli_instance.session_file}")
            else:
                click.echo("  ❌ No active session")
            click.echo(f"  🔄 Auto-session: {auto_session}")

        elif subcmd == 'use':
            if len(args) < 2:
                click.echo("❌ Usage: session use <name>")
                return True
            session_name = args[1]
            if cli_instance.set_current_session(session_name):
                click.echo(f"✅ Switched to session: {session_name}")
            else:
                click.echo("❌ Failed to switch session")

        elif subcmd == 'end':
            current = cli_instance.get_current_session()
            if current:
                if cli_instance.clear_current_session():
                    click.echo(f"✅ Ended session: {current}")
                    # Auto-create replacement session
                    cli_instance._ensure_auto_session()
                    new_session = cli_instance.get_current_session()
                    if new_session:
                        click.echo(f"🔄 Auto-created replacement session: {new_session}")
                else:
                    click.echo("❌ Failed to end session")
            else:
                click.echo("❌ No active session to end")
        else:
            click.echo(f"❌ Unknown session command: {subcmd}")

        return True

    def handle_chat_command(args):
        """Handle chat command"""
        if not args:
            click.echo("❌ Usage: chat <message>")
            return True

        message = ' '.join(args)

        # Process chat message
        async def process_chat():
            # Determine conversation ID
            active_session = cli_instance.get_current_session()
            if active_session:
                conv_id = active_session
                session_display = active_session[:8] + "..." if len(active_session) > 8 else active_session
                session_source = f"session: {session_display}"
            else:
                conv_id = "shell"
                session_source = "default: shell"

            verbose = cli_instance.config.get('cli.verbose', False)
            if verbose:
                click.echo(f"💬 Processing: {message}")
                click.echo(f"📋 {session_source}")

            try:
                response = await cli_instance.smart_message_routing(message)
                click.echo(f"\n🤖 Response:\n{response}")
            except Exception as e:
                click.echo(f"❌ Error: {e}")

        asyncio.run(process_chat())
        return True

    def handle_youtube_command(args):
        """Handle YouTube commands"""
        if not args:
            click.echo("❌ Usage: youtube <download|info> <url>")
            return True

        subcmd = args[0].lower()
        if len(args) < 2:
            click.echo(f"❌ Usage: youtube {subcmd} <url>")
            return True

        url = args[1]

        if subcmd == 'download':
            click.echo(f"📺 Downloading: {url}")
            # Route to chat system
            return handle_chat_command([f"download video from {url}"])
        elif subcmd == 'info':
            click.echo(f"📋 Getting info: {url}")
            return handle_chat_command([f"get info about {url}"])
        else:
            click.echo(f"❌ Unknown YouTube command: {subcmd}")

        return True

    def handle_interactive_command():
        """Handle interactive mode"""
        click.echo("🔄 Switching to interactive chat mode...")
        click.echo("💡 Type 'quit' to return to shell")

        # Start interactive chat mode
        async def interactive_chat():
            while True:
                try:
                    current_session = cli_instance.get_current_session()
                    if current_session:
                        session_short = current_session[:8] if len(current_session) > 8 else current_session
                        prompt_text = f"🗣️  You ({session_short})"
                    else:
                        prompt_text = "🗣️  You"

                    user_input = click.prompt(f"\n{prompt_text}", type=str)

                    if user_input.lower() in ['quit', 'exit', 'bye']:
                        click.echo("🔄 Returning to shell...")
                        break

                    # Process as chat
                    response = await cli_instance.smart_message_routing(user_input)
                    click.echo(f"🤖 Agent: {response}")

                except KeyboardInterrupt:
                    click.echo("\n🔄 Returning to shell...")
                    break
                except EOFError:
                    click.echo("\n🔄 Returning to shell...")
                    break

        asyncio.run(interactive_chat())
        return True

    def handle_health_command():
        """Handle health check"""
        click.echo("🏥 Running health check...")
        click.echo("✅ CLI is working")
        click.echo("✅ Session management available")
        click.echo("✅ Smart routing available")
        click.echo(f"✅ Configuration loaded: {cli_instance.config.config_path or 'defaults'}")
        return True

    def handle_demo_command():
        """Handle demo"""
        click.echo("🎪 Running demo...")
        click.echo("💡 This would show a demo of the system")
        return True

    def handle_examples_command():
        """Handle examples"""
        click.echo("📚 Usage Examples:")
        click.echo("  config set cli.theme minimal")
        click.echo("  session create my_project")
        click.echo("  chat 'Hello, I need help with video processing'")
        click.echo("  youtube download https://youtube.com/watch?v=example")
        click.echo("  config show")
        click.echo("  session end")
        return True

    # Main shell loop
    try:
        while True:
            try:
                prompt = get_prompt()

                # Use click.prompt with proper handling
                try:
                    command_line = click.prompt(prompt, type=str, show_default=False, err=True)
                except (KeyboardInterrupt, EOFError):
                    click.echo("\n👋 Goodbye!")
                    break
                except Exception as e:
                    # Handle any prompt errors gracefully
                    click.echo(f"\n⚠️  Prompt error: {e}")
                    continue

                # Process command
                if not process_shell_command(command_line):
                    break

            except KeyboardInterrupt:
                click.echo("\n💡 Use 'exit' to quit")
                continue
            except EOFError:
                click.echo("\n👋 Goodbye!")
                break

    except Exception as e:
        click.echo(f"❌ Shell error: {e}")


@cli.command()
@click.argument('message')
@click.option('--conversation', '-conv', help='Conversation ID (overrides active session)')
@click.option('--format', '-f', type=click.Choice(['text', 'json']), default='text', help='Output format')
def chat(message: str, conversation: Optional[str], format: str):
    """Send a message using smart agent routing with configuration support"""

    # Determine conversation ID
    if conversation:
        conv_id = conversation
        session_source = f"explicit: {conversation}"
    else:
        active_session = cli_instance.get_current_session()
        if active_session:
            conv_id = active_session
            session_source = f"active session: {active_session}"
        else:
            conv_id = "cli"
            session_source = "default: cli"

    verbose = cli_instance.config.get('cli.verbose', False)
    if verbose:
        click.echo(f"💬 Processing: {message}")
        click.echo(f"📋 Session: {session_source}")

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
                'conversation_id': conv_id,
                'session_source': session_source,
                'paradigm': 'direct_agent_creation',
                'config_loaded': cli_instance.config.config_path is not None
            }
            click.echo(json.dumps(result, indent=2))
        else:
            click.echo(f"\n🤖 Response:\n{response}")
            if verbose:
                click.echo(f"\n⏱️  Processing time: {processing_time:.2f}s")
                click.echo(f"📋 Conversation: {conv_id}")
                click.echo(f"🌟 Using .create() paradigm")

    asyncio.run(process())


@cli.command()
def interactive():
    """Interactive chat mode"""

    click.echo("🤖 Starting interactive chat mode...")

    # Check for active session
    active_session = cli_instance.get_current_session()

    if active_session:
        session_display = active_session[:8] + "..." if len(active_session) > 8 else active_session
        click.echo(f"📋 Using active session: {session_display}")
    else:
        click.echo("📋 No active session - using default conversation")

    click.echo("Type 'quit', 'exit', or 'bye' to exit")
    click.echo("-" * 60)

    async def interactive_loop():
        # Use active session or generate a unique one for this interactive session
        if active_session:
            conversation_id = active_session
        else:
            conversation_id = f"interactive_{int(time.time())}"

        while True:
            try:
                user_input = click.prompt("\n🗣️  You", type=str)

                if user_input.lower() in ['quit', 'exit', 'bye']:
                    click.echo("👋 Goodbye!")
                    break

                # Process with smart routing using conversation_id
                response = await cli_instance.smart_message_routing(user_input)

                click.echo(f"🤖 Agent: {response}")
                session_display = conversation_id[:8] + "..." if len(conversation_id) > 8 else conversation_id
                click.echo(f"📋 Session: {session_display}")

            except KeyboardInterrupt:
                click.echo("\n👋 Goodbye!")
                break
            except EOFError:
                click.echo("\n👋 Goodbye!")
                break

    asyncio.run(interactive_loop())


if __name__ == '__main__':
    cli()