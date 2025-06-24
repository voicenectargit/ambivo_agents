# ambivo_agents/services/factory.py
"""
Agent Factory for creating different types of agents - FIXED VERSION

Author: Hemant Gosain 'Sunny'
Company: Ambivo
Email: sgosain@ambivo.com
License: MIT
"""

import logging
from typing import Dict, Any, Optional

from ..core.base import AgentRole, BaseAgent, AgentMessage, MessageType
from ..core.memory import MemoryManagerInterface
from ..core.llm import LLMServiceInterface
from ..config.loader import load_config, get_config_section, validate_agent_capabilities

from ..agents.assistant import AssistantAgent
from ..agents.code_executor import CodeExecutorAgent


class ProxyAgent(BaseAgent):
    """Agent that routes messages to appropriate specialized agents - FIXED VERSION"""

    def __init__(self, agent_id: str, memory_manager, llm_service=None, **kwargs):
        super().__init__(
            agent_id=agent_id,
            role=AgentRole.PROXY,
            memory_manager=memory_manager,
            llm_service=llm_service,
            name="Proxy Agent",
            description="Agent that routes messages to appropriate specialized agents",
            **kwargs
        )
        self.agent_registry: Dict[str, BaseAgent] = {}

    def register_agent(self, agent: BaseAgent) -> bool:
        """Register an agent for routing"""
        if agent and hasattr(agent, 'agent_id'):
            self.agent_registry[agent.agent_id] = agent
            logging.info(f"Registered agent {agent.agent_id} ({agent.__class__.__name__}) with proxy")
            return True
        else:
            logging.error(f"Failed to register agent: invalid agent object")
            return False

    def get_registered_agents(self) -> Dict[str, BaseAgent]:
        """Get all registered agents"""
        return self.agent_registry.copy()

    def unregister_agent(self, agent_id: str) -> bool:
        """Unregister an agent"""
        if agent_id in self.agent_registry:
            del self.agent_registry[agent_id]
            logging.info(f"Unregistered agent {agent_id} from proxy")
            return True
        return False

    async def process_message(self, message, context):
        """Route messages to appropriate agents based on content analysis - FIXED"""
        self.memory.store_message(message)

        try:
            content = message.content.lower()

            # Improved routing logic with more specific patterns
            target_agent = None

            # Knowledge Base routing (HIGHEST PRIORITY for KB operations)
            if any(keyword in content for keyword in [
                'ingest_document', 'ingest_text', 'ingest', 'knowledge base', 'kb',
                'query_knowledge_base', 'query', 'search documents', 'vector database',
                'qdrant', 'semantic search', 'document', 'pdf', 'docx'
            ]):
                # Look for KnowledgeBaseAgent
                for agent in self.agent_registry.values():
                    if 'KnowledgeBaseAgent' in agent.__class__.__name__:
                        target_agent = agent
                        break

            # Code execution routing
            elif any(keyword in content for keyword in [
                'execute', 'run code', 'python', 'bash', '```python', '```bash',
                'script', 'code execution'
            ]):
                # Look for CodeExecutorAgent
                for agent in self.agent_registry.values():
                    if agent.role == AgentRole.CODE_EXECUTOR:
                        target_agent = agent
                        break

            # Web scraping routing (only for explicit scraping requests)
            elif any(keyword in content for keyword in [
                'scrape', 'web scraping', 'crawl', 'extract from url', 'scrape_url',
                'apartments.com', 'scrape website'
            ]):
                # Look for WebScraperAgent
                for agent in self.agent_registry.values():
                    if 'WebScraperAgent' in agent.__class__.__name__:
                        target_agent = agent
                        break

            # Media processing routing
            elif any(keyword in content for keyword in [
                'extract_audio', 'convert_video', 'media', 'ffmpeg', 'audio', 'video',
                'mp4', 'mp3', 'wav', 'extract audio'
            ]):
                # Look for MediaEditorAgent
                for agent in self.agent_registry.values():
                    if 'MediaEditorAgent' in agent.__class__.__name__:
                        target_agent = agent
                        break

            # Web search routing
            elif any(keyword in content for keyword in [
                'search web', 'web search', 'find online', 'search_web',
                'brave search', 'aves search'
            ]):
                # Look for WebSearchAgent
                for agent in self.agent_registry.values():
                    if 'WebSearchAgent' in agent.__class__.__name__:
                        target_agent = agent
                        break

            # Default to Assistant Agent if no specific routing found
            if not target_agent:
                for agent in self.agent_registry.values():
                    if agent.role == AgentRole.ASSISTANT:
                        target_agent = agent
                        break

            if target_agent:
                logging.info(f"Routing message to {target_agent.__class__.__name__} ({target_agent.agent_id})")
                logging.info(f"Routing reason: Content analysis for '{content[:50]}...'")

                response = await target_agent.process_message(message, context)

                response.metadata.update({
                    'routed_by': self.agent_id,
                    'routed_to': target_agent.agent_id,
                    'routed_to_class': target_agent.__class__.__name__,
                    'routing_reason': f"Content matched {target_agent.__class__.__name__} patterns"
                })

                return response
            else:
                error_response = self.create_response(
                    content=f"I couldn't find an appropriate agent to handle your request. Available agents: {[agent.__class__.__name__ for agent in self.agent_registry.values()]}",
                    recipient_id=message.sender_id,
                    message_type=MessageType.ERROR,
                    session_id=message.session_id,
                    conversation_id=message.conversation_id
                )
                return error_response

        except Exception as e:
            logging.error(f"Proxy routing error: {e}")
            error_response = self.create_response(
                content=f"Routing error: {str(e)}",
                recipient_id=message.sender_id,
                message_type=MessageType.ERROR,
                session_id=message.session_id,
                conversation_id=message.conversation_id
            )
            return error_response


class AgentFactory:
    """Factory for creating different types of agents"""

    @staticmethod
    def create_agent(role: AgentRole,
                     agent_id: str,
                     memory_manager: MemoryManagerInterface,
                     llm_service: LLMServiceInterface = None,
                     config: Dict[str, Any] = None,
                     **kwargs) -> BaseAgent:
        """Create an agent of the specified role"""

        if role == AgentRole.ASSISTANT:
            return AssistantAgent(
                agent_id=agent_id,
                memory_manager=memory_manager,
                llm_service=llm_service,
                **kwargs
            )
        elif role == AgentRole.CODE_EXECUTOR:
            return CodeExecutorAgent(
                agent_id=agent_id,
                memory_manager=memory_manager,
                llm_service=llm_service,
                **kwargs
            )
        elif role == AgentRole.RESEARCHER:
            # Import specialized agents dynamically based on configuration
            capabilities = validate_agent_capabilities()

            # PRIORITY ORDER: Knowledge Base > Web Search > Web Scraper > Media Editor
            if capabilities.get('enable_knowledge_base', False):
                try:
                    from ..agents.knowledge_base import KnowledgeBaseAgent
                    logging.info("Creating KnowledgeBaseAgent for RESEARCHER role")
                    return KnowledgeBaseAgent(
                        agent_id=agent_id,
                        memory_manager=memory_manager,
                        llm_service=llm_service,
                        **kwargs
                    )
                except Exception as e:
                    logging.error(f"Failed to create KnowledgeBaseAgent: {e}")

            elif capabilities.get('enable_web_search', False):
                try:
                    from ..agents.web_search import WebSearchAgent
                    logging.info("Creating WebSearchAgent for RESEARCHER role")
                    return WebSearchAgent(
                        agent_id=agent_id,
                        memory_manager=memory_manager,
                        llm_service=llm_service,
                        **kwargs
                    )
                except Exception as e:
                    logging.error(f"Failed to create WebSearchAgent: {e}")

            elif capabilities.get('enable_web_scraping', False):
                try:
                    from ..agents.web_scraper import WebScraperAgent
                    logging.info("Creating WebScraperAgent for RESEARCHER role")
                    return WebScraperAgent(
                        agent_id=agent_id,
                        memory_manager=memory_manager,
                        llm_service=llm_service,
                        **kwargs
                    )
                except Exception as e:
                    logging.error(f"Failed to create WebScraperAgent: {e}")

            elif capabilities.get('enable_media_processing', False):
                try:
                    from ..agents.media_editor import MediaEditorAgent
                    logging.info("Creating MediaEditorAgent for RESEARCHER role")
                    return MediaEditorAgent(
                        agent_id=agent_id,
                        memory_manager=memory_manager,
                        llm_service=llm_service,
                        **kwargs
                    )
                except Exception as e:
                    logging.error(f"Failed to create MediaEditorAgent: {e}")

            # Fallback to assistant if no specialized researcher is available
            logging.warning("No specialized researcher agents available, falling back to AssistantAgent")
            return AssistantAgent(
                agent_id=agent_id,
                memory_manager=memory_manager,
                llm_service=llm_service,
                **kwargs
            )

        elif role == AgentRole.PROXY:
            return ProxyAgent(
                agent_id=agent_id,
                memory_manager=memory_manager,
                llm_service=llm_service,
                **kwargs
            )
        else:
            raise ValueError(f"Unsupported agent role: {role}")

    @staticmethod
    def create_specialized_agent(agent_type: str,
                                 agent_id: str,
                                 memory_manager: MemoryManagerInterface,
                                 llm_service: LLMServiceInterface = None,
                                 **kwargs) -> BaseAgent:
        """Create specialized agents by type name"""

        capabilities = validate_agent_capabilities()

        if agent_type == "knowledge_base":
            if not capabilities.get('enable_knowledge_base', False):
                raise ValueError("Knowledge base not enabled in agent_config.yaml")
            from ..agents.knowledge_base import KnowledgeBaseAgent
            return KnowledgeBaseAgent(agent_id, memory_manager, llm_service, **kwargs)

        elif agent_type == "web_scraper":
            if not capabilities.get('enable_web_scraping', False):
                raise ValueError("Web scraping not enabled in agent_config.yaml")
            from ..agents.web_scraper import WebScraperAgent
            return WebScraperAgent(agent_id, memory_manager, llm_service, **kwargs)

        elif agent_type == "web_search":
            if not capabilities.get('enable_web_search', False):
                raise ValueError("Web search not enabled in agent_config.yaml")
            from ..agents.web_search import WebSearchAgent
            return WebSearchAgent(agent_id, memory_manager, llm_service, **kwargs)

        elif agent_type == "media_editor":
            if not capabilities.get('enable_media_processing', False):
                raise ValueError("Media processing not enabled in agent_config.yaml")
            from ..agents.media_editor import MediaEditorAgent
            return MediaEditorAgent(agent_id, memory_manager, llm_service, **kwargs)

        else:
            raise ValueError(f"Unknown or unavailable agent type: {agent_type}")

    @staticmethod
    def get_available_agent_types() -> Dict[str, bool]:
        """Get available agent types based on configuration"""
        try:
            capabilities = validate_agent_capabilities()
            return {
                'assistant': True,
                'code_executor': capabilities.get('enable_code_execution', False),
                'proxy': True,
                'knowledge_base': capabilities.get('enable_knowledge_base', False),
                'web_scraper': capabilities.get('enable_web_scraping', False),
                'web_search': capabilities.get('enable_web_search', False),
                'media_editor': capabilities.get('enable_media_processing', False)
            }
        except Exception as e:
            logging.error(f"Error getting available agent types: {e}")
            return {
                'assistant': True,
                'code_executor': True,
                'proxy': True,
                'knowledge_base': False,
                'web_scraper': False,
                'web_search': False,
                'media_editor': False
            }