# ambivo_agents/services/factory.py
"""
Agent Factory for creating different types of agents.
"""

import logging
from typing import Dict, Any, Optional

from ..core.base import AgentRole, BaseAgent
from ..core.memory import MemoryManagerInterface, create_redis_memory_manager
from ..core.llm import LLMServiceInterface
from ..config.loader import load_config, get_config_section, validate_agent_capabilities

from ..agents.assistant import AssistantAgent
from ..agents.code_executor import CodeExecutorAgent


class ProxyAgent(BaseAgent):
    """Agent that routes messages to appropriate specialized agents"""

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
        """Route messages to appropriate agents based on content analysis"""
        self.memory.store_message(message)

        try:
            content = message.content.lower()

            # Route based on content keywords
            if any(keyword in content for keyword in ['execute', 'run', 'code', 'python', 'script']):
                target_agents = [agent for agent in self.agent_registry.values()
                                 if agent.role == AgentRole.CODE_EXECUTOR]
            elif any(keyword in content for keyword in ['research', 'find', 'search', 'knowledge', 'scrape', 'web']):
                target_agents = [agent for agent in self.agent_registry.values()
                                 if agent.role == AgentRole.RESEARCHER]
            else:
                target_agents = [agent for agent in self.agent_registry.values()
                                 if agent.role == AgentRole.ASSISTANT]

            if target_agents:
                target_agent = target_agents[0]
                logging.info(f"Routing message to {target_agent.__class__.__name__} ({target_agent.agent_id})")

                response = await target_agent.process_message(message, context)

                response.metadata.update({
                    'routed_by': self.agent_id,
                    'routed_to': target_agent.agent_id,
                    'routing_reason': f"Content matched {target_agent.role.value} patterns"
                })

                return response
            else:
                response = self.create_response(
                    content=f"I couldn't find an appropriate agent to handle your request. Available agents: {list(self.agent_registry.keys())}",
                    recipient_id=message.sender_id,
                    message_type=message.MessageType.ERROR,
                    session_id=message.session_id,
                    conversation_id=message.conversation_id
                )
                return response

        except Exception as e:
            logging.error(f"Proxy routing error: {e}")
            error_response = self.create_response(
                content=f"Routing error: {str(e)}",
                recipient_id=message.sender_id,
                message_type=message.MessageType.ERROR,
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

            if capabilities.get('web_scraping', False):
                from ..agents.web_scraper import WebScraperAgent
                return WebScraperAgent(
                    agent_id=agent_id,
                    memory_manager=memory_manager,
                    llm_service=llm_service,
                    **kwargs
                )
            elif capabilities.get('knowledge_base', False):
                from ..agents.knowledge_base import KnowledgeBaseAgent
                return KnowledgeBaseAgent(
                    agent_id=agent_id,
                    memory_manager=memory_manager,
                    llm_service=llm_service,
                    **kwargs
                )
            elif capabilities.get('web_search', False):
                from ..agents.web_search import WebSearchAgent
                return WebSearchAgent(
                    agent_id=agent_id,
                    memory_manager=memory_manager,
                    llm_service=llm_service,
                    **kwargs
                )
            elif capabilities.get('media_processing', False):
                from ..agents.media_editor import MediaEditorAgent
                return MediaEditorAgent(
                    agent_id=agent_id,
                    memory_manager=memory_manager,
                    llm_service=llm_service,
                    **kwargs
                )
            else:
                # Fallback to assistant if no specialized researcher is configured
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

        if agent_type == "web_scraper":
            if not capabilities.get('web_scraping', False):
                raise ValueError("Web scraping not enabled in agent_config.yaml")
            from ..agents.web_scraper import WebScraperAgent
            return WebScraperAgent(agent_id, memory_manager, llm_service, **kwargs)

        elif agent_type == "knowledge_base":
            if not capabilities.get('knowledge_base', False):
                raise ValueError("Knowledge base not enabled in agent_config.yaml")
            from ..agents.knowledge_base import KnowledgeBaseAgent
            return KnowledgeBaseAgent(agent_id, memory_manager, llm_service, **kwargs)

        elif agent_type == "web_search":
            if not capabilities.get('web_search', False):
                raise ValueError("Web search not enabled in agent_config.yaml")
            from ..agents.web_search import WebSearchAgent
            return WebSearchAgent(agent_id, memory_manager, llm_service, **kwargs)

        elif agent_type == "media_editor":
            if not capabilities.get('media_processing', False):
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
                'code_executor': capabilities.get('code_execution', True),
                'proxy': True,
                'web_scraper': capabilities.get('web_scraping', False),
                'knowledge_base': capabilities.get('knowledge_base', False),
                'web_search': capabilities.get('web_search', False),
                'media_editor': capabilities.get('media_processing', False)
            }
        except Exception as e:
            logging.error(f"Error getting available agent types: {e}")
            return {
                'assistant': True,
                'code_executor': True,
                'proxy': True,
                'web_scraper': False,
                'knowledge_base': False,
                'web_search': False,
                'media_editor': False
            }

