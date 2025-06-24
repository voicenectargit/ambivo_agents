# ambivo_agents/core/base.py
"""
Core base classes and components for ambivo_agents.
"""

import asyncio
import uuid
import time
import tempfile
import os
from pathlib import Path
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Any, Optional, Callable
from concurrent.futures import ThreadPoolExecutor

# Docker imports
try:
    import docker
    DOCKER_AVAILABLE = True
except ImportError:
    DOCKER_AVAILABLE = False


class AgentRole(Enum):
    ASSISTANT = "assistant"
    PROXY = "proxy"
    ANALYST = "analyst"
    RESEARCHER = "researcher"
    COORDINATOR = "coordinator"
    VALIDATOR = "validator"
    CODE_EXECUTOR = "code_executor"


class MessageType(Enum):
    USER_INPUT = "user_input"
    AGENT_RESPONSE = "agent_response"
    SYSTEM_MESSAGE = "system_message"
    TOOL_CALL = "tool_call"
    TOOL_RESPONSE = "tool_response"
    ERROR = "error"
    STATUS_UPDATE = "status_update"


@dataclass
class AgentMessage:
    id: str
    sender_id: str
    recipient_id: Optional[str]
    content: str
    message_type: MessageType
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    session_id: Optional[str] = None
    conversation_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'id': self.id,
            'sender_id': self.sender_id,
            'recipient_id': self.recipient_id,
            'content': self.content,
            'message_type': self.message_type.value,
            'metadata': self.metadata,
            'timestamp': self.timestamp.isoformat(),
            'session_id': self.session_id,
            'conversation_id': self.conversation_id
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AgentMessage':
        """Create from dictionary"""
        return cls(
            id=data['id'],
            sender_id=data['sender_id'],
            recipient_id=data.get('recipient_id'),
            content=data['content'],
            message_type=MessageType(data['message_type']),
            metadata=data.get('metadata', {}),
            timestamp=datetime.fromisoformat(data['timestamp']),
            session_id=data.get('session_id'),
            conversation_id=data.get('conversation_id')
        )


@dataclass
class AgentTool:
    name: str
    description: str
    function: Callable
    parameters_schema: Dict[str, Any]
    requires_approval: bool = False
    timeout: int = 30


@dataclass
class ExecutionContext:
    session_id: str
    conversation_id: str
    user_id: str
    tenant_id: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ProviderConfig:
    """Configuration for LLM providers"""
    name: str
    model_name: str
    priority: int
    max_requests_per_minute: int = 60
    max_requests_per_hour: int = 3600
    cooldown_minutes: int = 5
    request_count: int = 0
    error_count: int = 0
    last_request_time: Optional[datetime] = None
    last_error_time: Optional[datetime] = None
    is_available: bool = True


class ProviderTracker:
    """Tracks provider usage and availability"""

    def __init__(self):
        self.providers: Dict[str, ProviderConfig] = {}
        self.current_provider: Optional[str] = None
        self.last_rotation_time: Optional[datetime] = None
        self.rotation_interval_minutes: int = 30

    def record_request(self, provider_name: str):
        """Record a request to a provider"""
        if provider_name in self.providers:
            provider = self.providers[provider_name]
            provider.request_count += 1
            provider.last_request_time = datetime.now()

    def record_error(self, provider_name: str, error_message: str):
        """Record an error for a provider"""
        if provider_name in self.providers:
            provider = self.providers[provider_name]
            provider.error_count += 1
            provider.last_error_time = datetime.now()

            if provider.error_count >= 3:
                provider.is_available = False

    def is_provider_available(self, provider_name: str) -> bool:
        """Check if a provider is available"""
        if provider_name not in self.providers:
            return False

        provider = self.providers[provider_name]

        if not provider.is_available:
            if (provider.last_error_time and
                    datetime.now() - provider.last_error_time > timedelta(minutes=provider.cooldown_minutes)):
                provider.is_available = True
                provider.error_count = 0
            else:
                return False

        now = datetime.now()
        if provider.last_request_time:
            time_since_last = (now - provider.last_request_time).total_seconds()

            if time_since_last > 3600:
                provider.request_count = 0

            if provider.request_count >= provider.max_requests_per_hour:
                return False

        return True

    def get_best_available_provider(self) -> Optional[str]:
        """Get the best available provider"""
        available_providers = [
            (name, config) for name, config in self.providers.items()
            if self.is_provider_available(name)
        ]

        if not available_providers:
            return None

        available_providers.sort(key=lambda x: (x[1].priority, x[1].error_count))
        return available_providers[0][0]


class DockerCodeExecutor:
    """Secure code execution using Docker containers"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.work_dir = config.get("work_dir", '/opt/ambivo/work_dir')
        self.docker_images = config.get("docker_images", ["sgosain/amb-ubuntu-python-public-pod"])
        self.timeout = config.get("timeout", 60)
        self.default_image = self.docker_images[0] if self.docker_images else "sgosain/amb-ubuntu-python-public-pod"

        if DOCKER_AVAILABLE:
            try:
                self.docker_client = docker.from_env()
                self.docker_client.ping()
                self.available = True
            except Exception as e:
                self.available = False
        else:
            self.available = False

    def execute_code(self, code: str, language: str = "python", files: Dict[str, str] = None) -> Dict[str, Any]:
        """Execute code in Docker container"""
        if not self.available:
            return {
                'success': False,
                'error': 'Docker not available',
                'language': language
            }

        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)

                if language == "python":
                    code_file = temp_path / "code.py"
                    code_file.write_text(code)
                    cmd = ["python", "/workspace/code.py"]
                elif language == "bash":
                    code_file = temp_path / "script.sh"
                    code_file.write_text(code)
                    cmd = ["bash", "/workspace/script.sh"]
                else:
                    raise ValueError(f"Unsupported language: {language}")

                if files:
                    for filename, content in files.items():
                        file_path = temp_path / filename
                        file_path.write_text(content)

                container_config = {
                    'image': self.default_image,
                    'command': cmd,
                    'volumes': {str(temp_path): {'bind': '/workspace', 'mode': 'rw'}},
                    'working_dir': '/workspace',
                    'mem_limit': '512m',
                    'network_disabled': True,
                    'remove': True,
                    'stdout': True,
                    'stderr': True
                }

                start_time = time.time()
                container = self.docker_client.containers.run(**container_config)
                execution_time = time.time() - start_time

                output = container.decode('utf-8') if isinstance(container, bytes) else str(container)

                return {
                    'success': True,
                    'output': output,
                    'execution_time': execution_time,
                    'language': language
                }

        except docker.errors.ContainerError as e:
            return {
                'success': False,
                'error': f"Container error: {e.stderr.decode('utf-8') if e.stderr else 'Unknown error'}",
                'exit_code': e.exit_status,
                'language': language
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'language': language
            }


class BaseAgent(ABC):
    """Base class for all agents"""

    def __init__(self,
                 agent_id: str,
                 role: AgentRole,
                 memory_manager,
                 llm_service = None,
                 name: str = None,
                 description: str = None,
                 tools: List[AgentTool] = None):
        self.agent_id = agent_id
        self.role = role
        self.name = name or f"{role.value}_{agent_id[:8]}"
        self.description = description or f"Agent with role: {role.value}"
        self.memory = memory_manager
        self.llm_service = llm_service
        self.tools = tools or []
        self.active = True
        self.executor = ThreadPoolExecutor(max_workers=4)

    @abstractmethod
    async def process_message(self, message: AgentMessage, context: ExecutionContext) -> AgentMessage:
        """Process incoming message and return response"""
        pass

    def add_tool(self, tool: AgentTool):
        """Add a tool to the agent"""
        self.tools.append(tool)

    def get_tool(self, tool_name: str) -> Optional[AgentTool]:
        """Get a tool by name"""
        return next((tool for tool in self.tools if tool.name == tool_name), None)

    async def execute_tool(self, tool_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a tool"""
        tool = self.get_tool(tool_name)
        if not tool:
            raise ValueError(f"Tool {tool_name} not found")

        try:
            if asyncio.iscoroutinefunction(tool.function):
                result = await tool.function(**parameters)
            else:
                result = await asyncio.get_event_loop().run_in_executor(
                    self.executor, tool.function, **parameters
                )

            return {'success': True, 'result': result}
        except Exception as e:
            return {'success': False, 'error': str(e)}

    def create_response(self,
                        content: str,
                        recipient_id: str,
                        message_type: MessageType = MessageType.AGENT_RESPONSE,
                        metadata: Dict[str, Any] = None,
                        session_id: str = None,
                        conversation_id: str = None) -> AgentMessage:
        """Create a response message"""
        return AgentMessage(
            id=str(uuid.uuid4()),
            sender_id=self.agent_id,
            recipient_id=recipient_id,
            content=content,
            message_type=message_type,
            metadata=metadata or {},
            session_id=session_id,
            conversation_id=conversation_id
        )

    def register_agent(self, agent: 'BaseAgent'):
        """Default implementation - only ProxyAgent should override this"""
        return False