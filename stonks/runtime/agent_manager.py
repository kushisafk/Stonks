from typing import Dict, Any
from stonks.runtime.exceptions import AgentRegistrationError
from stonks.logging.logger import logger

class AgentManager:
    """Registers and coordinates active agents executing predictions, technical updates, and risk checks."""
    
    def __init__(self):
        self._agents: Dict[str, Any] = {}
        
    def register_agent(self, name: str, agent_instance: Any) -> None:
        """Adds an agent to the manager dynamically."""
        if not name:
            raise AgentRegistrationError("Agent name cannot be empty.")
        self._agents[name.lower()] = agent_instance
        logger.info(f"AgentManager: Registered agent '{name}' successfully.")
        
    def get_agent(self, name: str) -> Any:
        """Retrieves a registered agent instance."""
        return self._agents.get(name.lower())
        
    def list_agents(self) -> Dict[str, Any]:
        """Lists all registered agent names and instances."""
        return dict(self._agents)
