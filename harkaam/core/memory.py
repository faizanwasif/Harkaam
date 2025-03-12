"""
Memory module for the Harkaam framework.

This module provides different memory implementations that agents
can use to store and retrieve information across execution steps.
"""

from typing import Any, Dict, List, Optional, Union
import datetime
from abc import ABC, abstractmethod

class BaseMemory(ABC):
    """
    Base class for memory implementations.
    
    Memory systems allow agents to store and retrieve information
    across multiple execution steps or sessions.
    """
    
    @abstractmethod
    def add(self, key: str, value: Any) -> None:
        """
        Add a value to memory.
        
        Args:
            key: The key for the value
            value: The value to store
        """
        pass
    
    @abstractmethod
    def get(self, key: str) -> Optional[Any]:
        """
        Get a value from memory.
        
        Args:
            key: The key for the value
            
        Returns:
            The value if found, None otherwise
        """
        pass
    
    @abstractmethod
    def update(self, key: str, value: Any) -> None:
        """
        Update a value in memory.
        
        Args:
            key: The key for the value
            value: The new value
        """
        pass
    
    @abstractmethod
    def delete(self, key: str) -> None:
        """
        Delete a value from memory.
        
        Args:
            key: The key for the value
        """
        pass
    
    @abstractmethod
    def clear(self) -> None:
        """Clear all values from memory."""
        pass

class SimpleMemory(BaseMemory):
    """
    A simple in-memory implementation of the memory system.
    """
    
    def __init__(self):
        """Initialize a new simple memory."""
        self._storage: Dict[str, Any] = {}
    
    def add(self, key: str, value: Any) -> None:
        """
        Add a value to memory.
        
        Args:
            key: The key for the value
            value: The value to store
        """
        # Add timestamp if it's a dictionary
        if isinstance(value, dict) and "created_at" not in value:
            value = value.copy()
            value["created_at"] = datetime.datetime.now().isoformat()
        
        self._storage[key] = value
    
    def get(self, key: str) -> Optional[Any]:
        """
        Get a value from memory.
        
        Args:
            key: The key for the value
            
        Returns:
            The value if found, None otherwise
        """
        return self._storage.get(key)
    
    def update(self, key: str, value: Any) -> None:
        """
        Update a value in memory.
        
        Args:
            key: The key for the value
            value: The new value
        """
        if key in self._storage:
            # Preserve the created_at timestamp if it exists
            if isinstance(self._storage[key], dict) and isinstance(value, dict):
                if "created_at" in self._storage[key] and "created_at" not in value:
                    value = value.copy()
                    value["created_at"] = self._storage[key]["created_at"]
                
                # Add updated_at timestamp
                value = value.copy()
                value["updated_at"] = datetime.datetime.now().isoformat()
            
            self._storage[key] = value
    
    def delete(self, key: str) -> None:
        """
        Delete a value from memory.
        
        Args:
            key: The key for the value
        """
        if key in self._storage:
            del self._storage[key]
    
    def clear(self) -> None:
        """Clear all values from memory."""
        self._storage.clear()
    
    def get_all(self) -> Dict[str, Any]:
        """
        Get all values from memory.
        
        Returns:
            A dictionary of all values in memory
        """
        return self._storage.copy()

class ConversationBufferMemory(BaseMemory):
    """
    A memory implementation that stores conversation history.
    """
    
    def __init__(self, max_messages: int = 100):
        """
        Initialize a new conversation buffer memory.
        
        Args:
            max_messages: Maximum number of messages to store
        """
        self.max_messages = max_messages
        self._storage: Dict[str, Any] = {}
        self.messages: List[Dict[str, Any]] = []
    
    def add_message(self, role: str, content: str, **kwargs) -> None:
        """
        Add a message to the conversation history.
        
        Args:
            role: The role of the message sender (e.g., "user", "agent")
            content: The content of the message
            **kwargs: Additional message metadata
        """
        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.datetime.now().isoformat(),
            **kwargs
        }
        
        self.messages.append(message)
        
        # Trim the messages if they exceed the maximum
        if len(self.messages) > self.max_messages:
            self.messages = self.messages[-self.max_messages:]
    
    def get_conversation_history(self, n: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get the conversation history.
        
        Args:
            n: The number of most recent messages to return (None for all)
            
        Returns:
            A list of messages
        """
        if n is None:
            return self.messages.copy()
        else:
            return self.messages[-n:].copy()
    
    def add(self, key: str, value: Any) -> None:
        """
        Add a value to memory.
        
        Args:
            key: The key for the value
            value: The value to store
        """
        self._storage[key] = value
    
    def get(self, key: str) -> Optional[Any]:
        """
        Get a value from memory.
        
        Args:
            key: The key for the value
            
        Returns:
            The value if found, None otherwise
        """
        return self._storage.get(key)
    
    def update(self, key: str, value: Any) -> None:
        """
        Update a value in memory.
        
        Args:
            key: The key for the value
            value: The new value
        """
        self._storage[key] = value
    
    def delete(self, key: str) -> None:
        """
        Delete a value from memory.
        
        Args:
            key: The key for the value
        """
        if key in self._storage:
            del self._storage[key]
    
    def clear(self) -> None:
        """Clear all values and messages from memory."""
        self._storage.clear()
        self.messages.clear()

# Factory function to create different types of memory
def create_memory(memory_type: str, **kwargs) -> BaseMemory:
    """
    Create a memory instance of the specified type.
    
    Args:
        memory_type: The type of memory to create
        **kwargs: Additional arguments for the memory constructor
        
    Returns:
        A memory instance
    """
    memory_types = {
        "simple": SimpleMemory,
        "conversation_buffer": ConversationBufferMemory,
    }
    
    if memory_type not in memory_types:
        raise ValueError(f"Unknown memory type: {memory_type}")
    
    memory_class = memory_types[memory_type]
    return memory_class(**kwargs)