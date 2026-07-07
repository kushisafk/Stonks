import uuid
from queue import PriorityQueue
from dataclasses import dataclass, field
from typing import Callable, Tuple, Dict, Any

@dataclass(order=True)
class PrioritizedJob:
    """Wrapper encapsulating job callables, prioritization keys, and retry state."""
    priority: int  # Lower numbers execute first
    job_id: str = field(compare=False)
    func: Callable = field(compare=False)
    args: Tuple = field(default_factory=tuple, compare=False)
    kwargs: Dict[str, Any] = field(default_factory=dict, compare=False)
    retries: int = field(default=0, compare=False)
    max_retries: int = field(default=3, compare=False)

    def __init__(
        self, 
        func: Callable, 
        priority: int = 5, 
        args: Tuple = None, 
        kwargs: Dict[str, Any] = None, 
        max_retries: int = 3,
        job_id: str = None
    ):
        self.priority = priority
        self.job_id = job_id or str(uuid.uuid4())
        self.func = func
        self.args = args or ()
        self.kwargs = kwargs or {}
        self.retries = 0
        self.max_retries = max_retries
