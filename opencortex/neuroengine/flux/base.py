from abc import ABC, abstractmethod
from typing import Any, Dict

class Node(ABC):
    """
    A base class for a processing unit in the BCI data flow.
    Each Node takes input(s) and produces output(s).
    """

    def __init__(self, name: str = None):
        self.name = name or self.__class__.__name__

    @abstractmethod
    def __call__(self, *args, **kwargs) -> Any:
        """
        Executes the node's computation.
        """
        pass

    def __str__(self):
        return f"{self.__class__.__name__}(name={self.name})"

    def __repr__(self):
        return self.__str__()

    def __rshift__(self, other):
        return Sequential(self, other)

class Sequential(Node):
    """
    A node that composes other nodes sequentially.
    """

    def __init__(self, *steps: Node, name: str = None):
        super().__init__(name or "Sequential")
        self.steps = steps

    def __call__(self, data: Any) -> Any:
        for step in self.steps:
            data = step(data)
        return data

class Parallel(Node):
    """
    A node that runs multiple branches in parallel on the same input.
    Returns a dictionary with each branch's output.
    """

    def __init__(self, **branches: Node):
        super().__init__("Parallel")
        self.branches = branches

    def __call__(self, data: Any) -> Dict[str, Any]:
        return {name: branch(data) for name, branch in self.branches.items()}