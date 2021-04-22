import logging
from abc import ABCMeta, abstractmethod
from typing import Dict, List, Any

class AbstractNode(metaclass=ABCMeta):

    def __init__(self, config: dict, node_name: str):

        self._inputs = config['input']
        self._outputs = config['output']
        self._name = node_name

        self.logger = logging.getLogger(self._name)

    @classmethod
    def __subclasshook__(cls, subclass):
        return (hasattr(subclass, 'run') and
                callable(subclass.run))

    @abstractmethod
    def run(self, inputs: Dict[str, Any]):
        raise NotImplementedError("This method needs to be implemented")

    @property
    def inputs(self) -> List[str]:
        return self._inputs

    @property
    def outputs(self) -> List[str]:
        return self._outputs

    @property
    def name(self) -> str:
        return self._name