from abc import ABC, abstractmethod

from pygame import Surface


class Component(ABC):
    @abstractmethod
    def draw(self, screen : Surface):
        pass

    @abstractmethod
    def update(self, mouse_pos : tuple, mouse_buttons : tuple):
        pass