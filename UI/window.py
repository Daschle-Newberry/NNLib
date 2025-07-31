import numpy as np
from UI.component import Component
import pygame
import sys

class Window:
    def __init__(self, width : int, height : int):
        if not pygame.get_init():
            pygame.init()

        self.width, self.height = width, height
        self.screen = pygame.display.set_mode((width,height))

        self.components = []

    def add_component(self, component : Component):
        self.components.append(component)

    def run(self):
        while True:
            self.screen.fill((0,0,0))

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
            mouse_pos = np.array(pygame.mouse.get_pos())
            mouse_events = pygame.mouse.get_pressed()
            for component in self.components:
                component.update(mouse_pos, mouse_events)
                component.draw(self.screen)

            pygame.display.update()