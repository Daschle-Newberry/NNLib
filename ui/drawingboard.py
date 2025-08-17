import numpy as np
from pygame import Surface

from ui.uimath import bounding_box_check
from ui.component import Component
import pygame

from PIL import Image
class DrawingBoard(Component):
    def __init__(self,size : tuple, position : tuple):
        size = np.array(size)
        position = np.array(position)
        self.canvas = pygame.Surface(size)
        self.canvas.fill('White')
        self.position = position - (.5 * size)

        top_left = position - (.5 * size)
        bottom_right = position + (.5 * size)


        self.bbox = np.array([top_left,bottom_right])


    def draw(self, screen : Surface):
        screen.blit(self.canvas, self.position)

    def update(self,  mouse_pos : np.ndarray, mouse_buttons : tuple):
        mouse_in_bounds = bounding_box_check(np.array(mouse_pos),self.bbox)

        if mouse_in_bounds and mouse_buttons[0]:
            local_mouse_pos = mouse_pos - self.position
            pygame.draw.circle(self.canvas, 'Black', local_mouse_pos, 10)
        if mouse_in_bounds and mouse_buttons[2]:
            self.canvas.fill('White')

    def to_array(self):
        compressed = pygame.transform.smoothscale(self.canvas,(28,28))
        arr = pygame.surfarray.array3d(compressed)
        return arr