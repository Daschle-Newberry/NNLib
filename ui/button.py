import numpy as np
from pygame import Surface

from ui.uimath import bounding_box_check
from ui.component import Component

import pygame

class Button(Component):
    def __init__(self,size : tuple, position : tuple, background_color : tuple):
        size = np.array(size)
        position = np.array(position)
        self.canvas = pygame.Surface(size)
        self.canvas.fill(background_color)
        self.position = position - (.5 * size)

        top_left = position - (.5 * size)
        bottom_right = position + (.5 * size)

        self.bbox = np.array([top_left,bottom_right])

        self.pressed = False

        self.background_color = background_color

        self.callback = None

    def set_callback(self, callback):
        self.callback = callback

    def draw(self, screen : Surface):
        screen.blit(self.canvas, self.position)

    def update(self,  mouse_pos : np.ndarray, mouse_buttons : tuple):
        mouse_in_bounds = bounding_box_check(np.array(mouse_pos),self.bbox)

        if mouse_in_bounds and mouse_buttons[0]:
            if not self.pressed:
                self.pressed = True
                self.canvas.fill((0,0,50))
                self.callback()
        else:
            if self.pressed:
                self.pressed = False
                self.canvas.fill((0,0,255))
