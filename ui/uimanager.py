import numpy as np

from ui.button import Button
from ui.drawingboard import DrawingBoard
from ui.uimath import convert_image_to_grayscale, invert_image
from NNLib.models.sequential import Sequential


class NetworkUIManager:
    def __init__(self, network : Sequential, drawing_board : DrawingBoard, button : Button, means, stds):
        self.network = network
        self.drawing_board = drawing_board
        self.button = button
        self.button.set_callback(self.on_click_callback)
        self.mean = means
        self.std = stds

    def on_click_callback(self):
        image = self.drawing_board.to_array()

        image = np.transpose(image, (1, 0, 2))

        image = convert_image_to_grayscale(image)

        image  = invert_image(image)
        image = (image - self.mean) / self.std

        image = image.reshape(1,1,28,28)
        print(np.argmax(self.network.predict(image)))