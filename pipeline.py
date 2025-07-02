import cv2
from GUI import GUI

import tkinter as tk



class Pipeline:
    def __init__(self, image_path):
        self.image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        self.root = tk.Tk()
        self.gui = GUI(self.root, self.image)

    def run(self):

        # Start the GUI and allow user interaction
        self.root.mainloop()

# Run the pipeline
if __name__ == '__main__':
    image_path = "path_to_your_image.jpg"  # Replace with your image path
    pipeline = Pipeline(image_path)
    pipeline.run()
