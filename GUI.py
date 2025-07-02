import tkinter as tk
from tkinter import messagebox
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
from extractROI import extract_roi
from robust_contrast_normalization import robust_contrast_normalization
from edge_detection import detect_edges
from circlefitGaussNewtonGeometric import fit_circle_gauss_newton

class GUI:
    def __init__(self, root, image):
        self.root = root
        self.image = image
        self.outer_points = []
        self.inner_points = []

        # Get image dimensions for scaling
        self.img_height, self.img_width = image.shape

        #For zooming and dragging
        self.zoom_factor = 1.0
        self.is_mouse_pressed = False
        self.is_dragging = False # To distinguish between dragging and point selection
        self.drag_threshold = 5  # Movement threshold to start dragging (in pixels)
        self.pan_start = None

        #ROI image
        self.cropped_roi = None
        self.roi_offset = None

        #ROI robust contrast
        self.normalized_roi = None

        #Edges detection
        self.edges_image = None

        #Fitting Edges into circle
        self.fitted_circle = None

        self.setup_gui()

    def setup_gui(self):

        self.root.title("GUI")

        # Create the matplotlib figure for displaying the image
        self.fig, self.ax = plt.subplots(figsize=(8, 8))
        self.ax.imshow(self.image, cmap='gray')
        self.ax.axis('on')
        self.ax.grid(True)

        # Embed the matplotlib figure into Tkinter window
        self.matplotlib_canvas = FigureCanvasTkAgg(self.fig, self.root)
        self.matplotlib_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Change cursor to a thin plus for better accuracy
        self.root.config(cursor="crosshair")

        # Bind mouse click to select points directly on the matplotlib canvas

        self.matplotlib_canvas.mpl_connect('button_press_event', self.on_click)  # Click point
        self.matplotlib_canvas.mpl_connect('motion_notify_event', self.on_drag_motion)  # Drag image
        self.matplotlib_canvas.mpl_connect('button_release_event', self.on_drag_release)  # End dragging
        self.matplotlib_canvas.mpl_connect('scroll_event', self.zoom_image)  # Zoom on scroll

        # Instructions label
        self.instructions = tk.Label(self.root, text="Click to select points for outer circle", font=("Arial", 14))
        self.instructions.pack()

        # Button to process ROI
        self.process_button = tk.Button(self.root, text="Confirm points", command=self.process_roi)
        self.process_button.pack(side=tk.LEFT,padx=5)

        # Button to reset and start over
        self.reset_button = tk.Button(self.root, text="Reset", command=self.reset_gui)
        self.reset_button.pack(side=tk.RIGHT,padx=5)

        # ROI button
        self.roi_button = tk.Button(
            self.root,
            text="ROI",
            command=self.show_roi_window,
            state=tk.DISABLED  # Start disabled, enable after ROI is processed
        )
        self.roi_button.pack(side=tk.LEFT, padx=5)

        # Normalize ROI button
        self.norm_button = tk.Button(
            self.root,
            text="Normalized ROI",
            command=self.show_normalized_roi_window,
            state=tk.DISABLED  # Disable until normalization is done
        )
        self.norm_button.pack(side=tk.LEFT, padx=5)

        # Edges detection button
        self.edges_button = tk.Button(
            self.root,
            text="Show Edges",
            command=self.show_edges_window,
            state=tk.DISABLED
        )
        self.edges_button.pack(side=tk.LEFT, padx=5)

    def zoom_image(self, event):
        # Zoom in and out based on scroll wheel
        base_zoom_factor = 0.1  # Zoom step size

        if event.button == 'up':  # Zoom in
            self.zoom_factor *= (1 + base_zoom_factor)
        elif event.button == 'down':  # Zoom out
            self.zoom_factor *= (1 - base_zoom_factor)

        # Apply zoom by setting axis limits based on zoom factor
        new_width = self.img_width * self.zoom_factor
        new_height = self.img_height * self.zoom_factor

        self.ax.set_xlim([0, new_width])
        self.ax.set_ylim([new_height, 0])  # Invert y-axis for correct display

        self.matplotlib_canvas.draw()  # Redraw canvas after zooming

    def on_click(self, event):
        if event.xdata is not None and event.ydata is not None:
            self.pan_start = (event.xdata, event.ydata)
            self.is_mouse_pressed = True
            self.is_dragging = False
            print("Mouse pressed.")

    def on_drag_motion(self, event):

        if self.is_mouse_pressed and event.xdata is not None and event.ydata is not None:

            dx = event.xdata - self.pan_start[0]
            dy = event.ydata - self.pan_start[1]

            if not self.is_dragging and (abs(dx) > 5 or abs(dy) > 5):
                self.is_dragging = True
                print("Dragging started.")

            if self.is_dragging:

                xlim, ylim = self.ax.get_xlim(), self.ax.get_ylim()
                self.ax.set_xlim([xlim[0] - dx, xlim[1] - dx])
                self.ax.set_ylim([ylim[0] - dy, ylim[1] - dy])

                self.pan_start = (event.xdata, event.ydata)
                self.matplotlib_canvas.draw()

    def on_drag_release(self, event):
        if self.is_mouse_pressed:
            if not self.is_dragging:
                # This was a click without drag => select point
                print("Mouse clicked (no drag). Adding point.")
                if len(self.outer_points) < 2:
                    scaled_x = event.xdata
                    scaled_y = event.ydata
                    if scaled_x is not None and scaled_y is not None:
                        self.outer_points.append((scaled_x, scaled_y))
                        self.ax.plot(scaled_x, scaled_y, 'ro', markersize=3)
                        self.matplotlib_canvas.draw()
                elif len(self.inner_points) < 2:
                    scaled_x = event.xdata
                    scaled_y = event.ydata
                    if scaled_x is not None and scaled_y is not None:
                        self.inner_points.append((scaled_x, scaled_y))
                        self.ax.plot(scaled_x, scaled_y, 'go', markersize=3)
                        self.matplotlib_canvas.draw()

                # Update instructions
                if len(self.outer_points) == 2 and len(self.inner_points) == 0:
                    self.instructions.config(text="Click to select points for inner circle")
            else:
                # This was a drag
                print("Mouse drag finished.")


        # Reset flags
        self.is_mouse_pressed = False
        self.is_dragging = False
        self.pan_start = None

    def process_roi(self):
        # Check if exactly two circles are selected
        if len(self.outer_points) != 2 or len(self.inner_points) != 2:
            messagebox.showerror("Error", "Please select exactly two points for both circles!")
            return

        # Calculate outer circle radius
        outer_center_x, outer_center_y = self.outer_points[0]
        outer_boundary_x, outer_boundary_y = self.outer_points[1]
        outer_radius = np.sqrt((outer_boundary_x - outer_center_x) ** 2 + (outer_boundary_y - outer_center_y) ** 2)

        # Calculate inner circle radius
        inner_center_x, inner_center_y = self.inner_points[0]
        inner_boundary_x, inner_boundary_y = self.inner_points[1]
        inner_radius = np.sqrt((inner_boundary_x - inner_center_x) ** 2 + (inner_boundary_y - inner_center_y) ** 2)

        # Show the results in the GUI
        self.instructions.config(text="Circles processed. Press Reset to start over.")

        # Extract ROI
        _, cropped, offset = extract_roi(self.image,self.outer_points[0],
                                 outer_radius,self.inner_points[0], inner_radius)

        self.cropped_roi = cropped
        self.roi_offset = offset

        # Compute robust normalized contrast
        normalized,stretchlim = robust_contrast_normalization(self.cropped_roi,(2 , 2))
        self.normalized_roi = normalized

        # Edge detection
        edges = detect_edges(self.normalized_roi, threshold=(0.1, 0.3), sigma=1.0, min_area=30)
        self.edges_image = edges

        # Circle fitting
        self.fitted_circle = fit_circle_gauss_newton(self.edges_image)

        print("ROI extracted, normalized, edges detected, and circle fitted:")

        #Show circle
        center_x = {"inner_center_x": inner_center_x,
                    "outer_center_x": outer_center_x,
                    "fitted_center_x": self.fitted_circle["x_center"]}
        center_y = {"inner_center_y": inner_center_y,
                    "outer_center_y": outer_center_y,
                    "fitted_center_y": self.fitted_circle["y_center"]}
        radius = {"inner_radius": inner_radius,
                  "outer_radius": outer_radius,
                  "fitted_center_radius": self.fitted_circle["radius"]}
        # Draw both circles on the image
        self.draw_circles(center_x, center_y, radius)

        self.roi_button.config(state=tk.NORMAL)
        self.norm_button.config(state=tk.NORMAL)
        self.edges_button.config(state=tk.NORMAL)



    def show_roi_window(self):
        if self.cropped_roi is None:
            print("No ROI computed yet.")
            return

        # Create a new window
        roi_win = tk.Toplevel(self.root)
        roi_win.title("ROI Result")

        # Create matplotlib figure for the ROI
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(self.cropped_roi, cmap='gray')
        ax.axis('off')
        ax.set_title("Cropped ROI")

        canvas = FigureCanvasTkAgg(fig, roi_win)
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Add a Back button to close the ROI window
        back_button = tk.Button(
            roi_win,
            text="Back",
            command=roi_win.destroy
        )
        back_button.pack(pady=5)

    def show_normalized_roi_window(self):
        if self.normalized_roi is None:
            print("No normalized ROI to display.")
            return

        norm_win = tk.Toplevel(self.root)
        norm_win.title("Normalized ROI")

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(self.normalized_roi, cmap='gray')
        ax.axis('off')
        ax.set_title("Normalized ROI")

        canvas = FigureCanvasTkAgg(fig, norm_win)
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        back_button = tk.Button(
            norm_win,
            text="Back",
            command=norm_win.destroy
        )
        back_button.pack(pady=5)

    def show_edges_window(self):
        if self.edges_image is None:
            print("No edges to display.")
            return

        edge_win = tk.Toplevel(self.root)
        edge_win.title("Detected Edges")

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(self.edges_image, cmap='gray')
        ax.axis('off')
        ax.set_title("Edges")

        canvas = FigureCanvasTkAgg(fig, edge_win)
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        back_button = tk.Button(
            edge_win,
            text="Back",
            command=edge_win.destroy
        )
        back_button.pack(pady=5)

    def reset_gui(self):
        # Clear all points and circles, and restart the process
        self.outer_points.clear()
        self.inner_points.clear()

        # Clear the figure and axes, then redraw the image
        self.ax.clear()
        self.ax.imshow(self.image, cmap='gray')
        self.ax.axis('on')
        self.ax.grid(True)

        # Redraw the canvas
        self.matplotlib_canvas.draw()

        # Update instructions
        self.instructions.config(text="Click to select points for outer circle")

        # Reset all results
        self.cropped_roi = None
        self.normalized_roi = None
        self.roi_button.config(state=tk.DISABLED)
        self.norm_button.config(state=tk.DISABLED)
        self.edges_button.config(state=tk.DISABLED)

    def draw_circles(self, center_x,center_y,radius):

        x_offset,y_offset = self.roi_offset
        fitted_center_x = center_x["fitted_center_x"]+x_offset
        fitted_center_y = center_y["fitted_center_y"]+y_offset


        # Draw the outer circle (blue)
        outer_circle = plt.Circle((center_x["outer_center_x"],center_y["outer_center_y"]),
                                  radius["outer_radius"], color='b', fill=False,
                                  linewidth=1)
        self.ax.add_artist(outer_circle)

        # Draw the inner circle (green)
        inner_circle = plt.Circle((center_x["inner_center_x"],center_y["inner_center_y"]),
                                  radius["inner_radius"], color='g', fill=False,
                                  linewidth=1)
        self.ax.add_artist(inner_circle)

        #Draw the cage (red)

        cage_circle = plt.Circle((fitted_center_x,fitted_center_y),
                                  radius["fitted_center_radius"], color='r', fill=False,
                                  linewidth=1)
        self.ax.add_artist(cage_circle)

        # Redraw the canvas to update the circles
        self.matplotlib_canvas.draw()

