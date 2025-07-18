import numpy as np


"""
construct_circle Compute the points of a circle for given center point and radius

   [ x_circle, y_circle ] = construct_circle( x_center, y_center, radius )

   Arguments:
               x_center:    x-coordinate of circle center
               y_center:    y-coordinate of cirle center
               radius:     Radius of the circle
   output:
               x_circle:    Vector containing the x-coordinates of circle points
               y_circle:    Vector containing the y-coordinates of circle points

"""


def construct_circle(x_center, y_center, radius):
    phi = np.arange(0, 2*np.pi, 0.0001)
    x_circle = radius * np.cos(phi) + x_center
    y_circle = radius * np.sin(phi) + y_center
    return x_circle, y_circle
