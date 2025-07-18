import numpy as np
from skimage import feature, morphology
from skimage.draw import disk
from edge_detection_dependance_function.sort_pixels_according_to_relative_distance import sort_pixels_according_to_relative_distance
from edge_detection_dependance_function.interparc import interparc
from edge_detection_dependance_function.make_binary_image_from_points import make_binary_image_from_points
from edge_detection_dependance_function.ellipse_fit_weighted_least_square import ellipse_fit_weighted_least_square
from edge_detection_dependance_function.construct_ellipse import construct_ellipse

def edge_detection_parser(image,**kwargs):
    """
    Parser function to handle edge detection parameters.
    """
    default_values = {
        'boundaryROIImage': None,
        'detector': 'Canny',
        'thresh': np.inf,
        'sigma': np.inf,
        'outlierDetection': 'ellipse',
        'circleOutlierDetectionThicken': 3,
        'xCenter': np.inf,
        'yCenter': np.inf,
        'radius': np.inf,
        'ellipseOutlierDetectionThicken': 3,
        'splineOutlierDetectionMaxDistanceSortPixels': 30,
        'splineOutlierDetectionNumberOfSplinePoints': 150,
        'splineOutlierDetectionMaxDistanceSortSplinePoints': 150,
        'splineOutlierDetectionSplinePointsThicken1': 3,
        'splineOutlierDetectionSplinePointsThicken2': 2,
        'minPixelArea': None,
        'maxPixelArea': None,
        'eccentricity': np.inf,
        'equivDiameter': None,
        'thickenValue': 0,
        'debugLevel': 0
    }

    # Update parameters with user inputs
    params = {**default_values, **kwargs,'image':image}

    # Validate image
    if not isinstance(image, np.ndarray) or image.ndim != 2:
        raise ValueError("Image should be a 2D numpy array.")

    return params


def edge_detection(params):
    """
    Perform edge detection with optional ROI masking and outlier detection.
    """
    detector = params["detector"]
    image = params["image"]
    sigma = params["sigma"]
    boundary_roi_image = params["boundaryROIImage"]
    thicken_value = params["thickenValue"]
    min_pixel_area = params["minPixelArea"]
    max_pixel_area = params["maxPixelArea"]
    outlier_detection = params["outlierDetection"]
    x_center = params["xCenter"]
    y_center = params["yCenter"]
    radius = params["radius"]

    # Edge detection (Canny by default)
    if detector.lower() == 'canny':
        edge = feature.canny(image, sigma=(sigma or 1))
    else:
        raise NotImplementedError(f"Detector '{detector}' not implemented.")

    # ROI Masking (remove edges outside the ROI if provided)
    if boundary_roi_image is not None:
        roi_morph = morphology.binary_dilation(boundary_roi_image, morphology.disk(thicken_value))
        edge = np.logical_and(edge, ~roi_morph)

    # Remove small objects based on pixel area
    if min_pixel_area:
        edge = morphology.remove_small_objects(edge, min_pixel_area)

    if max_pixel_area:
        edge = morphology.remove_small_objects(edge, max_pixel_area)

    # Outlier Detection (based on different methods)
    if outlier_detection == 'spline':
        y_pts, x_pts = np.where(edge)
        pts = np.column_stack([x_pts, y_pts])
        sorted_pts = sort_pixels_according_to_relative_distance(pts, 30)
        sorted_pts = np.vstack([sorted_pts, sorted_pts[0]])
        spline_pts = interparc(150, sorted_pts[:,0], sorted_pts[:,1])
        spline_mask = make_binary_image_from_points(edge.shape, spline_pts)
        spline_mask = morphology.binary_dilation(spline_mask, morphology.disk(3))
        edge = np.logical_and(edge, spline_mask)


    #was macht die?
    elif outlier_detection == 'circle':
        mask = np.zeros_like(edge)
        rr, cc = disk((y_center, x_center), radius, shape=edge.shape) #radius parameter should be use!!
        mask[rr, cc] = 1
        mask = morphology.binary_dilation(mask, morphology.disk(3))
        edge = np.logical_and(edge, mask)


    #was macht die?
    elif outlier_detection == 'ellipse':
        # Fit ellipse and remove outliers based on it
        ellipse_params, _, _ = ellipse_fit_weighted_least_square(edge)
        ellipse_pts = construct_ellipse(
            ellipse_params["xCenter"],
            ellipse_params["yCenter"],
            ellipse_params["a"],
            ellipse_params["b"],
            ellipse_params["alpha"]
        )
        ellipse_mask = make_binary_image_from_points(edge.shape, np.column_stack((ellipse_pts[0], -ellipse_pts[1])))
        ellipse_mask = morphology.binary_dilation(ellipse_mask, morphology.disk(3))
        edge = np.logical_and(edge, ellipse_mask)

    # Return the final edge-detected image
    return edge
