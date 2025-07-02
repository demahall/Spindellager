import matplotlib.pyplot as plt

def plot_roi_results(roi_image, boundary_image, cropped_roi):
    # Plot the full image with the ROI boundary
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(roi_image, cmap='gray')
    plt.title("Region of Interest with Boundary")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(cropped_roi, cmap='gray')
    plt.title("Cropped ROI")
    plt.axis('off')

    plt.show()
