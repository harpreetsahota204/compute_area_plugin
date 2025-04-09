import fiftyone as fo
import numpy as np

from fiftyone import ViewField as F


def compute_polygon_area(points, image_width, image_height):
    """
    Compute the area of a polygon in pixel units using the Shoelace formula.
    
    Args:
        points: List of (x,y) coordinates defining the polygon vertices, normalized to [0,1]
        image_width: Width of the image in pixels
        image_height: Height of the image in pixels
        
    Returns:
        float: Area of the polygon in square pixels
        
    Notes:
        The Shoelace formula (also known as the surveyor's formula) calculates the area 
        of a polygon by using the coordinates of its vertices. The formula gets its name
        from the way the computation "laces" together vertex coordinates.
    """
    # Convert points list to numpy array for vectorized operations
    points = np.array(points)
    
    # Scale normalized coordinates back to pixel dimensions
    points[:, 0] *= image_width  # Scale x coordinates
    points[:, 1] *= image_height # Scale y coordinates
    
    # Extract x and y coordinates into separate arrays
    x = points[:, 0]
    y = points[:, 1]
    
    # Create shifted versions of coordinate arrays
    # np.roll shifts array elements by 1 position for the formula
    x_shift = np.roll(x, 1)
    y_shift = np.roll(y, 1)
    
    # Apply Shoelace formula: A = 1/2 * |sum(x_i*y_i+1 - x_i+1*y_i)|
    return 0.5 * np.abs(np.sum(x * y_shift - x_shift * y))


rel_bbox_area = F("bounding_box")[2] * F("bounding_box")[3]

im_width, im_height = F("$metadata.width"), F("$metadata.height")

abs_area = rel_bbox_area * im_width * im_height

hub_dataset.set_field("ground_truth.detections.relative_bbox_area", rel_bbox_area).save()

hub_dataset.set_field("ground_truth.detections.absolute_bbox_area", abs_area).save()



# Get all ground truth detection masks from the dataset
# This returns a list of Detections objects, one per sample
segmentation_masks = hub_dataset.values("ground_truth.detections")

# Initialize an empty list to store polyline representations for each sample
all_polylines = []

# Iterate through detections for each sample in the dataset
for sample_segmentation in segmentation_masks:
    # For each detection in the sample, convert its segmentation mask to a polyline
    # If sample has no detections (None), create empty list
    polylines = [segmentation.to_polyline() for segmentation in sample_segmentation] if sample_segmentation else []
    
    # Create a FiftyOne Polylines field containing the polyline representations
    polylines_field = fo.Polylines(
        polylines=polylines,
        closed=True,
        filled=True,
        )
    
    # Add the polylines for this sample to our list
    all_polylines.append(polylines_field)

# Add the polylines field to every sample in the dataset
# This creates a new field called "polylines" containing the polyline representations
hub_dataset.set_values("polylines", all_polylines)



for sample in hub_dataset:
    # Get the points - take the first list from the nested structure
    points = np.array(sample.polylines.polylines[0].points[0])  # Note the [0] to get first list
    
    # Get image dimensions
    width = sample.metadata.width
    height = sample.metadata.height
    
    # Compute area using the helper function
    absolute_surface_area = compute_polygon_area(points, width, height)

    relative_surface_area = area / (width * height)
    
    # Store both relative and absolute areas
    sample.polylines.polylines[0].relative_surface_area = relative_surface_area
    sample.polylines.polylines[0].absolute_surface_area = absolute_surface_area
    
    # Save the sample
    sample.save()