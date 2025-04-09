import fiftyone as fo
from fiftyone import ViewField as F

import numpy as np

#####################
# bounding box areas# 
####################


def compute_and_set_bbox_areas(dataset, field_name):
    """
    Compute and set relative and absolute bounding box areas for detections in a FiftyOne dataset.
    
    This function calculates both relative (normalized) and absolute (pixel) areas of bounding boxes
    for all detections in the specified field of the dataset. The areas are then stored as new 
    attributes in the dataset.
    
    Args:
        dataset: A FiftyOne dataset containing detections
        field_name: String specifying the field containing detections (e.g., "ground_truth")
        
    Returns:
        None - The function modifies the dataset in place by adding two new fields:
            - {field_name}.detections.relative_bbox_area: Area normalized to [0,1]
            - {field_name}.detections.absolute_bbox_area: Area in square pixels
    """
    # Calculate relative bbox area (width * height in normalized coordinates)
    rel_bbox_area = F("bounding_box")[2] * F("bounding_box")[3]
    
    # Get image dimensions from metadata
    im_width, im_height = F("$metadata.width"), F("$metadata.height")
    
    # Calculate absolute area by multiplying relative area with image dimensions
    abs_area = rel_bbox_area * im_width * im_height
    
    # Set relative bbox area field
    dataset.set_field(
        f"{field_name}.detections.relative_bbox_area", 
        rel_bbox_area
    ).save()
    
    # Set absolute bbox area field
    dataset.set_field(
        f"{field_name}.detections.absolute_bbox_area", 
        abs_area
    ).save()


#####################
# surface areas# 
####################

def convert_segmentation_mask(dataset, field_name):
    """
    Convert segmentation masks to polyline representations in a FiftyOne dataset.
    
    This function takes segmentation masks from a specified field and converts them
    to polyline representations, storing the result in a new field named 
    '{field_name}_polylines'.
    
    Args:
        dataset: A FiftyOne dataset containing segmentation masks
        field_name: String specifying the field containing segmentation masks 
                   (e.g., "ground_truth")
        
    Returns:
        None - The function modifies the dataset in place by adding a new field:
            - {field_name}_polylines: Contains polyline representations of the 
              segmentation masks with closed and filled properties set to True
    """
    # Get all detection masks from the specified field in the dataset
    # This returns a list of Detections objects, one per sample
    segmentation_masks = dataset.values(f"{field_name}.detections")
    
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
            closed=True,  # Ensure polygons are closed
            filled=True,  # Indicate polygons should be filled when visualized
        )
        
        # Add the polylines for this sample to our list
        all_polylines.append(polylines_field)

    # Add the polylines field to every sample in the dataset
    # The new field name is the input field name + "_polylines"
    dataset.set_values(f"{field_name}_polylines", all_polylines)
    dataset.save()

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


def compute_and_set_polygon_areas(dataset, field_name):
    """
    Compute and set relative and absolute surface areas for polygons in a FiftyOne dataset.
    
    This function calculates both relative (normalized) and absolute (pixel) areas of polygons
    for all samples in the dataset using the Shoelace formula. The areas are stored as new 
    attributes in the polylines field.
    
    Args:
        dataset: A FiftyOne dataset containing polyline annotations
        field_name: String specifying the field containing polylines 
                   (e.g., "ground_truth_polylines")
        
    Returns:
        None - The function modifies the dataset in place by adding two new attributes
        to each polyline:
            - relative_surface_area: Area normalized to [0,1]
            - absolute_surface_area: Area in square pixels
    """
    for sample in dataset:
        # Get the points - take the first list from the nested structure
        # Assumes polylines field structure: sample.{field_name}.polylines[0].points[0]
        points = np.array(getattr(sample, field_name).polylines[0].points[0])
        
        # Get image dimensions from metadata
        width = sample.metadata.width
        height = sample.metadata.height
        
        # Compute absolute area using the helper function
        absolute_surface_area = compute_polygon_area(points, width, height)
        
        # Compute relative area by dividing by total image area
        relative_surface_area = absolute_surface_area / (width * height)
        
        # Store both relative and absolute areas in the polyline object
        polyline = getattr(sample, field_name).polylines[0]
        polyline.relative_surface_area = relative_surface_area
        polyline.absolute_surface_area = absolute_surface_area
        
        # Save the sample with updated attributes
        sample.save()

def compute_areas(dataset, field_name, computation_type="bbox_area", has_polylines=False):
    """
    Compute areas for bounding boxes and/or segmentation data in a FiftyOne dataset.
    
    This function serves as the main entry point for area computations. It can:
    1. Compute bounding box areas (relative and absolute)
    2. Compute segmentation areas (relative and absolute) from either:
       - Raw segmentation masks (will convert to polylines first)
       - Existing polyline representations
    
    Args:
        dataset: A FiftyOne dataset containing detections
        field_name: String specifying the field containing detections (e.g., "ground_truth")
        computation_type: String specifying type of area to compute. Must be one of:
            - "bbox_area": compute bounding box areas
            - "surface_area": compute segmentation/polyline areas
        has_polylines: Boolean indicating if the segmentation data is already in polyline format.
            Only relevant if computation_type="surface_area":
            - If False: will convert masks from {field_name} to polylines first
            - If True: will compute areas directly from existing polylines in {field_name}
    
    Returns:
        None - The function modifies the dataset in place by adding new fields/attributes:
        If computation_type="bbox_area":
            - {field_name}.detections.relative_bbox_area
            - {field_name}.detections.absolute_bbox_area
        If computation_type="surface_area":
            If has_polylines=False:
                - Creates {field_name}_polylines from masks
                - Adds relative_surface_area and absolute_surface_area to each polyline
            If has_polylines=True:
                - Adds relative_surface_area and absolute_surface_area to existing polylines
    
    Raises:
        ValueError: If computation_type is not one of "bbox_area" or "surface_area"
    """
    valid_types = ["bbox_area", "surface_area"]
    if computation_type not in valid_types:
        raise ValueError(
            f"Invalid computation_type '{computation_type}'. "
            f"Must be one of: {', '.join(valid_types)}"
        )

    # Compute bounding box areas
    if computation_type == "bbox_area":
        print(f"Computing bounding box areas for field '{field_name}'...")
        compute_and_set_bbox_areas(dataset, field_name)
        print("Bounding box area computation complete.")

    # Compute surface areas
    else:  # computation_type == "surface_area"
        if not has_polylines:
            # Convert masks to polylines first
            print(f"Converting segmentation masks from field '{field_name}' to polylines...")
            convert_segmentation_mask(dataset, field_name)
            print("Conversion to polylines complete.")
            
            # Compute areas using the newly created polylines field
            print(f"Computing polygon areas for field '{field_name}'...")
            polylines_field = f"{field_name}_polylines"
            compute_and_set_polygon_areas(dataset, polylines_field)
            print("Polygon area computation complete.")
        
        else:
            # Compute areas directly from existing polylines
            print(f"Computing polygon areas for existing polylines in field '{field_name}'...")
            compute_and_set_polygon_areas(dataset, field_name)
            print("Polygon area computation complete.")