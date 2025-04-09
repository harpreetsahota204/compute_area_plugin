# Compute Area Plugin for FiftyOne

## Overview
This FiftyOne plugin provides functionality to compute and analyze areas of both bounding boxes and segmentation masks in your computer vision datasets. It's designed to help researchers and developers better understand the size distribution of objects and regions in their image datasets.

## Features

### Bounding Box Area Computation

- Calculates both relative (normalized) and absolute (pixel) areas for bounding boxes
- Automatically adds area measurements as new attributes to each detection
- Supports any detection field in your FiftyOne dataset

### Segmentation Area Computation

- Converts segmentation masks to polyline representations
- Calculates surface areas for segmented regions
- Supports both:
  - Raw segmentation masks (automatically converts to polylines)
  - Existing polyline representations
- Computes both relative (normalized) and absolute (pixel) surface areas

## Use Cases

- Analyzing object size distributions in datasets
- Filtering detections based on area thresholds
- Quality assessment of annotations
- Dataset statistics and visualization
- Performance analysis based on object sizes

## Benefits
- Automated area calculations
- Seamless integration with FiftyOne
- Support for multiple annotation formats
- Dynamic field updates
- Non-destructive operations (preserves original annotations)
## Notes
- Areas are computed and stored directly in the dataset
- Relative areas are normalized to [0,1] range
- Absolute areas are in square pixels
- Supports datasets with mixed content (some samples can have no annotations)
- Handles both single and multiple detections per image

## Technical Details
- Uses the Shoelace formula for polygon area calculations
- Supports closed and filled polygon representations
- Maintains aspect ratios during conversions
- Handles coordinate normalization automatically