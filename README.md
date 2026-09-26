# Roll Up List

Roll Up List is a NovaVision component that transforms detections produced on cropped image regions back into the coordinate system of the original image.

The component receives parent detections that define crop regions and child detections produced inside those regions. Each child detection is translated using the corresponding parent bounding-box offset. Detections belonging to the same class can also be merged according to configurable overlap or keypoint-distance rules.

## Features

- Transforms crop-local detections into original-image coordinates
- Supports single-parent and multi-parent workflows
- Supports grouped child detections for multiple parent crops
- Supports flat child detection lists when only one parent exists
- Generates crop-zone coordinates for every parent detection
- Merges overlapping detections belonging to the same class
- Supports configurable overlap threshold
- Supports `max`, `mean`, and `min` confidence merge strategies
- Supports keypoint detections
- Supports configurable keypoint merge distance
- Supports empty child detection groups
- Validates parent and child detection group relationships

## Package Structure

```text
src/
├── executors/
│   └── RollUpList.py
├── models/
│   └── PackageModel.py
└── utils/
    └── response.py
```

## How It Works

A typical workflow contains two detection stages.

The first detector runs on the original image and produces parent detections. These parent detections define crop regions.

A second detector runs on the cropped regions and produces child detections whose coordinates are relative to the crop.

Roll Up List converts those local coordinates back into the original image coordinate system.

Example:

```text
Parent detection:
left = 100
top  = 50

Child detection:
left = 20
top  = 30

Rolled-up result:
left = 120
top  = 80
```

For multiple parents, child detections must be grouped in the same order as their corresponding parent detections.

```text
Parent 0 -> Child Group 0
Parent 1 -> Child Group 1
Parent 2 -> Child Group 2
```

## Inputs

### Parent Detections

`inputParentDetections`

A list of detections that define crop regions in the original image.

### Child Detections

`inputChildDetections`

Contains detections produced inside parent crop regions.

For a single parent, a flat detection list is supported.

For multiple parents, child detections must be grouped according to the parent order.

Empty child groups are also supported.

## Configurations

### Confidence Strategy

Defines how confidence values are calculated when multiple detections are merged.

Supported values:

- `max`
- `mean`
- `min`

### Overlap Threshold

Defines the overlap threshold used when deciding whether detections of the same class should be merged.

Valid range:

```text
0.0 - 1.0
```

Higher values require stronger overlap before detections are merged.

### Keypoint Merge Threshold

Defines the maximum average distance between corresponding keypoints for keypoint detections to be merged.

The value must be greater than or equal to `0`.

## Outputs

### Rolled Up Detections

`outputRolledUpDetections`

Contains child detections transformed into the coordinate system of the original image.

### Crop Zones

`outputCropZones`

Contains the polygon coordinates of each parent crop region.

## Detection Merge

After child detections are translated into original-image coordinates, detections are grouped by class.

Bounding-box detections belonging to the same class may be merged according to the configured overlap threshold.

When detections are merged:

- The resulting bounding box covers the merged detection area
- Confidence is calculated using the selected confidence strategy
- Detections belonging to different classes remain separate

Non-overlapping detections remain independent.

## Keypoint Support

Detections containing `keyPoints` are processed separately from standard bounding-box detections.

Keypoint coordinates are translated using the parent crop offset.

When keypoint detections are merged:

- Corresponding keypoint coordinates are averaged
- Keypoint confidence values are averaged
- Detection confidence uses the selected confidence strategy

## Validation

For a single parent, a flat `List[Detection]` is supported.

For multiple parents, `List[List[Detection]]` is required.

A flat child detection list is rejected when multiple parents exist because the relationship between child detections and parent crops cannot be determined safely.

The number of child detection groups must match the number of parent detections.

## Installation

```bash
pip install .
```

## Limitations

- Multiple parent detections require grouped child detections
- Instance segmentation mask transformation is not supported
- Roll Up List does not perform object detection itself
- Parent-child relationships must be preserved by the upstream workflow

## Version

`0.0.1`

## License

MIT