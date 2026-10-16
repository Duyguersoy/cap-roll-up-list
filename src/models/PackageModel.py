from typing import List, Literal, Optional, Union

from pydantic import Field

from sdks.novavision.src.base.model import (
    Package,
    Detection,
    ROI,
    Inputs,
    Configs,
    Outputs,
    Response,
    Request,
    Output,
    Input,
    Config,
)


# -----------------------------------------------------------------------------
# Inputs
# -----------------------------------------------------------------------------


class InputParentDetections(Input):
    name: Literal[
        "inputParentDetections"
    ] = "inputParentDetections"

    value: List[Detection]

    type: Literal[
        "list"
    ] = "list"

    class Config:
        title = "Parent Detections"

        json_schema_extra = {
            "shortDescription": (
                "Parent detections from the original image."
            ),
            "longDescription": (
                "Contains parent detections generated on the original image. "
                "The bounding box of each parent detection is used as the "
                "reference region for transforming child detection coordinates "
                "from crop coordinates back to the original image coordinates."
            ),
        }


class InputChildDetections(Input):
    name: Literal[
        "inputChildDetections"
    ] = "inputChildDetections"

    value: Union[
        List[Detection],
        List[List[Detection]],
    ]

    type: Literal[
        "list"
    ] = "list"

    class Config:
        title = "Child Detections"

        json_schema_extra = {
            "shortDescription": (
                "Detections generated from cropped image regions."
            ),
            "longDescription": (
                "Contains child detections generated on cropped image regions. "
                "For a single parent detection, a flat List[Detection] can be "
                "used. For multiple parent detections, child detections must "
                "be grouped as List[List[Detection]] so that each child "
                "detection group corresponds to its parent crop."
            ),
        }


# -----------------------------------------------------------------------------
# Outputs
# -----------------------------------------------------------------------------


class OutputRolledUpDetections(Output):
    name: Literal[
        "outputRolledUpDetections"
    ] = "outputRolledUpDetections"

    value: List[Detection]

    type: Literal[
        "list"
    ] = "list"

    class Config:
        title = "Rolled Up Detections"

        json_schema_extra = {
            "shortDescription": (
                "Child detections transformed to original image coordinates."
            ),
            "longDescription": (
                "Contains child detections whose bounding boxes and supported "
                "keypoints have been transformed from crop coordinates back "
                "to the coordinate system of the original image. Overlapping "
                "detections may also be merged according to the configured "
                "merge settings."
            ),
        }


class OutputCropZones(Output):
    name: Literal[
        "outputCropZones"
    ] = "outputCropZones"

    value: List[ROI]

    type: Literal[
        "list"
    ] = "list"

    class Config:
        title = "Crop Zones"

        json_schema_extra = {
            "shortDescription": (
                "Crop regions generated from parent detections."
            ),
            "longDescription": (
                "Returns each crop region using the NovaVision ROI base model. "
                "Each ROI contains the parent bounding box and the corner "
                "keypoints of the crop region, allowing downstream NovaVision "
                "components to consume standard boundingBox and keyPoints "
                "structures."
            ),
        }


# -----------------------------------------------------------------------------
# Confidence Strategy Options
# -----------------------------------------------------------------------------


class OptionMax(Config):
    name: Literal[
        "max"
    ] = "max"

    value: Literal[
        "max"
    ] = "max"

    type: Literal[
        "string"
    ] = "string"

    field: Literal[
        "option"
    ] = "option"

    class Config:
        title = "Max"


class OptionMean(Config):
    name: Literal[
        "mean"
    ] = "mean"

    value: Literal[
        "mean"
    ] = "mean"

    type: Literal[
        "string"
    ] = "string"

    field: Literal[
        "option"
    ] = "option"

    class Config:
        title = "Mean"


class OptionMin(Config):
    name: Literal[
        "min"
    ] = "min"

    value: Literal[
        "min"
    ] = "min"

    type: Literal[
        "string"
    ] = "string"

    field: Literal[
        "option"
    ] = "option"

    class Config:
        title = "Min"


ConfidenceStrategyOption = Union[
    OptionMax,
    OptionMean,
    OptionMin,
]


# -----------------------------------------------------------------------------
# Configs
# -----------------------------------------------------------------------------


class ConfigConfidenceStrategy(Config):
    name: Literal[
        "ConfigConfidenceStrategy"
    ] = "ConfigConfidenceStrategy"

    value: ConfidenceStrategyOption = Field(
        default_factory=OptionMax
    )

    type: Literal[
        "object"
    ] = "object"

    field: Literal[
        "dropdownlist"
    ] = "dropdownlist"

    class Config:
        title = "Confidence Strategy"

        json_schema_extra = {
            "shortDescription": (
                "Confidence merge strategy."
            ),
            "longDescription": (
                "Defines how confidence scores are combined when multiple "
                "detections are merged. Supported strategies are max, mean, "
                "and min."
            ),
        }


class ConfigOverlapThreshold(Config):
    name: Literal[
        "ConfigOverlapThreshold"
    ] = "ConfigOverlapThreshold"

    value: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
    )

    type: Literal[
        "number"
    ] = "number"

    field: Literal[
        "textInput"
    ] = "textInput"

    class Config:
        title = "Overlap Threshold"

        json_schema_extra = {
            "shortDescription": (
                "IoU threshold for bounding-box merging."
            ),
            "longDescription": (
                "Defines the minimum Intersection over Union (IoU) value "
                "required to merge overlapping bounding-box detections. "
                "The accepted range is 0.0 to 1.0. When the threshold is "
                "0.0, detections are merged when they have a positive "
                "intersection."
            ),
        }


class ConfigKeypointMergeThreshold(Config):
    name: Literal[
        "ConfigKeypointMergeThreshold"
    ] = "ConfigKeypointMergeThreshold"

    value: float = Field(
        default=10.0,
        ge=0.0,
    )

    type: Literal[
        "number"
    ] = "number"

    field: Literal[
        "textInput"
    ] = "textInput"

    class Config:
        title = "Keypoint Merge Threshold"

        json_schema_extra = {
            "shortDescription": (
                "Maximum keypoint merge distance in pixels."
            ),
            "longDescription": (
                "Defines the maximum average Euclidean distance between "
                "corresponding keypoints for two keypoint-based detections "
                "to be considered close enough to merge."
            ),
        }


# -----------------------------------------------------------------------------
# Input / Config / Output Groups
# -----------------------------------------------------------------------------


class RollUpListInputs(Inputs):
    inputParentDetections: InputParentDetections
    inputChildDetections: InputChildDetections


class RollUpListConfigs(Configs):
    configConfidenceStrategy: ConfigConfidenceStrategy = Field(
        default_factory=ConfigConfidenceStrategy
    )

    configOverlapThreshold: ConfigOverlapThreshold = Field(
        default_factory=ConfigOverlapThreshold
    )

    configKeypointMergeThreshold: ConfigKeypointMergeThreshold = Field(
        default_factory=ConfigKeypointMergeThreshold
    )


class RollUpListOutputs(Outputs):
    outputRolledUpDetections: OutputRolledUpDetections
    outputCropZones: OutputCropZones


# -----------------------------------------------------------------------------
# Request / Response
# -----------------------------------------------------------------------------


class RollUpListRequest(Request):
    inputs: Optional[
        RollUpListInputs
    ]

    configs: RollUpListConfigs

    class Config:
        json_schema_extra = {
            "target": "configs"
        }


class RollUpListResponse(Response):
    outputs: RollUpListOutputs


# -----------------------------------------------------------------------------
# Executor
# -----------------------------------------------------------------------------


class RollUpList(Config):
    name: Literal[
        "RollUpList"
    ] = "RollUpList"

    value: Union[
        RollUpListRequest,
        RollUpListResponse,
    ]

    type: Literal[
        "object"
    ] = "object"

    field: Literal[
        "option"
    ] = "option"

    class Config:
        title = "Roll Up List"

        json_schema_extra = {
            "target": {
                "value": 0
            }
        }


class ConfigExecutor(Config):
    name: Literal[
        "ConfigExecutor"
    ] = "ConfigExecutor"

    value: Union[
        RollUpList
    ]

    type: Literal[
        "executor"
    ] = "executor"

    field: Literal[
        "dependentDropdownlist"
    ] = "dependentDropdownlist"

    class Config:
        title = "Task"

        json_schema_extra = {
            "target": "value"
        }


# -----------------------------------------------------------------------------
# Package
# -----------------------------------------------------------------------------


class PackageConfigs(Configs):
    executor: ConfigExecutor


class PackageModel(Package):
    configs: PackageConfigs

    type: Literal[
        "component"
    ] = "component"

    name: Literal[
        "RollUpList"
    ] = "RollUpList"