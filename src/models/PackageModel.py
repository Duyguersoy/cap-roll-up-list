from typing import List, Literal, Optional, Union

from pydantic import Field

from sdks.novavision.src.base.model import (
    Package,
    Detection,
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


class InputChildDetections(Input):
    name: Literal[
        "inputChildDetections"
    ] = "inputChildDetections"

    value: List[List[Detection]]

    type: Literal[
        "list"
    ] = "list"

    class Config:
        title = "Child Detections"


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


class OutputCropZones(Output):
    name: Literal[
        "outputCropZones"
    ] = "outputCropZones"

    # [
    #   [
    #       [x1, y1],
    #       [x2, y2],
    #       [x3, y3],
    #       [x4, y4],
    #   ]
    # ]
    value: List[List[List[float]]]

    type: Literal[
        "list"
    ] = "list"

    class Config:
        title = "Crop Zones"


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
                "Strategy used to combine confidence scores "
                "when overlapping detections are merged."
            )
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
                "Minimum IoU required to merge overlapping "
                "detections. Range: 0.0 - 1.0."
            )
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
                "Maximum average keypoint distance in pixels "
                "for merging keypoint detections."
            )
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