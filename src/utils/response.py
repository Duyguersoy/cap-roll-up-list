from sdks.novavision.src.helper.package import PackageHelper


if __package__:
    from ..models.PackageModel import (
        PackageModel,
        PackageConfigs,
        ConfigExecutor,
        RollUpListOutputs,
        RollUpListResponse,
        RollUpList,
        OutputRolledUpDetections,
        OutputCropZones,
    )
else:
    from components.RollUpList.src.models.PackageModel import (
        PackageModel,
        PackageConfigs,
        ConfigExecutor,
        RollUpListOutputs,
        RollUpListResponse,
        RollUpList,
        OutputRolledUpDetections,
        OutputCropZones,
    )


def build_response(context):
    output_rolled_up_detections = OutputRolledUpDetections(
        value=context.output_rolled_up_detections
    )

    output_crop_zones = OutputCropZones(
        value=context.output_crop_zones
    )

    outputs = RollUpListOutputs(
        outputRolledUpDetections=output_rolled_up_detections,
        outputCropZones=output_crop_zones,
    )

    response = RollUpListResponse(
        outputs=outputs
    )

    executor = RollUpList(
        value=response
    )

    config_executor = ConfigExecutor(
        value=executor
    )

    package_configs = PackageConfigs(
        executor=config_executor
    )

    package = PackageHelper(
        packageModel=PackageModel,
        packageConfigs=package_configs,
    )

    return package.build_model(
        context
    )