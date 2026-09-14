import os
import sys

from copy import deepcopy
from typing import List

import numpy as np


sys.path.append(
    os.path.join(
        os.path.dirname(__file__),
        "../../../../",
    )
)


from sdks.novavision.src.base.component import Component
from sdks.novavision.src.helper.executor import Executor


if __package__:
    from ..models.PackageModel import PackageModel
    from ..utils.response import build_response
else:
    from components.RollUpList.src.models.PackageModel import PackageModel
    from components.RollUpList.src.utils.response import build_response


VALID_CONFIDENCE_STRATEGIES = {
    "max",
    "mean",
    "min",
}


class RollUpList(Component):

    def __init__(
        self,
        request,
        bootstrap,
    ):
        super().__init__(
            request,
            bootstrap,
        )

        self.request.model = PackageModel(
            **self.request.data
        )

        # -----------------------------------------------------------------
        # Inputs
        # -----------------------------------------------------------------

        self.input_parent_detections = self.request.get_param(
            "inputParentDetections"
        )

        self.input_child_detections = self.request.get_param(
            "inputChildDetections"
        )

        # -----------------------------------------------------------------
        # Configs
        # -----------------------------------------------------------------

        confidence_strategy = self.safe_get_param(
            "ConfigConfidenceStrategy",
            "configConfidenceStrategy",
            default="max",
        )

        overlap_threshold = self.safe_get_param(
            "ConfigOverlapThreshold",
            "configOverlapThreshold",
            default=0.0,
        )

        keypoint_merge_threshold = self.safe_get_param(
            "ConfigKeypointMergeThreshold",
            "configKeypointMergeThreshold",
            default=10.0,
        )

        self.confidence_strategy = (
            self.normalize_confidence_strategy(
                confidence_strategy
            )
        )

        self.overlap_threshold = self.parse_float(
            overlap_threshold,
            default=0.0,
            minimum=0.0,
            maximum=1.0,
        )

        self.keypoint_merge_threshold = self.parse_float(
            keypoint_merge_threshold,
            default=10.0,
            minimum=0.0,
        )

        # -----------------------------------------------------------------
        # Outputs
        # -----------------------------------------------------------------

        self.output_rolled_up_detections = []
        self.output_crop_zones = []

    # ---------------------------------------------------------------------
    # Bootstrap
    # ---------------------------------------------------------------------

    @staticmethod
    def bootstrap(
        config: dict = None,
    ) -> dict:
        return {}

    # ---------------------------------------------------------------------
    # Config Helpers
    # ---------------------------------------------------------------------

    def safe_get_param(
        self,
        *names,
        default=None,
    ):

        for name in names:

            try:
                value = self.request.get_param(
                    name
                )
            except Exception:
                continue

            if value is not None:
                return value

        return default

    @staticmethod
    def is_placeholder(
        value,
    ) -> bool:

        if not isinstance(
            value,
            str,
        ):
            return False

        value = value.strip()

        return (
            value.startswith("{{")
            and value.endswith("}}")
        )

    @staticmethod
    def unwrap_config_value(
        value,
        default=None,
    ):

        current = value

        for _ in range(6):

            if current is None:
                return default

            if isinstance(
                current,
                (int, float, bool),
            ):
                return current

            if isinstance(
                current,
                str,
            ):

                current = current.strip()

                if not current:
                    return default

                if RollUpList.is_placeholder(
                    current
                ):
                    return default

                return current

            if isinstance(
                current,
                dict,
            ):

                option_name = current.get(
                    "name"
                )

                if (
                    isinstance(
                        option_name,
                        str,
                    )
                    and option_name.lower()
                    in VALID_CONFIDENCE_STRATEGIES
                ):
                    return option_name.lower()

                if "value" not in current:
                    return default

                current = current.get(
                    "value"
                )

                continue

            option_name = getattr(
                current,
                "name",
                None,
            )

            if (
                isinstance(
                    option_name,
                    str,
                )
                and option_name.lower()
                in VALID_CONFIDENCE_STRATEGIES
            ):
                return option_name.lower()

            if hasattr(
                current,
                "value",
            ):
                current = current.value
                continue

            return default

        return default

    @staticmethod
    def normalize_confidence_strategy(
        value,
    ) -> str:

        value = RollUpList.unwrap_config_value(
            value,
            default="max",
        )

        value = str(
            value
        ).lower()

        if value not in VALID_CONFIDENCE_STRATEGIES:

            raise ValueError(
                "Confidence Strategy must be "
                "'max', 'mean' or 'min'."
            )

        return value

    @staticmethod
    def parse_float(
        value,
        default,
        minimum=None,
        maximum=None,
    ) -> float:

        value = RollUpList.unwrap_config_value(
            value,
            default=default,
        )

        try:
            value = float(
                value
            )

        except (
            TypeError,
            ValueError,
        ) as error:

            raise ValueError(
                "Config value must be numeric."
            ) from error

        if (
            minimum is not None
            and value < minimum
        ):
            raise ValueError(
                f"Config value cannot be less than {minimum}."
            )

        if (
            maximum is not None
            and value > maximum
        ):
            raise ValueError(
                f"Config value cannot be greater than {maximum}."
            )

        return value

    # ---------------------------------------------------------------------
    # Generic Helpers
    # ---------------------------------------------------------------------

    @staticmethod
    def get_value(
        obj,
        key,
        default=None,
    ):

        if obj is None:
            return default

        if isinstance(
            obj,
            dict,
        ):
            return obj.get(
                key,
                default,
            )

        return getattr(
            obj,
            key,
            default,
        )

    @staticmethod
    def detection_to_dict(
        detection,
    ) -> dict:

        if isinstance(
            detection,
            dict,
        ):
            return deepcopy(
                detection
            )

        if hasattr(
            detection,
            "model_dump",
        ):
            return deepcopy(
                detection.model_dump()
            )

        if hasattr(
            detection,
            "dict",
        ):
            return deepcopy(
                detection.dict()
            )

        raise TypeError(
            "Unsupported Detection type."
        )

    # ---------------------------------------------------------------------
    # Child Detection Input Normalization
    # ---------------------------------------------------------------------

    @staticmethod
    def normalize_child_detection_groups(
        child_detections,
        parent_count: int,
    ) -> List[List]:

        items = list(
            child_detections
            or []
        )

        # No child detections means every parent crop
        # produced an empty detection list.
        if not items:
            return [
                []
                for _ in range(parent_count)
            ]

        group_flags = [
            isinstance(
                item,
                (list, tuple),
            )
            for item in items
        ]

        # Expected Roll-Up format:
        #
        # [
        #     [Detection, Detection],
        #     [Detection],
        #     ...
        # ]
        if all(
            group_flags
        ):
            return [
                list(
                    group
                    or []
                )
                for group in items
            ]

        # A mixture of Detection and List[Detection]
        # is not a valid input representation.
        if any(
            group_flags
        ):
            raise ValueError(
                "inputChildDetections cannot mix "
                "Detection values and detection groups."
            )

        # NovaVision may return a flat List[Detection]
        # when a single parent crop is processed:
        #
        # [
        #     Detection,
        #     Detection,
        # ]
        #
        # In that case all detections belong to the
        # only parent crop.
        if parent_count == 1:
            return [
                items
            ]

        # With multiple parents a flat detection list
        # does not contain enough information to determine
        # which child belongs to which parent crop.
        raise ValueError(
            "inputChildDetections was received as a flat "
            "List[Detection], but multiple parent detections "
            "exist. Child detections must be grouped per "
            "parent crop as List[List[Detection]]."
        )

    # ---------------------------------------------------------------------
    # Bounding Box Helpers
    # ---------------------------------------------------------------------

    @staticmethod
    def get_bbox(
        detection,
    ):

        bbox = RollUpList.get_value(
            detection,
            "boundingBox",
        )

        if bbox is None:
            raise ValueError(
                "Detection does not contain boundingBox."
            )

        return bbox

    @staticmethod
    def bbox_to_xyxy(
        detection,
    ) -> np.ndarray:

        bbox = RollUpList.get_bbox(
            detection
        )

        left = RollUpList.get_value(
            bbox,
            "left",
        )

        top = RollUpList.get_value(
            bbox,
            "top",
        )

        width = RollUpList.get_value(
            bbox,
            "width",
        )

        height = RollUpList.get_value(
            bbox,
            "height",
        )

        if (
            left is None
            or top is None
            or width is None
            or height is None
        ):
            raise ValueError(
                "boundingBox must contain "
                "left, top, width and height."
            )

        left = float(
            left
        )

        top = float(
            top
        )

        width = float(
            width
        )

        height = float(
            height
        )

        return np.array(
            [
                left,
                top,
                left + width,
                top + height,
            ],
            dtype=float,
        )

    @staticmethod
    def set_bbox_from_xyxy(
        detection: dict,
        bbox,
    ) -> dict:

        left = float(
            bbox[0]
        )

        top = float(
            bbox[1]
        )

        right = float(
            bbox[2]
        )

        bottom = float(
            bbox[3]
        )

        detection["boundingBox"] = {
            "left": left,
            "top": top,
            "width": right - left,
            "height": bottom - top,
        }

        return detection

    # ---------------------------------------------------------------------
    # Coordinate Transform
    # ---------------------------------------------------------------------

    @staticmethod
    def shift_detection(
        detection,
        x_offset: float,
        y_offset: float,
    ) -> dict:

        result = RollUpList.detection_to_dict(
            detection
        )

        bbox = RollUpList.bbox_to_xyxy(
            result
        )

        bbox += np.array(
            [
                x_offset,
                y_offset,
                x_offset,
                y_offset,
            ],
            dtype=float,
        )

        RollUpList.set_bbox_from_xyxy(
            result,
            bbox,
        )

        keypoints = result.get(
            "keyPoints"
        )

        if keypoints:

            shifted_keypoints = []

            for keypoint in keypoints:

                keypoint = deepcopy(
                    keypoint
                )

                if "cx" in keypoint:
                    keypoint["cx"] = (
                        float(
                            keypoint["cx"]
                        )
                        + x_offset
                    )

                if "cy" in keypoint:
                    keypoint["cy"] = (
                        float(
                            keypoint["cy"]
                        )
                        + y_offset
                    )

                shifted_keypoints.append(
                    keypoint
                )

            result["keyPoints"] = (
                shifted_keypoints
            )

        return result

    # ---------------------------------------------------------------------
    # Crop Zones
    # ---------------------------------------------------------------------

    @staticmethod
    def create_crop_zone(
        parent_detection,
    ) -> List[List[float]]:

        bbox = RollUpList.bbox_to_xyxy(
            parent_detection
        )

        (
            x_min,
            y_min,
            x_max,
            y_max,
        ) = bbox

        return [
            [
                float(x_min),
                float(y_min),
            ],
            [
                float(x_max),
                float(y_min),
            ],
            [
                float(x_max),
                float(y_max),
            ],
            [
                float(x_min),
                float(y_max),
            ],
        ]

    # ---------------------------------------------------------------------
    # Detection Metadata
    # ---------------------------------------------------------------------

    @staticmethod
    def get_class_key(
        detection,
    ):

        class_id = RollUpList.get_value(
            detection,
            "classId",
        )

        class_label = RollUpList.get_value(
            detection,
            "classLabel",
        )

        return (
            class_id,
            class_label,
        )

    @staticmethod
    def get_confidence(
        detection,
    ) -> float:

        value = RollUpList.get_value(
            detection,
            "confidence",
            0.0,
        )

        if value is None:
            return 0.0

        return float(
            value
        )

    def merge_confidences(
        self,
        detections: List,
    ) -> float:

        confidences = [
            self.get_confidence(
                detection
            )
            for detection in detections
        ]

        if not confidences:
            return 0.0

        if self.confidence_strategy == "max":
            return float(
                max(confidences)
            )

        if self.confidence_strategy == "mean":
            return float(
                np.mean(
                    confidences
                )
            )

        return float(
            min(confidences)
        )

    # ---------------------------------------------------------------------
    # IoU
    # ---------------------------------------------------------------------

    @staticmethod
    def calculate_iou(
        bbox_a,
        bbox_b,
    ):

        x1 = max(
            bbox_a[0],
            bbox_b[0],
        )

        y1 = max(
            bbox_a[1],
            bbox_b[1],
        )

        x2 = min(
            bbox_a[2],
            bbox_b[2],
        )

        y2 = min(
            bbox_a[3],
            bbox_b[3],
        )

        intersection_width = max(
            0.0,
            x2 - x1,
        )

        intersection_height = max(
            0.0,
            y2 - y1,
        )

        intersection = (
            intersection_width
            * intersection_height
        )

        if intersection <= 0.0:
            return (
                0.0,
                0.0,
            )

        area_a = max(
            0.0,
            (
                bbox_a[2]
                - bbox_a[0]
            ),
        ) * max(
            0.0,
            (
                bbox_a[3]
                - bbox_a[1]
            ),
        )

        area_b = max(
            0.0,
            (
                bbox_b[2]
                - bbox_b[0]
            ),
        ) * max(
            0.0,
            (
                bbox_b[3]
                - bbox_b[1]
            ),
        )

        union = (
            area_a
            + area_b
            - intersection
        )

        if union <= 0.0:
            return (
                0.0,
                intersection,
            )

        return (
            float(
                intersection
                / union
            ),
            float(
                intersection
            ),
        )

    # ---------------------------------------------------------------------
    # Keypoint Helpers
    # ---------------------------------------------------------------------

    @staticmethod
    def get_keypoints(
        detection,
    ) -> List:

        keypoints = RollUpList.get_value(
            detection,
            "keyPoints",
        )

        return keypoints or []

    @staticmethod
    def has_keypoints(
        detection,
    ) -> bool:

        return bool(
            RollUpList.get_keypoints(
                detection
            )
        )

    @staticmethod
    def keypoint_distance(
        detection_a,
        detection_b,
    ):

        keypoints_a = RollUpList.get_keypoints(
            detection_a
        )

        keypoints_b = RollUpList.get_keypoints(
            detection_b
        )

        if (
            not keypoints_a
            or not keypoints_b
            or len(keypoints_a)
            != len(keypoints_b)
        ):
            return None

        distances = []

        for (
            point_a,
            point_b,
        ) in zip(
            keypoints_a,
            keypoints_b,
        ):

            ax = RollUpList.get_value(
                point_a,
                "cx",
            )

            ay = RollUpList.get_value(
                point_a,
                "cy",
            )

            bx = RollUpList.get_value(
                point_b,
                "cx",
            )

            by = RollUpList.get_value(
                point_b,
                "cy",
            )

            if None in (
                ax,
                ay,
                bx,
                by,
            ):
                return None

            distance = np.hypot(
                float(ax)
                - float(bx),
                float(ay)
                - float(by),
            )

            distances.append(
                distance
            )

        if not distances:
            return None

        return float(
            np.mean(
                distances
            )
        )

    def should_merge_keypoints(
        self,
        detection_a,
        detection_b,
    ) -> bool:

        distance = self.keypoint_distance(
            detection_a,
            detection_b,
        )

        if distance is None:
            return False

        return (
            distance
            < self.keypoint_merge_threshold
        )

    def merge_keypoint_group(
        self,
        detections: List,
    ) -> dict:

        result = self.detection_to_dict(
            detections[0]
        )

        result["confidence"] = (
            self.merge_confidences(
                detections
            )
        )

        bboxes = np.array(
            [
                self.bbox_to_xyxy(
                    detection
                )
                for detection in detections
            ],
            dtype=float,
        )

        averaged_bbox = np.mean(
            bboxes,
            axis=0,
        )

        self.set_bbox_from_xyxy(
            result,
            averaged_bbox,
        )

        first_keypoints = self.get_keypoints(
            detections[0]
        )

        merged_keypoints = []

        for keypoint_index in range(
            len(first_keypoints)
        ):

            point = (
                self.detection_to_dict(
                    first_keypoints[
                        keypoint_index
                    ]
                )
                if not isinstance(
                    first_keypoints[
                        keypoint_index
                    ],
                    dict,
                )
                else deepcopy(
                    first_keypoints[
                        keypoint_index
                    ]
                )
            )

            x_values = []
            y_values = []
            confidence_values = []

            for detection in detections:

                keypoints = self.get_keypoints(
                    detection
                )

                if keypoint_index >= len(
                    keypoints
                ):
                    continue

                keypoint = keypoints[
                    keypoint_index
                ]

                cx = self.get_value(
                    keypoint,
                    "cx",
                )

                cy = self.get_value(
                    keypoint,
                    "cy",
                )

                confidence = self.get_value(
                    keypoint,
                    "confidence",
                )

                if cx is not None:
                    x_values.append(
                        float(cx)
                    )

                if cy is not None:
                    y_values.append(
                        float(cy)
                    )

                if confidence is not None:
                    confidence_values.append(
                        float(confidence)
                    )

            if x_values:
                point["cx"] = float(
                    np.mean(
                        x_values
                    )
                )

            if y_values:
                point["cy"] = float(
                    np.mean(
                        y_values
                    )
                )

            if confidence_values:
                point["confidence"] = float(
                    np.mean(
                        confidence_values
                    )
                )

            merged_keypoints.append(
                point
            )

        result["keyPoints"] = (
            merged_keypoints
        )

        return result

    def merge_keypoint_detections(
        self,
        detections: List,
    ) -> List[dict]:

        if not detections:
            return []

        merged = []
        used = set()

        for i, detection_a in enumerate(
            detections
        ):

            if i in used:
                continue

            group = [
                detection_a
            ]

            used.add(
                i
            )

            for j in range(
                i + 1,
                len(detections),
            ):

                if j in used:
                    continue

                detection_b = detections[
                    j
                ]

                if self.should_merge_keypoints(
                    detection_a,
                    detection_b,
                ):

                    group.append(
                        detection_b
                    )

                    used.add(
                        j
                    )

            merged.append(
                self.merge_keypoint_group(
                    group
                )
            )

        return merged

    # ---------------------------------------------------------------------
    # Bounding Box Merge
    # ---------------------------------------------------------------------

    def should_merge_bbox(
        self,
        detection_a,
        detection_b,
    ) -> bool:

        bbox_a = self.bbox_to_xyxy(
            detection_a
        )

        bbox_b = self.bbox_to_xyxy(
            detection_b
        )

        (
            iou,
            intersection,
        ) = self.calculate_iou(
            bbox_a,
            bbox_b,
        )

        if self.overlap_threshold <= 0.0:
            return (
                intersection
                > 0.0
            )

        return (
            iou
            >= self.overlap_threshold
        )

    def merge_bbox_group(
        self,
        detections: List,
    ) -> dict:

        result = self.detection_to_dict(
            detections[0]
        )

        bboxes = np.array(
            [
                self.bbox_to_xyxy(
                    detection
                )
                for detection in detections
            ],
            dtype=float,
        )

        merged_bbox = np.array(
            [
                bboxes[:, 0].min(),
                bboxes[:, 1].min(),
                bboxes[:, 2].max(),
                bboxes[:, 3].max(),
            ],
            dtype=float,
        )

        self.set_bbox_from_xyxy(
            result,
            merged_bbox,
        )

        result["confidence"] = (
            self.merge_confidences(
                detections
            )
        )

        return result

    def merge_bbox_detections(
        self,
        detections: List,
    ) -> List[dict]:

        if not detections:
            return []

        count = len(
            detections
        )

        parent = list(
            range(count)
        )

        def find(index):

            root = index

            while parent[root] != root:
                root = parent[
                    root
                ]

            while parent[index] != root:

                next_index = parent[
                    index
                ]

                parent[
                    index
                ] = root

                index = next_index

            return root

        def union(
            first,
            second,
        ):

            first_root = find(
                first
            )

            second_root = find(
                second
            )

            if first_root != second_root:
                parent[
                    second_root
                ] = first_root

        for i in range(
            count
        ):

            for j in range(
                i + 1,
                count,
            ):

                if self.should_merge_bbox(
                    detections[i],
                    detections[j],
                ):

                    union(
                        i,
                        j,
                    )

        groups = {}

        for index in range(
            count
        ):

            root = find(
                index
            )

            groups.setdefault(
                root,
                []
            ).append(
                detections[
                    index
                ]
            )

        return [
            self.merge_bbox_group(
                group
            )
            for group in groups.values()
        ]

    # ---------------------------------------------------------------------
    # Detection Group Merge
    # ---------------------------------------------------------------------

    def merge_detection_group(
        self,
        detections: List,
    ) -> List[dict]:

        if not detections:
            return []

        keypoint_detections = [
            detection
            for detection in detections
            if self.has_keypoints(
                detection
            )
        ]

        non_keypoint_detections = [
            detection
            for detection in detections
            if not self.has_keypoints(
                detection
            )
        ]

        results = []

        if keypoint_detections:

            results.extend(
                self.merge_keypoint_detections(
                    keypoint_detections
                )
            )

        if non_keypoint_detections:

            results.extend(
                self.merge_bbox_detections(
                    non_keypoint_detections
                )
            )

        return results

    # ---------------------------------------------------------------------
    # Run
    # ---------------------------------------------------------------------

    def run(
        self,
    ):

        parent_detections = list(
            self.input_parent_detections
            or []
        )

        child_detection_groups = (
            self.normalize_child_detection_groups(
                child_detections=self.input_child_detections,
                parent_count=len(
                    parent_detections
                ),
            )
        )

        # One child detection group must correspond
        # to each parent crop.
        if (
            len(
                parent_detections
            )
            != len(
                child_detection_groups
            )
        ):

            raise ValueError(
                "Number of parent detections must match "
                "number of child detection groups."
            )

        transformed_detections = []
        crop_zones = []

        # -------------------------------------------------------------
        # Transform child coordinates to parent coordinates
        # -------------------------------------------------------------

        for (
            index,
            parent_detection,
        ) in enumerate(
            parent_detections
        ):

            parent_bbox = self.bbox_to_xyxy(
                parent_detection
            )

            x_offset = float(
                parent_bbox[
                    0
                ]
            )

            y_offset = float(
                parent_bbox[
                    1
                ]
            )

            crop_zones.append(
                self.create_crop_zone(
                    parent_detection
                )
            )

            child_detections = (
                child_detection_groups[
                    index
                ]
                or []
            )

            for child_detection in child_detections:

                transformed_detection = (
                    self.shift_detection(
                        detection=child_detection,
                        x_offset=x_offset,
                        y_offset=y_offset,
                    )
                )

                transformed_detections.append(
                    transformed_detection
                )

        # -------------------------------------------------------------
        # Group detections by class
        # -------------------------------------------------------------

        detections_by_class = {}

        for detection in transformed_detections:

            class_key = self.get_class_key(
                detection
            )

            detections_by_class.setdefault(
                class_key,
                []
            ).append(
                detection
            )

        # -------------------------------------------------------------
        # Merge detections
        # -------------------------------------------------------------

        rolled_up_detections = []

        for detections in detections_by_class.values():

            rolled_up_detections.extend(
                self.merge_detection_group(
                    detections
                )
            )

        # -------------------------------------------------------------
        # Outputs
        # -------------------------------------------------------------

        self.output_rolled_up_detections = (
            rolled_up_detections
        )

        self.output_crop_zones = (
            crop_zones
        )

        return build_response(
            context=self
        )


if __name__ == "__main__":
    Executor(
        sys.argv[1]
    ).run()