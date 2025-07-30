import cv2
import numpy as np
import supervision as sv
import torch
import torchvision
from groundingdino.util.inference import Model
from segment_anything import SamPredictor, sam_model_registry
from typing import List

from aera_semi_autonomous.config.constants import (
    DEVICE,
    GROUNDING_DINO_CONFIG_PATH,
    GROUNDING_DINO_CHECKPOINT_PATH,
    SAM_ENCODER_VERSION,
    SAM_CHECKPOINT_PATH,
    BOX_THRESHOLD,
    TEXT_THRESHOLD,
    NMS_THRESHOLD,
)


def segment(
    sam_predictor: SamPredictor, image: np.ndarray, xyxy: np.ndarray
) -> np.ndarray:
    """Prompting SAM with detected boxes"""
    sam_predictor.set_image(image)
    result_masks = []
    for box in xyxy:
        masks, scores, _ = sam_predictor.predict(box=box, multimask_output=True)
        index = np.argmax(scores)
        result_masks.append(masks[index])
    return np.array(result_masks)


class ObjectDetector:
    def __init__(self, logger):
        self.logger = logger
        
        # Initialize GroundingDINO model
        self.grounding_dino_model = Model(
            model_config_path=GROUNDING_DINO_CONFIG_PATH,
            model_checkpoint_path=GROUNDING_DINO_CHECKPOINT_PATH,
            device=DEVICE,
        )
        
        # Initialize SAM model
        self.sam = sam_model_registry[SAM_ENCODER_VERSION](
            checkpoint=SAM_CHECKPOINT_PATH
        )
        self.sam.to(device=DEVICE)
        self.sam_predictor = SamPredictor(self.sam)

    def detect_objects(
        self, image: np.ndarray, object_classes: List[str]
    ) -> sv.Detections:
        """Detect objects in the image using GroundingDINO and segment them with SAM."""
        self.logger.info(f"Detecting objects of classes: {object_classes}")

        detections: sv.Detections = self.grounding_dino_model.predict_with_classes(
            image=image,
            classes=object_classes,
            box_threshold=BOX_THRESHOLD,
            text_threshold=TEXT_THRESHOLD,
        )
        
        if len(detections.xyxy) == 0:  # Check after NMS
            self.logger.warn("No detections.")
            detections.mask = np.array([])
            return detections

        nms_idx = (
            torchvision.ops.nms(
                torch.from_numpy(detections.xyxy),
                torch.from_numpy(detections.confidence),
                NMS_THRESHOLD,
            )
            .numpy()
            .tolist()
        )

        detections.xyxy = detections.xyxy[nms_idx]
        detections.confidence = detections.confidence[nms_idx]
        detections.class_id = detections.class_id[nms_idx]

        detections.mask = segment(
            sam_predictor=self.sam_predictor,
            image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
            xyxy=detections.xyxy,
        )

        self.logger.info(f"Detected {detections}.")
        self.logger.info(f"detection confidence: {detections.confidence}")
        return detections
