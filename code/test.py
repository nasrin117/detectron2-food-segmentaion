import os
import sys
import click
import pprint
from detectron2.config import get_cfg
from detectron2.utils.visualizer import ColorMode
from detectron2.engine.defaults import DefaultPredictor 
from detectron2 import model_zoo
import cv2
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.utils.visualizer import ColorMode
from detectron2.utils.visualizer import Visualizer
from detectron2.data.datasets import register_coco_instances


@click.command()
@click.option("--model_path", required=True, help="Path 2 pre-trained model")
@click.option("--config", required=True, help="Path 2 config YAML file")
@click.option("--image_path", default=None, help="Path 2 image")

def test_model(model_path, config, image_path):

    cfg = get_cfg()
    cfg.merge_from_file(config)
    cfg.MODEL.WEIGHTS = model_path
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.8
    predictor = DefaultPredictor(cfg) 
 
    im = cv2.imread(image_path)
    outputs = predictor(im)
    v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    image_save_path = "/detectron2_repo/image.jpg"
    cv2.imwrite(image_save_path, out.get_image()[:, :, ::-1])
    

    
if __name__ == "__main__":
    test_model()

