import os
import click
import torch.cuda
from detectron2.data.datasets import register_coco_instances
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.utils.logger import setup_logger
from detectron2.data import MetadataCatalog
from detectron2.utils.events import TensorboardXWriter


@click.command()
@click.option("--train-ann-path", required=True, help="Path to train annotations JSON file")
@click.option("--train-img-dir", required=True, help="Path to train images directory")
@click.option("--val-ann-path", required=True, help="Path to validation annotations JSON file")
@click.option("--val-img-dir", required=True, help="Path to validation images directory")
@click.option("--config-path", default="COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml", help="Path to config file")
@click.option("--num-classes", default=2, type=int, help="Number of classes")
@click.option("--max-iter", default=300, type=int, help="Maximum number of iterations")
def train_model(train_ann_path, train_img_dir, val_ann_path, val_img_dir, config_path, num_classes, max_iter):

    register_coco_instances("train_dataset", {}, train_ann_path, train_img_dir)
    register_coco_instances("val_dataset", {}, val_ann_path, val_img_dir)
    object_metadata = MetadataCatalog.get("train_dataset")
    print(object_metadata)

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(config_path))
    cfg.DATASETS.TRAIN = ("train_dataset",)
    cfg.DATASETS.TEST = ("val_dataset",)
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(config_path)
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.SOLVER.MAX_ITER = max_iter
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes
    cfg.SOLVER.IMS_PER_BATCH = 1
    cfg.SOLVER.BASE_LR = 0.00025
    cfg.SOLVER.STEPS = []
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
    

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()

    output_config_path = os.path.join(cfg.OUTPUT_DIR, "output.yaml")
    with open(output_config_path, "w") as f:
        f.write(cfg.dump())

    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
    cfg.DATASETS.TEST = ("val_dataset",)
    evaluator = COCOEvaluator("val_dataset", cfg, False, output_dir=cfg.OUTPUT_DIR)
    val_loader = build_detection_test_loader(cfg, "val_dataset")
    inference_on_dataset(trainer.model, val_loader, evaluator)

if __name__ == "__main__":
    train_model()
