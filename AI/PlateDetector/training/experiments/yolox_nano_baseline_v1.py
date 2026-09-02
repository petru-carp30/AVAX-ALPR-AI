from pathlib import Path

import torch.nn as nn

from yolox.data import COCODataset, TrainTransform, ValTransform
from yolox.exp import Exp as YOLOXExp
from yolox.models import YOLOPAFPN, YOLOX, YOLOXHead


PROJECT_ROOT = Path(__file__).resolve().parents[4]

ADAPTER_ROOT = (
    PROJECT_ROOT
    / "AI"
    / "PlateDetector"
    / "training"
    / "datasets"
    / "baseline_v1_coco"
)

RUNS_ROOT = (
    PROJECT_ROOT
    / "AI"
    / "PlateDetector"
    / "training"
    / "runs"
)

EXP_NAME = "yolox_nano_baseline_v1"
TEST_BLOCKED_ANNOTATION = "__TEST_BLOCKED_DO_NOT_USE__.json"


class Exp(YOLOXExp):
    def __init__(self):
        super().__init__()

        self.num_classes = 1
        self.depth = 0.33
        self.width = 0.25

        self.seed = 42

        self.input_size = (416, 416)
        self.test_size = (416, 416)
        self.multiscale_range = 0
        self.random_size = (13, 13)

        self.data_dir = str(ADAPTER_ROOT)
        self.train_ann = "instances_train.json"
        self.val_ann = "instances_val.json"
        self.test_ann = TEST_BLOCKED_ANNOTATION

        self.output_dir = str(RUNS_ROOT)
        self.exp_name = EXP_NAME

        self.max_epoch = 120
        self.warmup_epochs = 5
        self.no_aug_epochs = 15

        self.basic_lr_per_img = 0.01 / 64.0
        self.scheduler = "yoloxwarmcos"
        self.min_lr_ratio = 0.05

        self.mosaic_prob = 0.5
        self.mosaic_scale = (0.5, 1.5)
        self.enable_mixup = False
        self.mixup_prob = 0.0

        self.hsv_prob = 1.0
        self.flip_prob = 0.5
        self.degrees = 10.0
        self.translate = 0.1
        self.shear = 2.0

        self.ema = True
        self.weight_decay = 5e-4
        self.momentum = 0.9

        self.data_num_workers = 4
        self.print_interval = 20
        self.eval_interval = 5
        self.save_history_ckpt = False

        self.test_conf = 0.01
        self.nmsthre = 0.65

        self._validate_avax_contract()

    def _validate_avax_contract(self):
        expected_data_dir = str(ADAPTER_ROOT)

        if self.num_classes != 1:
            raise RuntimeError(
                f"AVAX Exp contract violation: num_classes={self.num_classes}, expected=1"
            )

        if self.depth != 0.33:
            raise RuntimeError(
                f"AVAX Exp contract violation: depth={self.depth}, expected=0.33"
            )

        if self.width != 0.25:
            raise RuntimeError(
                f"AVAX Exp contract violation: width={self.width}, expected=0.25"
            )

        if self.input_size != (416, 416):
            raise RuntimeError(
                f"AVAX Exp contract violation: input_size={self.input_size}"
            )

        if self.test_size != (416, 416):
            raise RuntimeError(
                f"AVAX Exp contract violation: test_size={self.test_size}"
            )

        if self.data_dir != expected_data_dir:
            raise RuntimeError(
                f"AVAX Exp contract violation: data_dir={self.data_dir}"
            )

        if self.train_ann != "instances_train.json":
            raise RuntimeError(
                f"AVAX Exp contract violation: train_ann={self.train_ann}"
            )

        if self.val_ann != "instances_val.json":
            raise RuntimeError(
                f"AVAX Exp contract violation: val_ann={self.val_ann}"
            )

        if self.test_ann != TEST_BLOCKED_ANNOTATION:
            raise RuntimeError(
                f"AVAX Exp contract violation: test_ann={self.test_ann}"
            )

    def get_model(self, sublinear=False):
        def init_yolo(module):
            for layer in module.modules():
                if isinstance(layer, nn.BatchNorm2d):
                    layer.eps = 1e-3
                    layer.momentum = 0.03

        if self.num_classes != 1 or self.depth != 0.33 or self.width != 0.25:
            raise RuntimeError(
                "Refusing to construct model because AVAX YOLOX-Nano architecture contract is invalid"
            )

        if "model" not in self.__dict__:
            in_channels = [256, 512, 1024]

            backbone = YOLOPAFPN(
                self.depth,
                self.width,
                in_channels=in_channels,
                act=self.act,
                depthwise=True,
            )

            head = YOLOXHead(
                self.num_classes,
                self.width,
                in_channels=in_channels,
                act=self.act,
                depthwise=True,
            )

            self.model = YOLOX(backbone, head)

        self.model.apply(init_yolo)
        self.model.head.initialize_biases(1e-2)
        return self.model

    def get_dataset(self, cache=False, cache_type="ram"):
        return COCODataset(
            data_dir=self.data_dir,
            json_file=self.train_ann,
            name="train",
            img_size=self.input_size,
            preproc=TrainTransform(
                max_labels=50,
                flip_prob=self.flip_prob,
                hsv_prob=self.hsv_prob,
            ),
            cache=cache,
            cache_type=cache_type,
        )

    def get_eval_dataset(self, **kwargs):
        if kwargs.get("testdev", False):
            raise RuntimeError(
                "TEST is reserved for final evaluation and cannot be used during model development"
            )

        legacy = kwargs.get("legacy", False)

        return COCODataset(
            data_dir=self.data_dir,
            json_file=self.val_ann,
            name="val",
            img_size=self.test_size,
            preproc=ValTransform(legacy=legacy),
        )

    def eval(self, model, evaluator, is_distributed, half=False, return_outputs=False):
        import yolox.layers
        from pycocotools.cocoeval import COCOeval as StandardCOCOeval

        original_coco_eval = yolox.layers.COCOeval_opt

        try:
            yolox.layers.COCOeval_opt = StandardCOCOeval
            return evaluator.evaluate(
                model,
                is_distributed,
                half,
                return_outputs=return_outputs,
            )
        finally:
            yolox.layers.COCOeval_opt = original_coco_eval