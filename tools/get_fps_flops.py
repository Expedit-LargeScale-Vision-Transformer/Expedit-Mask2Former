import logging
import numpy as np
from collections import Counter
import tqdm
from fvcore.nn import flop_count_table  # can also try flop_count_str
import torch

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import CfgNode, LazyConfig, get_cfg, instantiate
from detectron2.data import build_detection_test_loader
from detectron2.engine import default_argument_parser
from detectron2.modeling import build_model
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.utils.analysis import (
    FlopCountAnalysis,
    activation_count_operators,
    parameter_count_table,
)
from detectron2.utils.logger import setup_logger

# fmt: off
import os
import sys
import time
sys.path.insert(1, os.path.join(sys.path[0], '..'))
# fmt: on

import warnings

warnings.filterwarnings("ignore", category=UserWarning)

from mask2former import add_maskformer2_config

logger = logging.getLogger("detectron2")


def setup(args):
    if args.config_file.endswith(".yaml"):
        cfg = get_cfg()
        add_deeplab_config(cfg)
        add_maskformer2_config(cfg)
        cfg.merge_from_file(args.config_file)
        cfg.DATALOADER.NUM_WORKERS = 0
        cfg.merge_from_list(args.opts)
        cfg.freeze()
    else:
        cfg = LazyConfig.load(args.config_file)
        cfg = LazyConfig.apply_overrides(cfg, args.opts)
    setup_logger(name="fvcore")
    setup_logger()
    return cfg


if __name__ == "__main__":
    parser = default_argument_parser(
        epilog="""
Examples:
To show flops and fps of a model:
$ ./get_fps_flops.py --config-file ../configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml \\
    MODEL.WEIGHTS /path/to/model.pkl
"""
    )

    args = parser.parse_args()
    assert not args.eval_only
    assert args.num_gpus == 1
    cfg = setup(args)
    # input_shape = (3, 1152, 1152)
    input_shape = (1, 3, 1152, 1152)

    data_loader = build_detection_test_loader(cfg, cfg.DATASETS.TEST[0])
    model = build_model(cfg)
    # DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)
    model.eval()

    num_warmup = 50
    log_interval = 50
    pure_inf_time = 0
    total_iters = 200
    for idx, data in zip(tqdm.trange(total_iters), data_loader):
        # for idx in tqdm.trange(total_iters):
        # data[0]["image"] = torch.randn(input_shape)
        data = torch.randn(input_shape).cuda()

        torch.cuda.synchronize()
        start_time = time.perf_counter()
        with torch.no_grad():
            # model(data)
            model.backbone(data)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start_time

        if idx >= num_warmup:
            pure_inf_time += elapsed

    fps = (total_iters - num_warmup) / pure_inf_time
    logger.info("Overall fps: {:.2f} img / s".format(fps))
    logger.info("Times per image: {:2f} s".format(1 / fps))
    logger.info("Times of backbone: {:2f} s".format(model.time_backbone / total_iters))
    logger.info("Times of head: {:2f} s".format(model.time_head / total_iters))

    flops = FlopCountAnalysis(model.backbone, data)
    # flops = FlopCountAnalysis(model, data)
    logger.info(
        "Flops table computed from only one input sample:\n" + flop_count_table(flops)
    )
    logger.info(f"Flops: {flops.total() / 1e9} G")
