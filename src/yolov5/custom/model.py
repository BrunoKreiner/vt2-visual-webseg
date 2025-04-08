
import sys
from pathlib import Path
import torch
import torch.nn as nn
FILE = Path(__file__).resolve()
ROOT = FILE.parents[1] / 'yolov5'
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from models.yolo import DetectionModel, BaseModel, parse_model, Detect, Segment, check_anchor_order, initialize_weights
from utils.general import LOGGER
from copy import deepcopy

class YOLOWS(DetectionModel):
    """YOLOv5 model with Weighted Self-attention modifications"""
    def __init__(self, cfg="yolov5s.yaml", ch=3, nc=None, anchors=None):
        super(BaseModel, self).__init__()  # Initialize nn.Module

        self.skip_features = None
        
        # The following is copied directly from DetectionModel.__init__
        if isinstance(cfg, dict):
            self.yaml = cfg  # model dict
        else:  # is *.yaml
            import yaml  # for torch hub
            self.yaml_file = Path(cfg).name
            with open(cfg, encoding="ascii", errors="ignore") as f:
                self.yaml = yaml.safe_load(f)  # model dict

        # Define model
        ch = self.yaml["ch"] = self.yaml.get("ch", ch)  # input channels
        if nc and nc != self.yaml["nc"]:
            LOGGER.info(f"Overriding model.yaml nc={self.yaml['nc']} with nc={nc}")
            self.yaml["nc"] = nc  # override yaml value
        if anchors:
            LOGGER.info(f"Overriding model.yaml anchors with anchors={anchors}")
            self.yaml["anchors"] = round(anchors)  # override yaml value
            
        self.model, self.save = parse_model(deepcopy(self.yaml), ch=[ch])  # model, savelist
        self.names = [str(i) for i in range(self.yaml["nc"])]  # default names
        self.inplace = self.yaml.get("inplace", True)

        # Build strides, anchors
        m = self.model[-1]  # Detect()
        if isinstance(m, (Detect, Segment)):
            def _forward(x):
                return self.forward(x)[0] if isinstance(m, Segment) else self.forward(x)
                
            s = 256  # 2x min stride
            m.inplace = self.inplace
            m.stride = torch.tensor([s / x.shape[-2] for x in _forward(torch.zeros(1, ch, s, s))])  # forward
            check_anchor_order(m)
            m.anchors /= m.stride.view(-1, 1, 1)
            self.stride = m.stride
            self._initialize_biases()  # only run once

        # Init weights, biases
        initialize_weights(self)
        self.info()
        LOGGER.info("")

    def forward(self, x, augment=False, profile=False, visualize=False):
        """Forward pass, inheriting all YOLOv5's training/inference modes"""
        return super().forward(x, augment=augment, profile=profile, visualize=visualize)

    def save_layer5_features(self, module, inp, output):
        self.skip_features = output.clone()