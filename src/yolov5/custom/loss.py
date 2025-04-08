import torch
import torch.nn as nn
import sys
from pathlib import Path

# Add YOLOv5 to path
FILE = Path(__file__).resolve()
ROOT = FILE.parents[1] / 'yolov5'
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from utils.general import box_iou
from utils.metrics import bbox_iou
from utils.loss import ComputeLoss, FocalLoss, smooth_BCE
from utils.torch_utils import de_parallel
from utils.general import xywh2xyxy
import logging
from datetime import datetime

def setup_logger():
    # Create logger with current timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    logger = logging.getLogger('EIoU_Debug')
    logger.setLevel(logging.DEBUG)
    
    # Create file handler
    fh = logging.FileHandler(f'eiou_debug_{timestamp}.log')
    fh.setLevel(logging.DEBUG)
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    
    return logger

logger = setup_logger()
def bbox_eiou(box1, box2, xywh=True, eps=1e-7):
    """
    Calculates EIoU loss between two boxes, supporting xywh/xyxy formats.
    Input shapes are box1(1,4) to box2(n,4).
    """
    # Get the coordinates of bounding boxes
    if xywh:  # transform from xywh to xyxy
        (x1, y1, w1, h1), (x2, y2, w2, h2) = box1.chunk(4, -1), box2.chunk(4, -1)
        w1_, h1_, w2_, h2_ = w1 / 2, h1 / 2, w2 / 2, h2 / 2
        b1_x1, b1_x2, b1_y1, b1_y2 = x1 - w1_, x1 + w1_, y1 - h1_, y1 + h1_
        b2_x1, b2_x2, b2_y1, b2_y2 = x2 - w2_, x2 + w2_, y2 - h2_, y2 + h2_
    else:  # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, -1)
        b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, -1)
        w1, h1 = b1_x2 - b1_x1, (b1_y2 - b1_y1).clamp(eps)
        w2, h2 = b2_x2 - b2_x1, (b2_y2 - b2_y1).clamp(eps)

    # Intersection area
    inter = (b1_x2.minimum(b2_x2) - b1_x1.maximum(b2_x1)).clamp(0) * (
        b1_y2.minimum(b2_y2) - b1_y1.maximum(b2_y1)
    ).clamp(0)

    # Union Area
    union = w1 * h1 + w2 * h2 - inter + eps

    # IoU
    iou = (inter / union)

    # Smallest enclosing box width and height
    wc = b1_x2.maximum(b2_x2) - b1_x1.minimum(b2_x1)  # convex width
    hc = b1_y2.maximum(b2_y2) - b1_y1.minimum(b2_y1)  # convex height
    wc_sq = wc ** 2
    hc_sq = hc ** 2

    # Center distance
    rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 + (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4  # center distance squared
    distance_loss = rho2 / (wc_sq + hc_sq + eps)
    logger.debug(f"Distance loss: {distance_loss}")

    # Aspect ratio loss
    width_distance = (w1 - w2) ** 2
    height_distance = (h1 - h2) ** 2
    aspect_loss = width_distance / (wc_sq + eps) + height_distance / (hc_sq + eps)
    logger.debug(f"Aspect loss: {aspect_loss}")

    # Final EIoU loss
    eiou_loss = (1 - iou) + distance_loss + aspect_loss
    logger.debug(f"EIoU loss: {eiou_loss}")
    logger.debug(f"IOU: {iou}")

    return eiou_loss, iou.squeeze()

class CustomComputeLoss:
    """Computes the total loss for YOLOv5 model predictions, including classification, box, and objectness losses."""

    sort_obj_iou = False

    # Compute losses
    def __init__(self, model, autobalance=False):
        """Initializes ComputeLoss with model and autobalance option, autobalances losses if True."""
        device = next(model.parameters()).device  # get model device
        h = model.hyp  # hyperparameters

        # Define criteria
        BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h["cls_pw"]], device=device))
        BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h["obj_pw"]], device=device))

        # Class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
        self.cp, self.cn = smooth_BCE(eps=h.get("label_smoothing", 0.0))  # positive, negative BCE targets

        # Focal loss
        g = h["fl_gamma"]  # focal loss gamma
        if g > 0:
            BCEcls, BCEobj = FocalLoss(BCEcls, g), FocalLoss(BCEobj, g)

        m = de_parallel(model).model[-1]  # Detect() module
        self.balance = {3: [4.0, 1.0, 0.4]}.get(m.nl, [4.0, 1.0, 0.25, 0.06, 0.02])  # P3-P7
        self.ssi = list(m.stride).index(16) if autobalance else 0  # stride 16 index
        self.BCEcls, self.BCEobj, self.gr, self.hyp, self.autobalance = BCEcls, BCEobj, 1.0, h, autobalance
        self.na = m.na  # number of anchors
        self.nc = m.nc  # number of classes
        self.nl = m.nl  # number of layers
        self.anchors = m.anchors
        self.device = device
        self.gamma = h["fl_gamma"]

    """def __call__(self, p, targets):  # predictions, targets
        lcls = torch.zeros(1, device=self.device)  # class loss
        lbox = torch.zeros(1, device=self.device)  # box loss
        lobj = torch.zeros(1, device=self.device)  # object loss
        tcls, tbox, indices, anchors = self.build_targets(p, targets)  # targets

        # Losses
        for i, pi in enumerate(p):  # layer index, layer predictions
            b, a, gj, gi = indices[i]  # image, anchor, gridy, gridx
            tobj = torch.zeros(pi.shape[:4], dtype=pi.dtype, device=self.device)  # target obj

            if n := b.shape[0]:
                # pxy, pwh, _, pcls = pi[b, a, gj, gi].tensor_split((2, 4, 5), dim=1)  # faster, requires torch 1.8.0
                pxy, pwh, _, pcls = pi[b, a, gj, gi].split((2, 2, 1, self.nc), 1)  # target-subset of predictions

                # Regression
                pxy = pxy.sigmoid() * 2 - 0.5
                pwh = (pwh.sigmoid() * 2) ** 2 * anchors[i]
                pbox = torch.cat((pxy, pwh), 1)  # predicted box
                iou = bbox_iou(pbox, tbox[i], CIoU=True).squeeze()  # iou(prediction, target)
                lbox += (1.0 - iou).mean()  # iou loss

                # Objectness
                iou = iou.detach().clamp(0).type(tobj.dtype)
                if self.sort_obj_iou:
                    j = iou.argsort()
                    b, a, gj, gi, iou = b[j], a[j], gj[j], gi[j], iou[j]
                if self.gr < 1:
                    iou = (1.0 - self.gr) + self.gr * iou
                tobj[b, a, gj, gi] = iou  # iou ratio

                # Classification
                if self.nc > 1:  # cls loss (only if multiple classes)
                    t = torch.full_like(pcls, self.cn, device=self.device)  # targets
                    t[range(n), tcls[i]] = self.cp
                    lcls += self.BCEcls(pcls, t)  # BCE

            obji = self.BCEobj(pi[..., 4], tobj)
            lobj += obji * self.balance[i]  # obj loss
            if self.autobalance:
                self.balance[i] = self.balance[i] * 0.9999 + 0.0001 / obji.detach().item()

        if self.autobalance:
            self.balance = [x / self.balance[self.ssi] for x in self.balance]
        lbox *= self.hyp["box"]
        lobj *= self.hyp["obj"]
        lcls *= self.hyp["cls"]
        bs = tobj.shape[0]  # batch size

        return (lbox + lobj + lcls) * bs, torch.cat((lbox, lobj, lcls)).detach()"""

    def __call__(self, p, targets, epoch=0):
        if epoch >= 0:
            logger.debug("\n=== Starting CustomComputeLoss Call ===")
            logger.debug(f"Predictions shape: {[p_.shape for p_ in p]}")
            logger.debug(f"Targets shape: {targets.shape}")
        
        lcls = torch.zeros(1, device=self.device)
        lbox = torch.zeros(1, device=self.device)
        lobj = torch.zeros(1, device=self.device)
        
        tcls, tbox, indices, anchors = self.build_targets(p, targets)

        if epoch >= 0:
            logger.debug(f"\nBuild targets output:")
            logger.debug(f"tcls shapes: {[t.shape for t in tcls]}")
            logger.debug(f"tbox shapes: {[t.shape for t in tbox]}")
            logger.debug(f"indices lengths: {[len(i) for i in indices]}")
            logger.debug(f"anchors shapes: {[a.shape for a in anchors]}")

        for i, pi in enumerate(p):
            if epoch >= 0:
                logger.debug(f"\nProcessing layer {i}:")
            b, a, gj, gi = indices[i]
            if epoch >= 0:
                logger.debug(f"Indices - b:{b.shape}, a:{a.shape}, gj:{gj.shape}, gi:{gi.shape}")
            
            tobj = torch.zeros(pi.shape[:4], dtype=pi.dtype, device=self.device)
            
            if n := b.shape[0]:
                if epoch >= 0:
                    logger.debug(f"Processing {n} targets")
                pxy, pwh, _, pcls = pi[b, a, gj, gi].split((2, 2, 1, self.nc), 1)
                if epoch >= 0:
                    logger.debug(f"Split shapes - pxy:{pxy.shape}, pwh:{pwh.shape}, pcls:{pcls.shape}")

                pxy = pxy.sigmoid() * 2 - 0.5
                pwh = (pwh.sigmoid() * 2) ** 2 * anchors[i]
                pbox = torch.cat((pxy, pwh), 1)
                if epoch >= 0:
                    logger.debug(f"Processed box shape: {pbox.shape}")

                eiou_loss, iou = bbox_eiou(pbox, tbox[i])
                if epoch >= 0:
                    logger.debug(f"EIoU loss output - loss:{eiou_loss}, iou:{iou}")

                """# focal loss from https://arxiv.org/pdf/2101.08158
                focal_weight = iou.pow(1.5)
                if epoch >= 0:
                    logger.debug(f"Focal weight: {focal_weight}")
                lbox += (focal_weight * eiou_loss).mean()"""

                #focal loss from yolo
                pred_prob = iou.clamp(1e-6, 1 - 1e-6)  # Avoid log(0)
                logit_iou = torch.log(pred_prob / (1 - pred_prob))  # Convert IoU to pseudo-logit
                true = torch.ones_like(logit_iou)  # Target is 1 (we want IoU to be high)
                p_t = true * pred_prob + (1 - true) * (1 - pred_prob)
                alpha_factor = true * 0.25 + (1 - true) * (1 - 0.25)
                modulating_factor = (1.0 - p_t) ** self.gamma
                focal_weight = alpha_factor * modulating_factor

                lbox += (focal_weight * eiou_loss).mean()

                if epoch >= 0:
                    logger.debug(f"Current lbox: {lbox}")

                # Objectness
                iou = iou.detach().clamp(0).type(tobj.dtype)
                if self.sort_obj_iou:
                    j = iou.argsort()
                    b, a, gj, gi, iou = b[j], a[j], gj[j], gi[j], iou[j]
                if self.gr < 1:
                    iou = (1.0 - self.gr) + self.gr * iou
                tobj[b, a, gj, gi] = iou  # iou ratio

                # Classification
                if self.nc > 1:  # cls loss (only if multiple classes)
                    t = torch.full_like(pcls, self.cn, device=self.device)  # targets
                    t[range(n), tcls[i]] = self.cp
                    lcls += self.BCEcls(pcls, t)  # BCE

            obji = self.BCEobj(pi[..., 4], tobj)
            lobj += obji * self.balance[i]  # obj loss
            if self.autobalance:
                self.balance[i] = self.balance[i] * 0.9999 + 0.0001 / obji.detach().item()

        if self.autobalance:
            self.balance = [x / self.balance[self.ssi] for x in self.balance]
        lbox *= self.hyp["box"]
        lobj *= self.hyp["obj"]
        lcls *= self.hyp["cls"]
        bs = tobj.shape[0]  # batch size

        return (lbox + lobj + lcls) * bs, torch.cat((lbox, lobj, lcls)).detach()

    def build_targets(self, p, targets):
        """Prepares model targets from input targets (image,class,x,y,w,h) for loss computation, returning class, box,
        indices, and anchors.
        """
        logger.debug(f"Original targets shape: {targets.shape}")
        logger.debug(f"Original targets values: {targets}")
        na, nt = self.na, targets.shape[0]  # number of anchors, targets
        tcls, tbox, indices, anch = [], [], [], []
        gain = torch.ones(7, device=self.device)  # normalized to gridspace gain
        ai = torch.arange(na, device=self.device).float().view(na, 1).repeat(1, nt)  # same as .repeat_interleave(nt)
        targets = torch.cat((targets.repeat(na, 1, 1), ai[..., None]), 2)  # append anchor indices

        g = 0.5  # bias
        off = (
            torch.tensor(
                [
                    [0, 0],
                    [1, 0],
                    [0, 1],
                    [-1, 0],
                    [0, -1],  # j,k,l,m
                    # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
                ],
                device=self.device,
            ).float()
            * g
        )  # offsets

        for i in range(self.nl):
            anchors, shape = self.anchors[i], p[i].shape
            logger.debug(f"Layer {i} anchor shapes: {anchors.shape}")
            logger.debug(f"Layer {i} anchor values: {anchors}")
            gain[2:6] = torch.tensor(shape)[[3, 2, 3, 2]]  # xyxy gain

            # Match targets to anchors
            t = targets * gain  # shape(3,n,7)
            if nt:
                # Matches
                r = t[..., 4:6] / anchors[:, None]  # wh ratio
                j = torch.max(r, 1 / r).max(2)[0] < self.hyp["anchor_t"]  # compare
                # j = wh_iou(anchors, t[:, 4:6]) > model.hyp['iou_t']  # iou(3,n)=wh_iou(anchors(3,2), gwh(n,2))
                t = t[j]  # filter

                # Offsets
                gxy = t[:, 2:4]  # grid xy
                gxi = gain[[2, 3]] - gxy  # inverse
                j, k = ((gxy % 1 < g) & (gxy > 1)).T
                l, m = ((gxi % 1 < g) & (gxi > 1)).T
                j = torch.stack((torch.ones_like(j), j, k, l, m))
                t = t.repeat((5, 1, 1))[j]
                offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]
            else:
                t = targets[0]
                offsets = 0

            # Define
            bc, gxy, gwh, a = t.chunk(4, 1)  # (image, class), grid xy, grid wh, anchors
            a, (b, c) = a.long().view(-1), bc.long().T  # anchors, image, class
            gij = (gxy - offsets).long()
            gi, gj = gij.T  # grid indices

            # Append
            indices.append((b, a, gj.clamp_(0, shape[2] - 1), gi.clamp_(0, shape[3] - 1)))  # image, anchor, grid
            tbox.append(torch.cat((gxy - gij, gwh), 1))  # box
            anch.append(anchors[a])  # anchors
            tcls.append(c)  # class

        return tcls, tbox, indices, anch