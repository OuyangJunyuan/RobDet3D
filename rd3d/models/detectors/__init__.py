from .detector3d_template import Detector3DTemplate
from .PartA2_net import PartA2Net
from .point_rcnn import PointRCNN
from .pointpillar import PointPillar
from .pv_rcnn import PVRCNN
from .second_net import SECONDNet
from .second_net_iou import SECONDNetIoU
from .caddn import CaDDN
from .voxel_rcnn import VoxelRCNN
from .centerpoint import CenterPoint
from .pv_rcnn_plusplus import PVRCNNPlusPlus
from .point_3dssd import Point3DSSD
from .IASSD import IASSD
from .point_vote import PointVote

__all__ = {
    'Detector3DTemplate': Detector3DTemplate,
    'SECONDNet': SECONDNet,
    'PartA2Net': PartA2Net,
    'PVRCNN': PVRCNN,
    'PointPillar': PointPillar,
    'PointRCNN': PointRCNN,
    'SECONDNetIoU': SECONDNetIoU,
    'CaDDN': CaDDN,
    'VoxelRCNN': VoxelRCNN,
    'CenterPoint': CenterPoint,
    'PVRCNNPlusPlus': PVRCNNPlusPlus,
    '3DSSD': Point3DSSD,
    'IASSD': IASSD,
    'PointVote': PointVote,
}


def build_detector(model_cfg, dataset, num_class=None):
    model = __all__[model_cfg.NAME](
        model_cfg=model_cfg, num_class=len(model_cfg.CLASS_NAMES) if num_class is None else num_class, dataset=dataset
    )

    return model
