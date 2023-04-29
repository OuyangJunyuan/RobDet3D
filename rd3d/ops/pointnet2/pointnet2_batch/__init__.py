from .pointnet2_utils import FarthestPointSampling, furthest_point_sample_matrix, furthest_point_sample_weights


class FPS(FarthestPointSampling):
    @staticmethod
    def symbolic(g, xyz, sample_num):
        return g.op(
            "rd3d::FPSampling", xyz,
            num_sample_i=sample_num
        )
