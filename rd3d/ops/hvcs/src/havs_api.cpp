#include "havs_api.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("potential_voxel_size", &potential_voxel_size_wrapper, "potential_voxel_size_wrapper (cuda)");
    m.def("potential_voxel_size_stack", &potential_voxel_size_stack_wrapper,
          "potential_voxel_size_stack_wrapper (cuda)");
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    m.def("voxel_size_experiments", &voxel_size_experiments_wrapper, "voxel info statistic (cuda)");
    m.def("adaptive_sampling_and_query", &AdaptiveSamplingAndQuery, "AdaptiveSamplingAndQuery");
    m.def("query_from_hash_table", &QueryFromHash, "QueryFromHash");
}

