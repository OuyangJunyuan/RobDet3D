#include <torch/extension.h>
#include <torch/serialize/tensor.h>

/**
 * @brief  We adaptively search a optimal voxel that used to down-sample the sources.
 * Further, the point closest to its voxel center is sampled as the representative point of this voxel.
 * To fast count into non-empty voxels, we perform voxel-hashing for above operations.
 * @note It typically (on 2080Ti GPU)
 * @note takes 0.35 ms to sample  4096 from 16384 with a batch size of 8,
 * @note takes 0.26 ms to sample 16384 from 65536 with a batch size of 1.
 * @note takes 0.86 ms to sample 16384 from 65536 with a batch size of 8.
 * @param source the source points
 * @param sampled_ids the index in sources of each sampled points.
 * @param init_voxel_x,init_voxel_y,init_voxel_z initialize voxel size. We search optimal voxel between 0 and 2X init_voxel.
 * @param tolerance is converge if the number of non-empty voxels falls in [1,1+tolerance]*num_trg.
 * @param return_details
 * @param return_coord2subset some information for grid query.
 */
std::vector<at::Tensor> HAVSamplingBatchWrapper(at::Tensor &sources,
                                                at::Tensor &sampled_ids,
                                                double init_voxel_x,
                                                double init_voxel_y,
                                                double init_voxel_z,
                                                double tolerance,
                                                long max_iterations,
                                                bool return_details,
                                                bool return_coord2subset);

/**
 * @brief We traverse through source points and scatter them to their neighbour queries,
 * instead of gathering source points for each query.
 * Besides, we leverage the voxel-hash table constructed previously in HAVSampler to accelerate neighbour searching.
 * What' more, our GridQuery have complexity of O(num_src X K^3) rather than origin O(num_src X num_qry).
 * By doing that, we gain 100 ~ 1000 times faster.
 * @note It typically (on 2080Ti GPU)
 * @note takes 0.50 ms to query 32 neighbors in 16384 sources for 4096 query with a batch size of 8.
 * @note takes 0.37 ms to query 32 neighbors in 65536 sources for 16384 query with a batch size of 1.
 * @note takes 3.50 ms to query 32 neighbors in 65536 sources for 16384 query with a batch size of 8.
 * @param queries the center of each groups, which known as query.
 * @param sources the source points to be queried.
 * @param queried_ids the neighbors' index in source of each query.
 * @param num_queried the number of neighbors in sources of each query.
 * @param voxel_sizes the voxel size that use to sample query point from sources.
 * @param hash_tables voxel-hashing table, used to exam if a voxel is non-empty.
 * @param coord2query the table that map a hash-value to queries' index.
 * @param search_radius the radius of nearest points searching.
 * @param num_neighbours the number of nearest points the query_xyz need to query.
 */
void QueryByPointHashingBatchWrapper(at::Tensor &queries, at::Tensor &sources,
                                     at::Tensor &queried_ids, at::Tensor &num_queried,
                                     at::Tensor &voxel_sizes, at::Tensor &hash_tables, at::Tensor &coord2query,
                                     double search_radius, long num_neighbours);


/**
 * @brief TORCH_EXTENSION_NAME will be automatically replaced by the extension name writen in "setup.py", i.e., havs_cuda.
 * @example from havs import havs_cuda
 */
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("havs_batch", &HAVSamplingBatchWrapper,
          "hierarchical adaptive voxel-guided point sampling (batch)");
    m.def("query_batch", &QueryByPointHashingBatchWrapper,
          "query neighbor from sources by point-voxel hashing (batch)");
}
/**
 * @brief Register to TorchScript operation set.
 * @example cuda_spec = importlib.machinery.PathFinder().find_spec(library, [os.path.dirname(__file__)])
 * @example torch.ops.load_library(cuda_spec.origin)
 */
TORCH_LIBRARY(havs_cuda, m) {
    m.def("havs_batch", &HAVSamplingBatchWrapper);
    m.def("query_batch", &QueryByPointHashingBatchWrapper);
}