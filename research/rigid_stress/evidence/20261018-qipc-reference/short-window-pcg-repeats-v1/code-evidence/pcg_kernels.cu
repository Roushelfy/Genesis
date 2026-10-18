/// @file pcg_kernels.cu
/// @brief Non-template PCG kernel definitions using PCGContext.

#include <solver/subgraph/pcg_kernels.h>
#include <solver/pcg_device_functions.h>
#include <eigen3/inverse.h>
#include <common.h>
#include <grid_stride_loop.h>
#include <cub/block/block_reduce.cuh>
#include <cub/warp/warp_reduce.cuh>

namespace qipc
{

// ---- __device__ helper definitions (declared in pcg_device_functions.h) ----

__device__ void spmv(BufferView<int32_t> bcoo_row,
                     BufferView<int32_t> bcoo_col,
                     BufferView<double>  bcoo_val,
                     BufferView<double>  p,
                     BufferView<double>  Ap,
                     int64_t             total_nnz,
                     typename cub::WarpReduce<double>::TempStorage* temp_storage)
{
    using WarpReduce = cub::WarpReduce<double>;

    for(int64_t tid = blockIdx.x * blockDim.x + threadIdx.x; tid < total_nnz;
        tid += blockDim.x * (int64_t)gridDim.x)
    {
        int32_t ri   = bcoo_row[tid];
        int32_t ci   = bcoo_col[tid];
        auto    Bmat = bcoo_val.as<3, 3>(tid);

        Eigen::Vector3d yi = Bmat * p.as<3>(ci);

        // Only the (ri, ci) block is stored; its mirror is the transpose.
        if(ri != ci)
        {
            Eigen::Vector3d yj = Bmat.transpose() * p.as<3>(ri);
            Ap.atomic_add_as<3>(ci, yj);
        }

        // Consecutive lanes hold consecutive nnz and the BCOO is row-sorted, so a
        // lane heads a row segment iff its row differs from the previous lane's.
        int32_t lane    = threadIdx.x & 31;
        int32_t warp_id = threadIdx.x >> 5;
        int32_t prev_ri = __shfl_up_sync(0xffffffff, ri, 1);
        bool    is_head = (lane == 0) || (ri != prev_ri);

        double red0 =
            WarpReduce(temp_storage[warp_id]).HeadSegmentedReduce(yi(0), is_head, cub::Sum());
        double red1 =
            WarpReduce(temp_storage[warp_id]).HeadSegmentedReduce(yi(1), is_head, cub::Sum());
        double red2 =
            WarpReduce(temp_storage[warp_id]).HeadSegmentedReduce(yi(2), is_head, cub::Sum());

        if(is_head)
        {
            Eigen::Vector3d red(red0, red1, red2);
            Ap.atomic_add_as<3>(ri, red);
        }
    }
}

__device__ void update_xr(BufferView<double> x_sol,
                          BufferView<double> p,
                          BufferView<double> r,
                          BufferView<double> Ap,
                          double             alpha,
                          int64_t            n)
{
    grid_stride_loop(n,
                     [&](int64_t i)
                     {
                         x_sol[i] += alpha * p[i];
                         r[i] -= alpha * Ap[i];
                     });
}

__device__ void update_p(BufferView<double> p, BufferView<double> z, double beta, int64_t n)
{
    grid_stride_loop(n, [&](int64_t i) { p[i] = z[i] + beta * p[i]; });
}

__device__ bool check_pcg_convergence(VarView<int64_t> pcg_iter,
                                      VarView<double>  rz_new,
                                      VarView<double>  conv_threshold)
{
    int64_t rem = *pcg_iter - 1;
    *pcg_iter   = rem;
    return (rem <= 0 || fabs(*rz_new) <= *conv_threshold);
}

// ---- __global__ kernel definitions ----

__global__ void pcg_ctx_copy_z_to_p_kernel(PCGContext* ctx)
{
    auto p = ctx->p_view();
    auto z = ctx->z_view();
    grid_stride_loop(ctx->total_dof(), [&](int64_t i) { p[i] = z[i]; });
}

__global__ void pcg_ctx_memset_Ap_kernel(PCGContext* ctx)
{
    auto Ap = ctx->Ap_view();
    grid_stride_loop(ctx->total_dof(), [&](int64_t i) { Ap[i] = 0.0; });
}

__global__ void pcg_dot_pAp_partial_kernel(PCGContext* ctx)
{
    using BlockReduce = cub::BlockReduce<double, PCG_BLOCK_DIM>;
    __shared__ typename BlockReduce::TempStorage temp;
    dot_partial<PCG_BLOCK_DIM>(
        ctx->p_view(), ctx->Ap_view(), ctx->partial_view(), ctx->total_dof(), temp);
}

__global__ void pcg_dot_pAp_final_kernel(PCGContext* ctx)
{
    using BlockReduce = cub::BlockReduce<double, PCG_BLOCK_DIM>;
    __shared__ typename BlockReduce::TempStorage temp;
    dot_final<PCG_BLOCK_DIM>(ctx->partial_view(), ctx->pAp_view(), temp);
}

__global__ void pcg_dot_rz_partial_kernel(PCGContext* ctx)
{
    using BlockReduce = cub::BlockReduce<double, PCG_BLOCK_DIM>;
    __shared__ typename BlockReduce::TempStorage temp;
    dot_partial<PCG_BLOCK_DIM>(
        ctx->r_view(), ctx->z_view(), ctx->partial_view(), ctx->total_dof(), temp);
}

__global__ void pcg_dot_rz_final_kernel(PCGContext* ctx)
{
    using BlockReduce = cub::BlockReduce<double, PCG_BLOCK_DIM>;
    __shared__ typename BlockReduce::TempStorage temp;
    dot_final<PCG_BLOCK_DIM>(ctx->partial_view(), ctx->rz_new_view(), temp);
}

__global__ void pcg_compute_alpha_kernel(PCGContext* ctx)
{
    double pAp_val     = *ctx->pAp_view();
    *ctx->alpha_view() = (pAp_val != 0.0) ? (*ctx->rz_view() / pAp_val) : 0.0;
}

__global__ void pcg_compute_beta_kernel(PCGContext* ctx)
{
    double rz_val     = *ctx->rz_view();
    *ctx->beta_view() = (rz_val != 0.0) ? (*ctx->rz_new_view() / rz_val) : 0.0;
}

__global__ void pcg_update_p_kernel(PCGContext* ctx)
{
    double b = *ctx->beta_view();
    update_p(ctx->p_view(), ctx->z_view(), b, ctx->total_dof());
}

__global__ void pcg_swap_rz_kernel(PCGContext* ctx)
{
    *ctx->rz_view() = *ctx->rz_new_view();
}

__global__ void pcg_check_convergence_kernel(PCGContext* ctx, int32_t* pcg_cond)
{
    bool done = check_pcg_convergence(ctx->pcg_iter_remaining_view(),
                                      ctx->rz_new_view(),
                                      ctx->conv_threshold_view());

    if(done)
        *pcg_cond = 0;
}

__global__ void pcg_compute_conv_threshold_kernel(PCGContext* ctx)
{
    double rz0 = fabs(*ctx->rz_view());
    QIPC_KERNEL_ASSERT(!isnan(rz0) && !isinf(rz0),
                       "PCG init rz0 is NaN/Inf (%.6e), gradient or preconditioner corrupted",
                       rz0);
    *ctx->conv_threshold_view() = ctx->tol_rate() * rz0;
}

__global__ void pcg_set_cond_kernel(int32_t* pcg_cond)
{
    *pcg_cond = 1;
}

__global__ void pcg_set_cond_gated_kernel(int32_t* pcg_cond, const int32_t* skip)
{
    *pcg_cond = (*skip != 0) ? 0 : 1;
}

}  // namespace qipc

// ============================================================================
// Toy preconditioner kernels (ToyPCGContext)
// ============================================================================

#include <solver/subgraph/diag_3x3_preconditioner.h>

namespace qipc
{

__global__ void toy_precond_memset_kernel(ToyPCGContext* ctx)
{
    auto inv_diag = ctx->precond_inv_diag_view();

    grid_stride_loop(ctx->n_diag_blocks,
                     [&](int64_t blk) { inv_diag.as<3, 3>(blk).setIdentity(); });
}

__global__ void toy_precond_invert_kernel(ToyPCGContext* ctx)
{
    auto inv_diag = ctx->precond_inv_diag_view();

    grid_stride_loop(ctx->n_diag_blocks,
                     [&](int64_t blk)
                     {
                         Eigen::Matrix<double, 3, 3> A = inv_diag.as<3, 3>(blk);
                         inv_diag.as<3, 3>(blk)        = eigen::inverse(A);
                     });
}

__global__ void toy_precond_apply_kernel(ToyPCGContext* ctx, PCGPrecondContext* pcg)
{
    auto    inv_diag   = ctx->precond_inv_diag_view();
    int64_t n_verts    = ctx->n_diag_blocks;
    int64_t dof_offset = ctx->dof_offset;
    auto    r          = pcg->r().subview(dof_offset, n_verts * 3);
    auto    z          = pcg->z().subview(dof_offset, n_verts * 3);

    grid_stride_loop(n_verts,
                     [&](int64_t v)
                     {
                         auto Minv = inv_diag.as<3, 3>(v);
                         auto ri   = r.as<3>(v);
                         auto zi   = z.as<3>(v);
                         zi        = Minv * ri;
                     });
}

}  // namespace qipc
