#include <solver/solver_kernels.h>
#include <solver/pcg_device_functions.h>
#include <common.h>
#include <cub/block/block_reduce.cuh>
#include <grid_stride_loop.h>
#include <solver/sim_context.h>
#include <solver/global_linear_system_context.h>
#include <solver/contact_context.h>
#include <buffer_view.h>
#include <cstdint>

namespace qipc
{

// =====================================================================
// memset (trivial)
// =====================================================================

__global__ void memset_b_rhs_kernel(const GlobalLinearSystemContext* lctx)
{
    auto b_rhs = lctx->b_rhs_view();  // RHS = +gradient, zeroed before assemble
    grid_stride_loop(lctx->total_dof, [&](int64_t i) { b_rhs[i] = 0.0; });
}

__global__ void memset_triplet_kernel(const GlobalLinearSystemContext* lctx)
{
    auto    triplet_val = lctx->triplet_val_view();
    int64_t total_vals  = lctx->max_triplets * 9;
    grid_stride_loop(total_vals, [&](int64_t i) { triplet_val[i] = 0.0; });
}

__global__ void reset_newton_convergence_partials_kernel(const SimContext* ctx)
{
    auto free_partials    = ctx->conv_partial_view();
    auto contact_partials = ctx->conv_partial_contact_view();
    grid_stride_loop(free_partials.size(),
                     [&](int64_t i)
                     {
                         free_partials[i]    = 0.0;
                         contact_partials[i] = 0.0;
                     });
}

// =====================================================================
// convergence check (CUB block reduction)
// =====================================================================

template <int BlockDim>
__device__ double reduce_partial_max(BufferView<double> partial,
                                     typename cub::BlockReduce<double, BlockDim>::TempStorage& temp)
{
    using BlockReduce = cub::BlockReduce<double, BlockDim>;

    double val = 0.0;
    for(int32_t i = threadIdx.x; i < partial.size(); i += blockDim.x)
    {
        double v = partial[i];
        if(v > val)
            val = v;
    }
    return BlockReduce(temp).Reduce(val, cub::Max());
}

// Split convergence check: the free bucket is judged against abs_tol, the
// contact bucket (vertices in an in-band barrier pair this iteration)
// against abs_tol * conv_tol_scale. scale == 1 (feature off) makes both
// criteria identical, so the split is exactly the legacy single check.
// max_disp keeps the TRUE combined max: the displacement clamp and the
// diagnostics must see the unrelaxed magnitude.
__global__ void pre_ccd_convergence_check_kernel(const SimContext* ctx,
                                                 const GlobalContactContext* cctx)
{
    using BlockReduce = cub::BlockReduce<double, PCG_BLOCK_DIM>;
    __shared__ typename BlockReduce::TempStorage temp;

    double max_free = reduce_partial_max<PCG_BLOCK_DIM>(ctx->conv_partial_view(), temp);
    __syncthreads();  // temp storage reuse between the two reductions
    double max_contact =
        reduce_partial_max<PCG_BLOCK_DIM>(ctx->conv_partial_contact_view(), temp);

    if(threadIdx.x == 0)
    {
        double abs_tol = *ctx->abs_tol_view();
        double scale   = cctx ? cctx->conv_tol_scale : 1.0;

        *ctx->max_disp_view()         = fmax(max_free, max_contact);
        *ctx->max_disp_free_view()    = max_free;
        *ctx->max_disp_contact_view() = max_contact;

        bool conv = (max_free < abs_tol) && (max_contact < abs_tol * scale);
        *ctx->converged_view() = conv ? 1 : 0;
    }
}

__global__ void compute_clamp_alpha_kernel(const SimContext*           ctx,
                                           const GlobalContactContext* cctx)
{
    double s        = 1.0;
    double max_step = *ctx->max_step_in_d_hat_view();
    if(max_step > 0.0)
    {
        double max_disp = *ctx->max_disp_view();
        double d_hat    = cctx->d_hat;
        double bound    = max_step * d_hat;
        if(d_hat > 0.0 && max_disp > bound)
            s = bound / max_disp;
    }
    *ctx->clamp_alpha_view() = s;
}

}  // namespace qipc
