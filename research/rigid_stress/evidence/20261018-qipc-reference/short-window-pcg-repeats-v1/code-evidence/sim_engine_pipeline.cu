// sim_engine_pipeline.cu
// Pipeline builder implementations for qipc::SimEngine.
// Absorbs the former newton_graph_builder.cu and init_graph_builder.cu.

#include <solver/sim_engine.h>
#include <solver/sim_config.h>
#include <solver/mpm_system.h>
#include <solver/mpm_kernels.h>
#include <solver/masked_pcg_solver.h>
#include <solver/ccd_component_partitioner.h>
#include <solver/ccd_component_partitioner_context.h>
#include <solver/subgraph/pcg_solver_system.h>
#include <solver/fem_mas_preconditioner.h>
#include <solver/contact_system.h>
#include <solver/finite_element_method.h>
#include <solver/global_vertex_manager.h>
#include <solver/global_body_manager.h>
#include <solver/cut_detection_system.h>
#include <solver/dump_diag.h>
#include <solver/abd_diag_preconditioner.h>
#include <solver/abd_tree_preconditioner.h>
#include <solver/fem_diag_preconditioner.h>
#include <solver/bspline_diag_preconditioner.h>
#include <solver/bspline_system.h>
#include <solver/bspline_context.h>
#include <solver/bspline_constitution.h>
#include <solver/bspline_contact_assemble.h>
#include <solver/bspline_contact_route.h>
#include <solver/subgraph/dynamic_radix_sort.h>
#include <solver/subgraph/dynamic_exclusive_sum.h>
#include <solver/broad_phase_system.h>
#include <solver/abd_kernels.h>
#include <solver/solver_kernels.h>
#include <solver/fem_kernels.h>
#include <solver/fem_constitution.h>
#include <solver/abd_constitution.h>
#include <solver/inter_abd_constitution.h>
#include <solver/abd_bdf1.h>
#include <solver/affine_body_dynamics.h>
#include <solver/contact_constitution.h>
#include <solver/abd_context.h>
#include <solver/linear_system.h>
#include <solver/kernel.h>
#include <common.h>
#include <cuda_check.h>
#include <grid_stride_loop.h>
#include <cstdlib>
#include <solver/core/pipeline_builder.h>
#include <solver/core/pipeline_kernels.h>
#include <solver/contact_kernels.h>
#include <solver/contact_device_functions.h>
#include <solver/contact_assembly_routes.h>
#include <solver/abd_contact_assemble.h>
#include <solver/rigid_contact_assemble.h>
#include <solver/rigid_kernels.h>
#include <solver/rigid_body_dynamics.h>
#include <solver/rigid_diag_preconditioner.h>
#include <solver/rigid_tree_preconditioner.h>
#include <solver/rigid_context.h>
#include <solver/bond_system.h>
#include <solver/picking_point.h>
#include <solver/component_partitioner.h>
#include <solver/abd_fem_coupling.h>
#include <solver/twist_contact_distribute.h>
#include <solver/kirchhoff_rod_twisting.h>
#include <solver/fsr_reduce.h>
#include <solver/ccd_kernels.h>
#include <solver/subgraph/bvh_build.h>
#include <solver/solver_sort_accessors.h>
#include <solver/global_surface_manager.h>
#include <solver/global_linear_system.h>
#include <solver/global_joint_system.h>
#include <solver/global_joint_context.h>
#include <solver/algebraic_coarsening.h>
#include <solver/fem_cluster_manager.h>
#include <solver/affine_body_external_force.h>
#include <solver/rigid_body_external_force.h>
#include <solver/global_body_context.h>
#include <solver/pcg_device_functions.h>
#include <solver/edge_triangle_checker.h>
#include <buffer_view.h>
#include <logger.h>
#include <solver/contact_context.h>
#include <solver/core/pipeline.h>
#include <tuple>
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <utility>

using qipc::GlobalContactContext;
using qipc::GlobalLinearSystemContext;
using qipc::GlobalVertexContext;

namespace qipc
{

Pipeline SimEngine::build_step_pipeline()
{
    PipelineGraphBuilder builder;

    VarView<int32_t> overflow_var =
        contact_ ? contact_->overflow_flag_var() : VarView<int32_t>{};
    // Always valid: the in-graph extent check yields on elastic/dof growth
    // after topo events even without contact.
    VarView<int32_t> triplet_overflow_var = triplet_overflow_flag_.view();
    VarView<int32_t> count_overflow_var =
        contact_ ? count_overflow_flag_.view() : VarView<int32_t>{};
    VarView<int32_t> export_flag_var =
        export_flag_.empty() ? VarView<int32_t>{} : export_flag_.view();
    VarView<int32_t> closure_flag_var =
        closure_flag_.empty() ? VarView<int32_t>{} : closure_flag_.view();

    Pipeline pipe = builder.main(
        [&](MainScope& m)
        {
            // ==========================================================
            // MPM grid stage (pre-Newton): deterministic grid rebuild +
            // P2G. Grid topology is frame-fixed from here on; block-pool
            // overflow yields and re-runs this checkpoint after growth
            // (docs/mpm_coupling_design.md decision #9).
            // ==========================================================
            if(mpm_)
            {
                m.checkpoint(
                    [&](CheckpointScope& cp)
                    {
                        if(auto sg =
                               mpm_->build_grid_p2g(sim_config_->ctx().view().data()))
                        {
                            auto n_build = cp.subgraph(sg, {}).tail();
                            // Frame-head contact-vertex refresh: advection
                            // moved x_p and the reset zeroed u, so the
                            // contact positions restart at x_p (decision #10).
                            if(mpm_->n_particles() > 0 && contact_)
                                cp.subgraph(mpm_->forward_global_vertices(
                                                vertex_->ctx().view().data()),
                                            {n_build});
                        }
                    },
                    mpm_->grid_overflow_var(),
                    [this](YieldContext& yc) { handle_mpm_grid_overflow(yc); },
                    YieldResume::This);
            }

            // MPM grid rows are global-system DOFs (decision #9), so an
            // MPM-only scene still runs the full Newton machinery.
            bool has_lagrangian_dofs = abd_ || rigid_ || fem_ || bspline_ || mpm_;
            if(has_lagrangian_dofs)
            {
                // ==========================================================
                // Checkpoint 0: Initialize frame
                // ==========================================================
                m.checkpoint(
                    [&](CheckpointScope& cp)
                    {
                        auto n_init = cp.kernel(
                            KernelConfig{"init_frame"}.grid(1).block(1),
                            [ctx = sim_config_->ctx().view(),
                             cctx = contact_ ? contact_->ctx().view() :
                                               VarView<const GlobalContactContext>{}] __device__() mutable
                            {
                                *ctx->abs_tol_view() = *ctx->tol_view() * *ctx->dt_view();
                                *ctx->max_pcg_used_view()    = 0;
                                *ctx->max_ls_used_view()     = 0;
                                *ctx->total_pcg_used_view()  = 0;
                                *ctx->total_ls_used_view()   = 0;
                                *ctx->min_ccd_alpha_view()   = 1.0;
                                *ctx->total_ccd_alpha_view() = 0.0;
                                *ctx->newton_iter_view()     = 0;
                                *ctx->converged_view()       = 0;
                                // Per-frame like the CCD alpha trackers above;
                                // null view when the scene has no contact.
                                if(cctx.data())
                                {
                                    *cctx->buffer_frame_counts_view() =
                                        ContactBufferFrameCounts{};
                                    *cctx->max_accd_iters_view() = 0;
                                    // The global adaptive-kappa controller
                                    // triggers off the finished frame's value,
                                    // so snapshot it before re-arming (its
                                    // update node is a separate CP0 node and
                                    // must not race this one for the tracker).
                                    *cctx->prev_min_gap_ratio_view() =
                                        *cctx->min_gap_ratio_view();
                                    *cctx->min_gap_ratio_view() = MIN_GAP_RATIO_UNRECORDED;
                                }
                            },
                            {});

                        // Adaptive kappa: grow/relax the stiffness multipliers from
                        // last frame's gaps and re-arm the per-vertex tracker.
                        // Must run before anything reads kappa this frame; CP0
                        // precedes every filter/energy/friction kernel, and the
                        // frame's E0 is computed fresh on Newton iteration 1, so
                        // the scale change never mixes two stiffnesses in one
                        // solve. No-ops (bar the re-arm) while the feature is off.
                        // Ordered after init_frame, which hands the global mode its
                        // trigger signal by snapshotting the min-gap tracker.
                        if(contact_)
                            cp.subgraph(contact_->adaptive_kappa_update(), {n_init});

                        // DumpDiag log reset: in-graph so a yield/resume within
                        // the frame never re-zeroes a partially filled log.
                        if(dump_diag_)
                            cp.subgraph(dump_diag_->reset(), {});

                        std::vector<cudaGraphNode_t> predict_deps;

                        // needs_sticky_normals() must be in this OR: a scene that only
                        // enables distance-lock bonds leaves the adhesion tables
                        // unwired, so needs_lagged_positions() is false, yet the
                        // single-sided lock gate still needs the CP0 snapshot and the
                        // step-open normals derived from it.
                        if(contact_
                           && (contact_->has_friction()
                               || contact_->constitution().needs_lagged_positions()
                               || contact_->needs_sticky_normals()))
                        {
                            auto n_snap =
                                cp.subgraph(contact_->friction_snapshot(
                                                vertex_->ctx().view().data(),
                                                surface_->ctx().view().data()),
                                            {})
                                    .tail();
                            predict_deps.push_back(n_snap);
                        }

                        {
                            auto abd_deps = predict_deps;
                            if(abd_)
                            {
                                auto n_seed =
                                    cp.subgraph(abd_->seed_applied_force(
                                                    sim_config_->ctx().view().data()),
                                                abd_deps)
                                        .tail();
                                abd_deps = {n_seed};
                            }

                            if(ext_force_)
                            {
                                auto collect_tpl = ext_force_->collect_force();
                                if(collect_tpl)
                                {
                                    auto n_collect =
                                        cp.subgraph(collect_tpl, abd_deps).tail();
                                    abd_deps = {n_collect};
                                }
                            }

                            if(abd_)
                            {
                                cp.subgraph(
                                    abd_->predict(sim_config_->ctx().view().data()), abd_deps);
                            }
                        }
                        if(rigid_)
                        {
                            auto rigid_deps = predict_deps;
                            if(rigid_ext_force_)
                            {
                                auto collect = rigid_ext_force_->collect_force();
                                if(collect)
                                    rigid_deps = {
                                        cp.subgraph(collect, rigid_deps).tail()};
                            }
                            cp.subgraph(
                                rigid_->predict(sim_config_->ctx().view().data()), rigid_deps);
                        }
                        if(fem_)
                        {
                            cp.subgraph(fem_->predict(sim_config_->ctx().view().data()),
                                        predict_deps);
                        }
                        if(bspline_)
                        {
                            auto n_predict =
                                cp.subgraph(bspline_->predict(
                                                sim_config_->ctx().view().data()),
                                            predict_deps)
                                    .tail();
                            cp.subgraph(bspline_->forward_global_vertices(
                                            vertex_->ctx().view().data()),
                                        {n_predict});
                        }
                    });

                // PickingPoint owns a separate checkpoint so candidate-buffer
                // growth replays only its snapshot/query/freeze work. Replaying
                // CP0 would incorrectly tick adaptive kappa and prediction twice.
                if(picking_point_)
                {
                    m.checkpoint(
                        [&](CheckpointScope& cp)
                        { cp.subgraph(picking_point_->prepare_step(), {}); },
                        picking_point_->overflow_flag_var(),
                        [this](YieldContext& ctx)
                        {
                            picking_point_->handle_overflow(stream_);
                            sync_dirty_contexts(stream_);
                            ctx.node_params_updated();
                        },
                        YieldResume::This);
                    m.checkpoint(
                        [&](CheckpointScope& cp)
                        { cp.subgraph(picking_point_->commit_step(), {}); });
                }

                auto newton = m.loop(
                    VarView<const int32_t>(newton_cond_.view().data()),
                    [&](LoopScope& newton)
                    {
                        // ==========================================================
                        // CP_count: count-only narrow pass for assembly buffer sizing.
                        // Runs the distance filter without scatter to predict the
                        // actual doublet/triplet counts. Yields if the current
                        // assembly buffers are too small, resizes, then skips to CP1a.
                        // ==========================================================
                        if(contact_)
                        {
                            newton.checkpoint(
                                [&](CheckpointScope& cp)
                                {
                                    cp.subgraph(contact_->count_active(
                                                    vertex_->ctx().view().data(),
                                                    surface_->ctx().view().data(),
                                                    count_overflow_var),
                                                {});
                                },
                                count_overflow_var,
                                [this](YieldContext& ctx)
                                { handle_count_overflow(ctx); },
                                YieldResume::Next);
                        }

                        // ==========================================================
                        // Checkpoint 1a-filter: init_newton + filter_assemble +
                        // contact padding overflow check.
                        // Yields if actual contact count exceeds padded range
                        // (Resume Next -> 1a-sort uses expanded padding).
                        // ==========================================================
                        {
                            VarView<int32_t> pad_overflow_var =
                                contact_ ? contact_padding_overflow_flag_.view() :
                                           VarView<int32_t>{};

                            newton.checkpoint(
                                [&](CheckpointScope& cp)
                                {
                                    auto n_init_newton = cp.kernel(
                                        KernelConfig{"init_newton"}.grid(1).block(1),
                                        [ctx = sim_config_->ctx().view(),
                                         newton_cond = newton_cond_.view(),
                                         first_flag = first_newton_flag_.view(),
                                         lctx = linear_system_->ctx().view(),
                                         pcg = linear_system_->pcg_solver().context()] __device__() mutable
                                        {
                                            *newton_cond = 1;
                                            (*ctx->newton_iter_view())++;
                                            *first_flag =
                                                (*ctx->newton_iter_view() == 1) ? 1 : 0;
                                            *lctx->bcoo_nnz_device_view() = 0;
                                            *pcg->pcg_iter_remaining_view() =
                                                *ctx->max_pcg_iter_view();
                                        },
                                        {});

                                    if(contact_)
                                    {
                                        // Newton-tick adaptive-kappa GROW node
                                        // (Scheme A): at the start of iteration
                                        // k+1, BEFORE its assemble, consume the
                                        // min gap iteration k's assemble filters
                                        // folded into the per-iteration trackers,
                                        // grow the multiplier(s) if it collapsed
                                        // below the trigger, and re-arm the
                                        // trackers. Everything downstream in this
                                        // iteration -- assemble (dep below),
                                        // energy, line search -- then sees ONE
                                        // consistent kappa; the objective is
                                        // piecewise-constant across iterations.
                                        // Because each decision re-checks the
                                        // freshest gap, growth stops by itself
                                        // once kappa re-opens the gap (natural
                                        // hysteresis; adaptive_max_scale is the
                                        // budget for the compounding per-
                                        // iteration grow). A grow flips
                                        // first_newton_flag so CP1b's E0
                                        // condition reprices the start-point
                                        // energy under the new kappa (E0 of
                                        // iterations >= 2 is otherwise carried
                                        // over from the OLD kappa's accepted
                                        // trial). Relax/decay stays on the CP0
                                        // frame nodes. After init_newton (both
                                        // write first_newton_flag); no-op unless
                                        // tick == Newton.
                                        auto n_kappa_tick = cp.kernel(
                                            KernelConfig{"adaptive_kappa_newton_tick"}
                                                .grid(1)
                                                .block(256),
                                            [cctx = contact_->ctx().view(),
                                             first_flag = first_newton_flag_.view()] __device__() mutable
                                            {
                                                if(cctx->adaptive_tick != AdaptiveKappaTick::Newton)
                                                    return;
                                                const GlobalContactContext* c = &*cctx;
                                                if(c->adaptive_mode == AdaptiveKappaMode::Global
                                                   && threadIdx.x == 0)
                                                {
                                                    auto scale = c->global_kappa_scale_view();
                                                    auto gap = c->iter_min_gap_ratio_view();
                                                    double s = *scale;
                                                    if(adaptive_kappa_newton_grow(c, s, *gap))
                                                    {
                                                        *scale      = s;
                                                        *first_flag = 1;
                                                    }
                                                    *gap = MIN_GAP_RATIO_UNRECORDED;
                                                }
                                                if(c->adaptive_mode == AdaptiveKappaMode::PerBody)
                                                {
                                                    auto scales =
                                                        c->body_kappa_scale_view();
                                                    auto gaps = c->iter_body_min_gap_view();
                                                    for(int32_t b =
                                                            (int32_t)threadIdx.x;
                                                        b < c->n_adaptive_bodies;
                                                        b += (int32_t)blockDim.x)
                                                    {
                                                        double s = scales[b];
                                                        if(adaptive_kappa_newton_grow(
                                                               c, s, gaps[b]))
                                                        {
                                                            scales[b] = s;
                                                            // Concurrent stores of
                                                            // the same 1: benign.
                                                            *first_flag = 1;
                                                        }
                                                        gaps[b] = MIN_GAP_RATIO_UNRECORDED;
                                                    }
                                                }
                                            },
                                            {n_init_newton});

                                        auto n_filter =
                                            cp.subgraph(
                                                  contact_->filter_assemble(
                                                      vertex_->ctx().view().data(),
                                                      surface_->ctx().view().data(),
                                                      sim_config_->ctx().view().data(),
                                                      export_flag_var),
                                                  {n_kappa_tick})
                                                .tail();

                                        cp.kernel(
                                            KernelConfig{"contact_padding_overflow_check"}
                                                .grid(1)
                                                .block(1),
                                            [cctx = contact_->ctx().view(),
                                             flag = pad_overflow_var] __device__() mutable
                                            {
                                                int32_t n_d =
                                                    *cctx->n_contact_doublets_view();
                                                int32_t n_t =
                                                    *cctx->n_contact_triplets_view();
                                                *flag = (n_d > (int32_t)cctx->padded_contact_doublets
                                                         || n_t > (int32_t)cctx->padded_contact_triplets) ?
                                                            1 :
                                                            0;
                                            },
                                            {n_filter});
                                    }
                                },
                                pad_overflow_var,
                                [this](YieldContext& ctx)
                                { handle_contact_padding_overflow(ctx); },
                                YieldResume::Next);
                        }

                        // ==========================================================
                        // Checkpoint 1a-sort: extent report/derive + contact sort +
                        // body triplet overflow check. Yields on triplet or dof
                        // capacity overflow. A B-spline route-key overflow must
                        // rerun this checkpoint after exact growth, so all
                        // overflows resume This. When no contact,
                        // triplet_overflow_var still yields on elastic/dof growth.
                        // ==========================================================
                        newton.checkpoint(
                            [&](CheckpointScope& cp)
                            {
                                // In-graph report-driven extents (B4): EVERY
                                // row-owning emitter publishes its live demand
                                // from its own context into its slot (parallel
                                // one-thread nodes), then GLS derives the
                                // offsets and the live elastic total
                                // device-side.
                                std::vector<cudaGraphNode_t> derive_deps;
                                auto add_report = [&](Subgraph sg)
                                {
                                    if(sg)
                                        derive_deps.push_back(
                                            cp.subgraph(std::move(sg), {}).tail());
                                };
                                const GlobalLinearSystemContext* d_lctx =
                                    linear_system_->ctx().view().data();
                                if(abd_)
                                {
                                    add_report(abd_->kinetic()->report_extent_node(d_lctx));
                                    for(auto* c : abd_->constitutions())
                                        add_report(c->report_extent_node(d_lctx));
                                }
                                if(rigid_)
                                    add_report(rigid_->report_extent_node(d_lctx));
                                if(joint_system_ && !joint_system_->empty())
                                {
                                    for(auto* c : joint_system_->constitutions())
                                        add_report(c->report_extent_node(d_lctx));
                                }
                                cudaGraphNode_t n_fem_report{};
                                if(fem_)
                                {
                                    n_fem_report =
                                        cp.subgraph(fem_->report_extent_node(d_lctx), {})
                                            .tail();
                                    derive_deps.push_back(n_fem_report);
                                    for(auto* c : fem_->constitutions())
                                        add_report(c->report_extent_node(d_lctx));
                                }
                                cudaGraphNode_t n_bspline_report{};
                                if(bspline_)
                                {
                                    std::vector<cudaGraphNode_t> bspline_deps;
                                    if(n_fem_report)
                                        bspline_deps.push_back(n_fem_report);
                                    n_bspline_report =
                                        cp.subgraph(bspline_->report_extent_node(d_lctx), bspline_deps)
                                            .tail();
                                    derive_deps.push_back(n_bspline_report);
                                }
                                if(mpm_)
                                {
                                    // Publishes the live MPM triplet demand AND
                                    // overwrites required_block_rows with the
                                    // full-system row demand (block_base +
                                    // active nodes); must order after the last
                                    // Lagrangian row report (B-spline when
                                    // present, otherwise FEM).
                                    std::vector<cudaGraphNode_t> mdeps;
                                    if(n_bspline_report)
                                        mdeps.push_back(n_bspline_report);
                                    else if(n_fem_report)
                                        mdeps.push_back(n_fem_report);
                                    derive_deps.push_back(
                                        cp.subgraph(mpm_->report_extent_node(d_lctx), mdeps)
                                            .tail());
                                }
                                auto n_derive =
                                    cp.subgraph(linear_system_->derive_extents(), derive_deps)
                                        .tail();

                                std::vector<cudaGraphNode_t> count_deps = {n_derive};
                                if(contact_)
                                {
                                    auto n_sort_reduce =
                                        cp.subgraph(contact_->sort_reduce(
                                                        vertex_->ctx().view().data(),
                                                        surface_->ctx().view().data()),
                                                    {})
                                            .tail();
                                    count_deps.push_back(n_sort_reduce);
                                    if(bspline_contact_route_)
                                        count_deps.push_back(
                                            cp.subgraph(bspline_contact_route_->prepare(triplet_overflow_var),
                                                        {n_sort_reduce})
                                                .tail());
                                }

                                const GlobalContactContext* cctx_ptr =
                                    contact_ ? contact_->ctx().view().data() : nullptr;
                                // Affine-cluster relocation appends expansion
                                // triplets after assemble; its running demand
                                // maximum joins the capacity check so growth
                                // rides the same yield path.
                                VarView<int32_t> cluster_reserve =
                                    cluster_manager_ ?
                                        cluster_manager_->reserve_view() :
                                        VarView<int32_t>{};
                                cp.kernel(
                                    KernelConfig{"compute_n_triplets"}.grid(1).block(1),
                                    [cctx_ptr,
                                     cluster_reserve,
                                     bspline_unique = bspline_contact_route_ ?
                                                          bspline_contact_route_->n_unique_view() :
                                                          VarView<int32_t>{},
                                     lctx = linear_system_->ctx().view(),
                                     overflow = triplet_overflow_var] __device__() mutable
                                    {
                                        int64_t n_el = *lctx->n_elastic_device_view();
                                        int32_t n =
                                            (int32_t)n_el
                                            + (cctx_ptr ? cctx_ptr->total_contact_triplets() : 0);
                                        if(bspline_unique.data())
                                            n += *bspline_unique;
                                        *lctx->n_triplets_device_view() = n;
                                        int32_t n_req =
                                            n + (cluster_reserve.data() ? *cluster_reserve : 0);
                                        int32_t of = *overflow;
                                        if(n_req > (int32_t)lctx->padded_triplets)
                                            of = 1;
                                        if(*lctx->required_block_rows_view() * 3 > lctx->total_dof)
                                            of = 1;
                                        *overflow = of;
                                    },
                                    count_deps);
                            },
                            triplet_overflow_var,
                            [this](YieldContext& ctx)
                            { handle_triplet_overflow(ctx); },
                            YieldResume::This);

                        // ==========================================================
                        // Checkpoint 1b: Assemble + PCG + Post-PCG (+ BVH build).
                        // Landing CP for triplet overflow resume from CP1a.
                        // ==========================================================
                        newton.checkpoint(
                            [&](CheckpointScope& cp)
                            {
                                // === ASSEMBLE SECTION ===

                                GridStrideKernel<memset_b_rhs_kernel> k_memset_grad;
                                auto n_memset_grad = cp.kernel(
                                    k_memset_grad,
                                    std::make_tuple(linear_system_->ctx().view().data()),
                                    {});

                                GridStrideKernel<memset_triplet_kernel> k_memset_trip;
                                auto n_memset_trip = cp.kernel(
                                    k_memset_trip,
                                    std::make_tuple(linear_system_->ctx().view().data()),
                                    {});

                                // --------------------------------------------------------
                                // Solve-scope derivation (TransformDBC / PositionDBC):
                                // writes this iteration's is_fixed, pinning bodies/verts
                                // that arrived at their target since the last iteration.
                                //
                                // The ABD, rigid, and FEM derivations run in parallel under ONE
                                // subgraph, whose empty tail node is the single common
                                // parent every is_fixed reader below hangs off of. The
                                // assemblies AND the contact distribution all mask fixed
                                // rows, and a fixed row that still receives a gradient gets
                                // a nonzero dq/dx while step_forward refuses to advance it
                                // -- permanently desyncing the global vertex positions
                                // (which the broad phase builds its AABBs from) from the
                                // true state, which silently drops collision pairs
                                // (issue #268). One parent for all of them keeps that
                                // contract checkable by inspection.
                                // --------------------------------------------------------
                                std::vector<Subgraph> scope_tpls;
                                if(abd_)
                                {
                                    auto t = abd_->derive_solve_scope(
                                        sim_config_->ctx().view().data(),
                                        linear_system_->ctx().view().data());
                                    if(t)
                                        scope_tpls.push_back(std::move(t));
                                }
                                if(rigid_)
                                {
                                    auto t = rigid_->derive_solve_scope(
                                        sim_config_->ctx().view().data());
                                    if(t)
                                        scope_tpls.push_back(std::move(t));
                                }
                                if(fem_)
                                {
                                    auto t = fem_->derive_solve_scope(
                                        sim_config_->ctx().view().data(),
                                        linear_system_->ctx().view().data());
                                    if(t)
                                        scope_tpls.push_back(std::move(t));
                                }

                                std::vector<cudaGraphNode_t> scope_deps = {n_memset_grad, n_memset_trip};
                                if(!scope_tpls.empty())
                                    scope_deps = {
                                        cp.subgraph(Subgraph::parallel(std::move(scope_tpls)), scope_deps)
                                            .tail()};

                                // --------------------------------------------------------
                                // ABD and FEM Kinetic/Elastic Assembly
                                // --------------------------------------------------------
                                std::vector<cudaGraphNode_t> assemble_deps;
                                if(abd_)
                                {
                                    auto n_abd_assemble =
                                        cp.subgraph(
                                              abd_->assemble(
                                                  sim_config_->ctx().view().data(),
                                                  linear_system_->ctx().view().data()),
                                              scope_deps)
                                            .tail();
                                    assemble_deps.push_back(n_abd_assemble);
                                }

                                if(rigid_)
                                {
                                    auto n_rigid_assemble =
                                        cp.subgraph(
                                              rigid_->assemble(
                                                  sim_config_->ctx().view().data(),
                                                  linear_system_->ctx().view().data()),
                                              scope_deps)
                                            .tail();
                                    assemble_deps.push_back(n_rigid_assemble);
                                }

                                if(fem_)
                                {
                                    auto n_fem_assemble =
                                        cp.subgraph(
                                              fem_->assemble(
                                                  sim_config_->ctx().view().data(),
                                                  linear_system_->ctx().view().data()),
                                              scope_deps)
                                            .tail();
                                    assemble_deps.push_back(n_fem_assemble);
                                }

                                if(bspline_)
                                {
                                    auto n_bspline_assemble =
                                        cp.subgraph(
                                              bspline_->assemble(
                                                  sim_config_->ctx().view().data(),
                                                  linear_system_->ctx().view().data()),
                                              scope_deps)
                                            .tail();
                                    assemble_deps.push_back(n_bspline_assemble);
                                }

                                if(has_joints())
                                {
                                    auto n_joint_assemble =
                                        cp.subgraph(joint_system_->assemble(
                                                        sim_config_->ctx().view(),
                                                        linear_system_->ctx().view()),
                                                    scope_deps)
                                            .tail();
                                    assemble_deps.push_back(n_joint_assemble);
                                }

                                cudaGraphNode_t n_mpm_assemble{};
                                if(mpm_)
                                {
                                    // Grid-row kinetic + elastic (CP0 P2G ran
                                    // before the loop; grid topology is frame-
                                    // fixed). Contact terms arrive through the
                                    // distribute below.
                                    n_mpm_assemble =
                                        cp.subgraph(
                                              mpm_->assemble(
                                                  linear_system_->ctx().view().data()),
                                              scope_deps)
                                            .tail();
                                    assemble_deps.push_back(n_mpm_assemble);
                                }

                                // --------------------------------------------------------
                                // Contact Distribute (reads from ContactSystem buffers)
                                // --------------------------------------------------------
                                if(contact_)
                                {
                                    // --- Distribute hessian (parallel) ---
                                    if(abd_)
                                    {
                                        assemble_deps.push_back(AbdAbdContactRoute::append(
                                            cp,
                                            std::make_tuple(
                                                abd_->ctx().view().data(),
                                                vertex_->ctx().view().data(),
                                                contact_->ctx().view().data(),
                                                linear_system_->ctx().view().data()),
                                            scope_deps));
                                    }

                                    if(abd_ && fem_)
                                    {
                                        // Masks both scopes: ABD rows and FEM columns.
                                        assemble_deps.push_back(AbdFemContactRoute::append(
                                            cp,
                                            std::make_tuple(
                                                abd_->ctx().view().data(),
                                                fem_->ctx().view().data(),
                                                vertex_->ctx().view().data(),
                                                contact_->ctx().view().data(),
                                                linear_system_->ctx().view().data()),
                                            scope_deps));
                                    }

                                    if(rigid_)
                                    {
                                        assemble_deps.push_back(RigidRigidContactRoute::append(
                                            cp,
                                            std::make_tuple(
                                                rigid_->ctx().view().data(),
                                                vertex_->ctx().view().data(),
                                                contact_->ctx().view().data(),
                                                linear_system_->ctx().view().data()),
                                            scope_deps));
                                    }

                                    if(abd_ && rigid_)
                                    {
                                        assemble_deps.push_back(AbdRigidContactRoute::append(
                                            cp,
                                            std::make_tuple(
                                                abd_->ctx().view().data(),
                                                rigid_->ctx().view().data(),
                                                vertex_->ctx().view().data(),
                                                contact_->ctx().view().data(),
                                                linear_system_->ctx().view().data()),
                                            scope_deps));
                                    }

                                    if(rigid_ && fem_)
                                    {
                                        assemble_deps.push_back(RigidFemContactRoute::append(
                                            cp,
                                            std::make_tuple(
                                                rigid_->ctx().view().data(),
                                                fem_->ctx().view().data(),
                                                vertex_->ctx().view().data(),
                                                contact_->ctx().view().data(),
                                                linear_system_->ctx().view().data()),
                                            scope_deps));
                                    }

                                    if(fem_)
                                    {
                                        assemble_deps.push_back(FemFemContactRoute::append(
                                            cp,
                                            std::make_tuple(
                                                fem_->ctx().view().data(),
                                                contact_->ctx().view().data(),
                                                linear_system_->ctx().view().data()),
                                            scope_deps));
                                    }

                                    if(fem_ && bspline_)
                                    {
                                        assemble_deps.push_back(FemBSplineContactRoute::append(
                                            cp,
                                            std::make_tuple(
                                                fem_->ctx().view().data(),
                                                bspline_->ctx().view().data(),
                                                contact_->ctx().view().data(),
                                                linear_system_->ctx().view().data()),
                                            scope_deps));
                                    }

                                    if(abd_ && bspline_)
                                    {
                                        assemble_deps.push_back(AbdBSplineContactRoute::append(
                                            cp,
                                            std::make_tuple(
                                                abd_->ctx().view().data(),
                                                bspline_->ctx().view().data(),
                                                vertex_->ctx().view().data(),
                                                contact_->ctx().view().data(),
                                                linear_system_->ctx().view().data()),
                                            scope_deps));
                                    }

                                    if(rigid_ && bspline_)
                                    {
                                        assemble_deps.push_back(RigidBSplineContactRoute::append(
                                            cp,
                                            std::make_tuple(
                                                rigid_->ctx().view().data(),
                                                bspline_->ctx().view().data(),
                                                vertex_->ctx().view().data(),
                                                contact_->ctx().view().data(),
                                                linear_system_->ctx().view().data()),
                                            scope_deps));
                                    }
                                    if(bspline_contact_route_)
                                        assemble_deps.push_back(
                                            cp.subgraph(bspline_contact_route_->reduce(
                                                            linear_system_->ctx()
                                                                .view()
                                                                .data()),
                                                        scope_deps)
                                                .tail());

                                    // --- Twist (spin) friction distribute ---
                                    // (docs/rod_twist_friction_plan.md section 5)
                                    //
                                    // On scope_deps like the rest of the block:
                                    // fem_twist masks fem->is_fixed and abd_twist
                                    // masks abd->is_fixed. twist_twist and the twist
                                    // gradient only mask theta_fixed (host-written,
                                    // static within the solve), but the whole block
                                    // shares one parent -- the rule is "contact
                                    // distribution runs after the scope", not a
                                    // per-kernel audit of which flag it reads.
                                    if(contact_->has_twist_friction())
                                    {
                                        QIPC_ASSERT(twist_,
                                                    "twist friction wired but no "
                                                    "KirchhoffRodTwisting system");
                                        assemble_deps.push_back(FemTwistContactRoute::append(
                                            cp,
                                            std::make_tuple(
                                                fem_->ctx().view().data(),
                                                twist_->ctx().view().data(),
                                                contact_->ctx().view().data(),
                                                linear_system_->ctx().view().data()),
                                            scope_deps));

                                        assemble_deps.push_back(TwistTwistContactRoute::append(
                                            cp,
                                            std::make_tuple(
                                                twist_->ctx().view().data(),
                                                contact_->ctx().view().data(),
                                                linear_system_->ctx().view().data()),
                                            scope_deps));

                                        if(abd_)
                                        {
                                            assemble_deps.push_back(AbdTwistContactRoute::append(
                                                cp,
                                                std::make_tuple(
                                                    abd_->ctx().view().data(),
                                                    twist_->ctx().view().data(),
                                                    vertex_->ctx().view().data(),
                                                    contact_->ctx().view().data(),
                                                    linear_system_->ctx().view().data()),
                                                scope_deps));
                                        }

                                        assemble_deps.push_back(TwistGradientContactRoute::append(
                                            cp,
                                            std::make_tuple(
                                                twist_->ctx().view().data(),
                                                contact_->ctx().view().data(),
                                                linear_system_->ctx().view().data()),
                                            scope_deps));
                                    }

                                    // --- Distribute gradient (parallel) ---
                                    if(abd_)
                                    {
                                        assemble_deps.push_back(AbdGradientContactRoute::append(
                                            cp,
                                            std::make_tuple(
                                                abd_->ctx().view().data(),
                                                vertex_->ctx().view().data(),
                                                contact_->ctx().view().data(),
                                                linear_system_->ctx().view().data()),
                                            scope_deps));
                                    }

                                    if(rigid_)
                                    {
                                        assemble_deps.push_back(RigidGradientContactRoute::append(
                                            cp,
                                            std::make_tuple(
                                                rigid_->ctx().view().data(),
                                                vertex_->ctx().view().data(),
                                                contact_->ctx().view().data(),
                                                linear_system_->ctx().view().data()),
                                            scope_deps));
                                    }

                                    if(fem_)
                                    {
                                        assemble_deps.push_back(FemGradientContactRoute::append(
                                            cp,
                                            std::make_tuple(
                                                fem_->ctx().view().data(),
                                                contact_->ctx().view().data(),
                                                linear_system_->ctx().view().data()),
                                            scope_deps));
                                    }
                                    if(bspline_)
                                    {
                                        assemble_deps.push_back(BSplineGradientContactRoute::append(
                                            cp,
                                            std::make_tuple(
                                                bspline_->ctx().view().data(),
                                                contact_->ctx().view().data(),
                                                linear_system_->ctx().view().data()),
                                            scope_deps));
                                    }

                                    // MPM distribute chain-rules the vertex-
                                    // space contact terms onto grid rows. Both
                                    // depend on the MPM assemble: the gradient
                                    // gather ASSIGNS the grid rows, and the
                                    // distribute atomically appends to them.
                                    if(mpm_ && mpm_->n_particles() > 0)
                                    {
                                        assemble_deps.push_back(MpmMpmContactRoute::append(
                                            cp,
                                            std::make_tuple(
                                                mpm_->ctx().view().data(),
                                                contact_->ctx().view().data(),
                                                linear_system_->ctx().view().data()),
                                            {n_mpm_assemble}));

                                        assemble_deps.push_back(MpmGradientContactRoute::append(
                                            cp,
                                            std::make_tuple(
                                                mpm_->ctx().view().data(),
                                                contact_->ctx().view().data(),
                                                linear_system_->ctx().view().data()),
                                            {n_mpm_assemble}));

                                        // Mixed pairs: sand against a mesh.
                                        // Without these the Newton keeps a
                                        // correct gradient but drops the
                                        // coupling blocks, degrading to block
                                        // Jacobi exactly along the direction
                                        // the sand pushes the body.
                                        if(abd_)
                                        {
                                            assemble_deps.push_back(AbdMpmContactRoute::append(
                                                cp,
                                                std::make_tuple(
                                                    abd_->ctx().view().data(),
                                                    mpm_->ctx().view().data(),
                                                    vertex_->ctx().view().data(),
                                                    contact_->ctx().view().data(),
                                                    linear_system_->ctx().view().data()),
                                                {n_mpm_assemble}));
                                        }
                                        if(rigid_)
                                        {
                                            assemble_deps.push_back(RigidMpmContactRoute::append(
                                                cp,
                                                std::make_tuple(
                                                    rigid_->ctx().view().data(),
                                                    mpm_->ctx().view().data(),
                                                    vertex_->ctx().view().data(),
                                                    contact_->ctx().view().data(),
                                                    linear_system_->ctx().view().data()),
                                                {n_mpm_assemble}));
                                        }
                                        if(fem_)
                                        {
                                            assemble_deps.push_back(FemMpmContactRoute::append(
                                                cp,
                                                std::make_tuple(
                                                    fem_->ctx().view().data(),
                                                    mpm_->ctx().view().data(),
                                                    contact_->ctx().view().data(),
                                                    linear_system_->ctx().view().data()),
                                                {n_mpm_assemble}));
                                        }
                                        if(bspline_)
                                        {
                                            assemble_deps.push_back(BSplineMpmContactRoute::append(
                                                cp,
                                                std::make_tuple(
                                                    bspline_->ctx().view().data(),
                                                    mpm_->ctx().view().data(),
                                                    contact_->ctx().view().data(),
                                                    linear_system_->ctx().view().data()),
                                                {n_mpm_assemble}));
                                        }
                                    }
                                }

                                // Affine-cluster relocation: after every writer,
                                // before sort/reduce (docs/affine_cluster.md 3.4).
                                if(cluster_manager_)
                                {
                                    auto n_reloc = ContactPostAssemblyRoute::append(
                                        cp,
                                        cluster_manager_->relocate(
                                            linear_system_->ctx().view().data()),
                                        assemble_deps);
                                    assemble_deps = {n_reloc};
                                }

                                auto n_body_sort_reduce =
                                    cp.subgraph(linear_system_->body_sort_reduce(), assemble_deps)
                                        .tail();

                                // DumpDiag per-Newton BCOO/rhs snapshot: reads
                                // only, parallel with the preconditioner build.
                                // Deliberately not joined into the PCG deps (the
                                // PCIe copy must not serialize the solve); the
                                // Newton WHILE body completes as a whole before
                                // the next iteration's assemble overwrites the
                                // BCOO, so the ordering is safe.
                                if(dump_diag_ && dump_diag_->dump_system_enabled())
                                    cp.subgraph(dump_diag_->record_system(),
                                                {n_body_sort_reduce});

                                // === PRECONDITIONER BUILD (parallel fork) ===

                                // Diag preconditioners (ABD/FEM/rigid/twist/MPM)
                                // run inside ONE fused JIT BCOO traverse scan:
                                // parallel(memset) -> scan -> parallel(invert).
                                // Tree and MAS keep their own standalone scans
                                // (heavy per-entry work).
                                std::vector<cudaGraphNode_t> precond_done;
                                if(linear_system_->has_traverse_visitors())
                                    precond_done.push_back(
                                        cp.subgraph(linear_system_->traverse(), {n_body_sort_reduce})
                                            .tail());
                                if(tree_precond_)
                                {
                                    auto tpl = tree_precond_->build();
                                    precond_done.push_back(
                                        cp.subgraph(tpl, {n_body_sort_reduce}).tail());
                                }
                                if(rigid_tree_precond_)
                                {
                                    auto tpl = rigid_tree_precond_->build();
                                    precond_done.push_back(
                                        cp.subgraph(tpl, {n_body_sort_reduce}).tail());
                                }
                                if(mas_)
                                {
                                    auto tpl = mas_->build();
                                    precond_done.push_back(
                                        cp.subgraph(tpl, {n_body_sort_reduce}).tail());
                                }
                                // === PCG SECTION ===

                                auto pcg_precond_view =
                                    linear_system_->pcg_solver().precond_context();
                                Subgraph precond_apply_tpl(
                                    [this, pcg_precond_view](
                                        SubgraphScope& sg,
                                        const std::vector<cudaGraphNode_t>& deps) -> std::vector<cudaGraphNode_t>
                                    {
                                        std::vector<cudaGraphNode_t> tails;
                                        if(tree_precond_)
                                            tails.push_back(
                                                sg.subgraph(tree_precond_->apply(
                                                                pcg_precond_view.data()),
                                                            deps)
                                                    .tail());
                                        else if(abd_precond_)
                                            tails.push_back(
                                                sg.subgraph(abd_precond_->apply(
                                                                pcg_precond_view.data()),
                                                            deps)
                                                    .tail());
                                        if(mas_)
                                            tails.push_back(
                                                sg.subgraph(mas_->apply(
                                                                pcg_precond_view.data()),
                                                            deps)
                                                    .tail());
                                        else if(fem_precond_)
                                            tails.push_back(
                                                sg.subgraph(fem_precond_->apply(
                                                                pcg_precond_view.data()),
                                                            deps)
                                                    .tail());
                                        if(bspline_precond_)
                                            tails.push_back(
                                                sg.subgraph(bspline_precond_->apply(
                                                                pcg_precond_view.data()),
                                                            deps)
                                                    .tail());
                                        if(rigid_precond_)
                                            tails.push_back(
                                                sg.subgraph(rigid_precond_->apply(
                                                                pcg_precond_view.data()),
                                                            deps)
                                                    .tail());
                                        else if(rigid_tree_precond_)
                                            tails.push_back(
                                                sg.subgraph(rigid_tree_precond_->apply(
                                                                pcg_precond_view.data()),
                                                            deps)
                                                    .tail());
                                        // DOF-owning extra constitutions (twist partition)
                                        // apply their own preconditioner block.
                                        if(fem_)
                                            for(auto* c : fem_->dof_owning_constitutions())
                                                tails.push_back(
                                                    sg.subgraph(
                                                          c->precond_apply(
                                                              pcg_precond_view.data()),
                                                          deps)
                                                        .tail());
                                        if(mpm_)
                                            tails.push_back(
                                                sg.subgraph(mpm_->precond_apply(
                                                                pcg_precond_view.data()),
                                                            deps)
                                                    .tail());
                                        QIPC_ASSERT(!tails.empty(),
                                                    "precond_apply: no preconditioner registered. "
                                                    "Add ABDDiagPreconditioner/FEMDiagPreconditioner via add_system().");
                                        return tails;
                                    });
                                // matrix-free matvec: collected from FEM
                                // constitutions, executed in parallel with SpMV.
                                auto& pcg_solver = linear_system_->pcg_solver();
                                if(dump_diag_)
                                    pcg_solver.set_diag_hook(dump_diag_->pcg_hook());
                                Subgraph matvec_tpl{};
                                if(fem_)
                                {
                                    auto mv_ctx = pcg_solver.matvec_context();
                                    std::vector<Subgraph> hooks;
                                    for(auto* c : fem_->constitutions())
                                        if(auto h = c->matvec(mv_ctx))
                                            hooks.push_back(std::move(h));
                                    if(!hooks.empty())
                                        matvec_tpl = Subgraph::parallel(hooks);
                                }
                                QIPC_ASSERT(!precond_done.empty(),
                                            "precond_build: no preconditioner registered. "
                                            "Add ABDDiagPreconditioner/FEMDiagPreconditioner via add_system().");

                                Subgraph pcg_tpl;
                                auto     pcg_deps = precond_done;
                                // TEMP DIAG: stage bisection mask for the
                                // coarsening overhead investigation. 0 = no
                                // coarsening nodes, 1 = tag+gate, 3 = +grad,
                                // 7 = +hess, 15 (default) = full pipeline.
                                int32_t coarsen_mask = 15;
                                if(const char* e = std::getenv("QIPC_COARSEN_STAGE_MASK"))
                                    coarsen_mask = std::atoi(e);
                                if(coarsening_ && coarsen_mask != 15)
                                    logger().info("coarsening stage mask (diag): {}",
                                                  coarsen_mask);
                                if(coarsening_ && (coarsen_mask & 1))
                                {
                                    // AGIPC in-solve coarsening: aggregate ->
                                    // Galerkin projection -> coarse solve +
                                    // prolongation, then a warm-started fine
                                    // refinement CG under its own budget
                                    // (docs/adaptive_coarsening.md section 7).
                                    // Runs parallel with the fine
                                    // preconditioner builds above.
                                    auto n_agg =
                                        cp.subgraph(coarsening_->tag_and_aggregate(),
                                                    {n_body_sort_reduce})
                                            .tail();
                                    // Fine budget: engine init repointed the
                                    // device max_pcg_iter at post_cg_max_iter
                                    // (init_newton arms the refinement
                                    // budget); the gate below restores the
                                    // full budget on bailed-out iterations
                                    // (skip flag from tag_and_aggregate).
                                    auto n_gate =
                                        cp.subgraph(
                                              coarsening_->fine_budget_gate(
                                                  pcg_solver.context().data(),
                                                  sim_config_->ctx().view().data()),
                                              {n_agg})
                                            .tail();
                                    pcg_deps.push_back(n_gate);
                                    std::vector<cudaGraphNode_t> proj_done;
                                    if(coarsen_mask & 2)
                                        proj_done.push_back(
                                            cp.subgraph(coarsening_->project_gradient(), {n_agg})
                                                .tail());
                                    if(coarsen_mask & 4)
                                        proj_done.push_back(
                                            cp.subgraph(coarsening_->project_hessian(), {n_agg})
                                                .tail());
                                    if(coarsen_mask & 8)
                                    {
                                        auto n_coarse =
                                            cp.subgraph(coarsening_->coarse_solve(), proj_done)
                                                .tail();
                                        pcg_deps.push_back(n_coarse);
                                        pcg_tpl = pcg_solver.solve_warm(precond_apply_tpl,
                                                                        matvec_tpl);
                                    }
                                    else
                                    {
                                        for(auto n : proj_done)
                                            pcg_deps.push_back(n);
                                        pcg_tpl = pcg_solver.solve(precond_apply_tpl, matvec_tpl);
                                    }
                                }
                                else
                                {
                                    pcg_tpl = pcg_solver.solve(precond_apply_tpl, matvec_tpl);
                                }
                                auto pcg_while = cp.subgraph(pcg_tpl, pcg_deps).tail();

                                // === POST-PCG SECTION ===

                                // Record start point: for safe rollback in line search
                                std::vector<cudaGraphNode_t> record_start_point_deps;
                                {
                                    GridStrideKernel<record_vert_start_point_kernel> k_rvsp;
                                    auto n_rvsp = cp.kernel(
                                        k_rvsp,
                                        std::make_tuple(vertex_->ctx().view().data()),
                                        {pcg_while});
                                    record_start_point_deps.push_back(n_rvsp);

                                    if(abd_)
                                    {
                                        auto n_abd_rvsp =
                                            cp.subgraph(abd_->record_start_point(), {pcg_while})
                                                .tail();
                                        record_start_point_deps.push_back(n_abd_rvsp);
                                    }

                                    if(rigid_)
                                    {
                                        auto n_rigid_rvsp =
                                            cp.subgraph(rigid_->record_start_point(),
                                                        {pcg_while})
                                                .tail();
                                        record_start_point_deps.push_back(n_rigid_rvsp);
                                    }

                                    if(fem_)
                                    {
                                        auto n_fem_rvsp =
                                            cp.subgraph(fem_->record_start_point(), {pcg_while})
                                                .tail();
                                        record_start_point_deps.push_back(n_fem_rvsp);
                                    }

                                    if(bspline_)
                                    {
                                        auto n_bspline_record =
                                            cp.subgraph(bspline_->record_start_point(),
                                                        {pcg_while})
                                                .tail();
                                        record_start_point_deps.push_back(n_bspline_record);
                                    }

                                    if(mpm_)
                                    {
                                        auto n_mpm_rvsp =
                                            cp.subgraph(mpm_->record_start_point(), {pcg_while})
                                                .tail();
                                        record_start_point_deps.push_back(n_mpm_rvsp);
                                    }
                                }

                                cudaGraphNode_t n_abd_direction{};
                                cudaGraphNode_t n_rigid_direction{};
                                cudaGraphNode_t n_fem_direction{};
                                cudaGraphNode_t n_bspline_direction{};
                                cudaGraphNode_t n_mpm_direction{};
                                if(abd_)
                                    n_abd_direction =
                                        cp.subgraph(
                                              abd_->negate_dq(
                                                  sim_config_->ctx().view().data(),
                                                  linear_system_->ctx().view().data()),
                                              {pcg_while})
                                            .tail();
                                if(rigid_)
                                    n_rigid_direction =
                                        cp.subgraph(
                                              rigid_->negate_dq(
                                                  sim_config_->ctx().view().data(),
                                                  linear_system_->ctx().view().data()),
                                              {pcg_while})
                                            .tail();
                                if(fem_)
                                {
                                    n_fem_direction =
                                        cp.subgraph(
                                              fem_->negate_dx(
                                                  linear_system_->ctx().view().data()),
                                              {pcg_while})
                                            .tail();
                                    if(cluster_manager_)
                                    {
                                        std::vector<cudaGraphNode_t> deps = {n_fem_direction};
                                        if(abd_)
                                            deps.push_back(n_abd_direction);
                                        if(rigid_)
                                            deps.push_back(n_rigid_direction);
                                        n_fem_direction =
                                            cp.subgraph(cluster_manager_->transport(), deps)
                                                .tail();
                                    }
                                }
                                if(mpm_)
                                    n_mpm_direction =
                                        cp.subgraph(
                                              mpm_->negate_du(
                                                  linear_system_->ctx().view().data()),
                                              {pcg_while})
                                            .tail();
                                if(bspline_)
                                    n_bspline_direction =
                                        cp.subgraph(
                                              bspline_->negate_dq(
                                                  linear_system_->ctx().view().data()),
                                              {pcg_while})
                                            .tail();

                                // Each subsystem contributes its unclamped
                                // world-space Newton motion directly to the
                                // shared free/contact max reduction.
                                GridStrideKernel<reset_newton_convergence_partials_kernel> k_reset_conv;
                                auto n_reset_conv = cp.kernel(
                                    k_reset_conv,
                                    std::make_tuple(sim_config_->ctx().view().data()),
                                    {pcg_while});
                                std::vector<cudaGraphNode_t> convergence_tails;
                                if(abd_)
                                    convergence_tails.push_back(
                                        cp.subgraph(
                                              abd_->contribute_newton_max_disp(
                                                  sim_config_->ctx().view().data(),
                                                  vertex_->ctx().view().data()),
                                              {n_reset_conv, n_abd_direction})
                                            .tail());
                                if(rigid_)
                                    convergence_tails.push_back(
                                        cp.subgraph(
                                              rigid_->contribute_newton_max_disp(
                                                  sim_config_->ctx().view().data(),
                                                  vertex_->ctx().view().data()),
                                              {n_reset_conv, n_rigid_direction})
                                            .tail());
                                if(fem_)
                                    convergence_tails.push_back(
                                        cp.subgraph(
                                              fem_->contribute_newton_max_disp(
                                                  sim_config_->ctx().view().data(),
                                                  vertex_->ctx().view().data()),
                                              {n_reset_conv, n_fem_direction})
                                            .tail());
                                if(bspline_)
                                    convergence_tails.push_back(
                                        cp.subgraph(
                                              bspline_->contribute_newton_max_disp(
                                                  sim_config_->ctx().view().data(),
                                                  vertex_->ctx().view().data()),
                                              {n_reset_conv, n_bspline_direction})
                                            .tail());
                                if(mpm_)
                                    convergence_tails.push_back(
                                        cp.subgraph(
                                              mpm_->contribute_newton_max_disp(
                                                  sim_config_->ctx().view().data(),
                                                  vertex_->ctx().view().data()),
                                              {n_reset_conv, n_mpm_direction})
                                            .tail());
                                if(convergence_tails.empty())
                                    convergence_tails.push_back(n_reset_conv);

                                auto n_conv = cp.kernel<pre_ccd_convergence_check_kernel>(
                                    std::make_tuple(
                                        sim_config_->ctx().view().data(),
                                        contact_ ? contact_->ctx().view().data() :
                                                   (const GlobalContactContext*)nullptr),
                                    1,
                                    PCG_BLOCK_DIM,
                                    convergence_tails);

                                // One global scalar clamps every subsystem's
                                // generalized direction. Each subsystem then
                                // publishes its exact alpha=1 endpoint.
                                cudaGraphNode_t n_clamp_factor = n_conv;
                                if(contact_)
                                    n_clamp_factor = cp.kernel<compute_clamp_alpha_kernel>(
                                        std::make_tuple(
                                            sim_config_->ctx().view().data(),
                                            contact_->ctx().view().data()),
                                        1,
                                        1,
                                        {n_conv});

                                std::vector<cudaGraphNode_t> trajectory_tails;
                                cudaGraphNode_t              n_rigid_clamp{};
                                cudaGraphNode_t              n_fem_endpoint{};
                                if(abd_)
                                {
                                    auto n_clamp =
                                        contact_ ?
                                            cp.subgraph(
                                                  abd_->clamp_newton_direction(
                                                      sim_config_->ctx().view().data()),
                                                  {n_clamp_factor})
                                                .tail() :
                                            n_abd_direction;
                                    trajectory_tails.push_back(
                                        cp.subgraph(abd_->publish_trajectory_end_positions(
                                                        vertex_->ctx().view().data()),
                                                    {n_clamp})
                                            .tail());
                                }
                                if(rigid_)
                                {
                                    n_rigid_clamp =
                                        contact_ ?
                                            cp.subgraph(
                                                  rigid_->clamp_newton_direction(
                                                      sim_config_->ctx().view().data()),
                                                  {n_clamp_factor})
                                                .tail() :
                                            n_rigid_direction;
                                    trajectory_tails.push_back(
                                        cp.subgraph(rigid_->publish_global_vertex_trajectory(
                                                        vertex_->ctx().view().data()),
                                                    {n_rigid_clamp})
                                            .tail());
                                }
                                if(fem_)
                                {
                                    auto n_clamp =
                                        contact_ ?
                                            cp.subgraph(
                                                  fem_->clamp_newton_direction(
                                                      sim_config_->ctx().view().data()),
                                                  {n_clamp_factor})
                                                .tail() :
                                            n_fem_direction;
                                    n_fem_endpoint =
                                        cp.subgraph(fem_->publish_trajectory_end_positions(
                                                        vertex_->ctx().view().data()),
                                                    {n_clamp})
                                            .tail();
                                    if(!(cluster_manager_ && rigid_))
                                        trajectory_tails.push_back(n_fem_endpoint);
                                }
                                if(mpm_ && mpm_->n_particles() > 0)
                                {
                                    auto n_clamp =
                                        contact_ ?
                                            cp.subgraph(
                                                  mpm_->clamp_newton_direction(
                                                      sim_config_->ctx().view().data()),
                                                  {n_clamp_factor})
                                                .tail() :
                                            n_mpm_direction;
                                    trajectory_tails.push_back(
                                        cp.subgraph(mpm_->publish_trajectory_end_positions(
                                                        vertex_->ctx().view().data()),
                                                    {n_clamp})
                                            .tail());
                                }
                                if(bspline_)
                                {
                                    auto n_clamp =
                                        contact_ ?
                                            cp.subgraph(
                                                  bspline_->clamp_newton_direction(
                                                      sim_config_->ctx().view().data()),
                                                  {n_clamp_factor})
                                                .tail() :
                                            n_bspline_direction;
                                    trajectory_tails.push_back(
                                        cp.subgraph(bspline_->publish_trajectory_end_positions(
                                                        vertex_->ctx().view().data()),
                                                    {n_clamp})
                                            .tail());
                                }
                                if(cluster_manager_ && rigid_ && fem_)
                                    trajectory_tails.push_back(
                                        cp.subgraph(cluster_manager_->publish_rigid_rider_trajectory(
                                                        vertex_->ctx().view().data()),
                                                    {n_fem_endpoint, n_rigid_clamp})
                                            .tail());

                                // OPT-2: E0 only on first Newton iteration.
                                // For iterations 2+, finalize_newton_iter copies
                                // E_trial -> E0 (valid within frame).
                                auto n_cond_e0 = cp.condition(
                                    first_newton_flag_.view(),
                                    {pcg_while},
                                    [&](ConditionScope& e0)
                                    {
                                        std::vector<cudaGraphNode_t> sum_deps;

                                        if(abd_)
                                        {
                                            auto n_abd_ef =
                                                e0.subgraph(abd_->energy(sim_config_
                                                                             ->ctx()
                                                                             .view()
                                                                             .data()),
                                                            {})
                                                    .tail();
                                            sum_deps.push_back(n_abd_ef);
                                        }

                                        if(rigid_)
                                        {
                                            auto n_rigid_ef =
                                                e0.subgraph(rigid_->energy(sim_config_
                                                                               ->ctx()
                                                                               .view()
                                                                               .data()),
                                                            {})
                                                    .tail();
                                            sum_deps.push_back(n_rigid_ef);
                                        }

                                        if(fem_)
                                        {
                                            auto n_fem_ef =
                                                e0.subgraph(fem_->energy(sim_config_
                                                                             ->ctx()
                                                                             .view()
                                                                             .data()),
                                                            {})
                                                    .tail();
                                            sum_deps.push_back(n_fem_ef);
                                        }

                                        if(bspline_)
                                        {
                                            auto n_bspline_energy =
                                                e0.subgraph(
                                                      bspline_->energy(
                                                          sim_config_->ctx()
                                                              .view()
                                                              .data()),
                                                      {})
                                                    .tail();
                                            sum_deps.push_back(n_bspline_energy);
                                        }

                                        if(has_joints())
                                        {
                                            auto n_joint_ef =
                                                e0.subgraph(
                                                      joint_system_->energy(
                                                          sim_config_->ctx().view()),
                                                      {})
                                                    .tail();
                                            auto n_joint_final =
                                                e0.subgraph(joint_system_->energy_final(),
                                                            {n_joint_ef})
                                                    .tail();
                                            sum_deps.push_back(n_joint_final);
                                        }

                                        if(contact_)
                                        {
                                            auto n_ce =
                                                e0.subgraph(
                                                      contact_->contact_energy(
                                                          vertex_->ctx().view().data(),
                                                          surface_->ctx().view().data(),
                                                          sim_config_->ctx()
                                                              .view()
                                                              .data()),
                                                      {})
                                                    .tail();
                                            sum_deps.push_back(n_ce);
                                        }

                                        if(mpm_)
                                        {
                                            auto n_mpm_ef =
                                                e0.subgraph(mpm_->energy(), {}).tail();
                                            sum_deps.push_back(n_mpm_ef);
                                        }

                                        e0.kernel(
                                            KernelConfig{"sum_total_energy"}
                                                .grid(1)
                                                .block(1),
                                            [ctx = sim_config_->ctx().view(),
                                             abd = abd_ ? abd_->ctx().view() :
                                                          VarView<const ABDContext>{},
                                             rigid = rigid_ ? rigid_->ctx().view() :
                                                              VarView<const RigidContext>{},
                                             fem = fem_ ? fem_->ctx().view() :
                                                          VarView<const FEMContext>{},
                                             bspline = bspline_ ?
                                                           bspline_->ctx().view() :
                                                           VarView<const BSplineContext>{},
                                             jctx = (joint_system_
                                                     && !joint_system_->empty()) ?
                                                        joint_system_->ctx().view() :
                                                        VarView<const GlobalJointContext>{},
                                             cctx = contact_ ? contact_->ctx().view() :
                                                               VarView<const GlobalContactContext>{},
                                             mpm = mpm_ ? mpm_->ctx().view() :
                                                          VarView<const MpmContext>{}] __device__() mutable
                                            {
                                                double e = 0.0;
                                                if(abd.data())
                                                    e += *abd->abd_energy_view();
                                                if(rigid.data())
                                                    e += *rigid->rigid_energy_view();
                                                if(fem.data())
                                                    e += *fem->fem_energy_view();
                                                if(bspline.data())
                                                    e += bspline->energy_view()[0];
                                                if(jctx.data())
                                                    e += *jctx->joint_energy_view();
                                                if(cctx.data())
                                                    e += *cctx->contact_total_energy_view();
                                                if(mpm.data())
                                                    e += *mpm->energy_view();
                                                ctx->energy_buf_view()[0] = e;
                                            },
                                            sum_deps);
                                    });

                                // === BVH BUILD (contact only) ===
                                // Must depend on n_cond_e0 to avoid racing with
                                // filter_energy_* in the condition body on n_pairs.
                                if(contact_)
                                {
                                    std::vector<cudaGraphNode_t> bvh_build_deps;
                                    bvh_build_deps.reserve(trajectory_tails.size() + 1);
                                    std::ranges::copy(trajectory_tails,
                                                      std::back_inserter(bvh_build_deps));
                                    bvh_build_deps.push_back(n_cond_e0);

                                    auto n_reset = cp.kernel(
                                        KernelConfig{"reset_collision_count"}
                                            .grid(1)
                                            .block(1),
                                        [cctx = contact_->ctx().view()] __device__() mutable
                                        {
                                            *cctx->n_pairs_pt_view() = 0;
                                            *cctx->n_pairs_ee_view() = 0;
                                            if(cctx->max_pairs_pe > 0)
                                                *cctx->n_pairs_pe_view() = 0;
                                            if(cctx->max_pairs_pp > 0)
                                                *cctx->n_pairs_pp_view() = 0;
                                        },
                                        bvh_build_deps);

                                    {
                                        auto bvh_face_tpl = contact_->bvh_triangle_build(
                                            vertex_->ctx().view().data(),
                                            surface_->ctx().view().data());
                                        if(bvh_face_tpl)
                                            cp.subgraph(bvh_face_tpl, {n_reset});

                                        auto bvh_edge_tpl = contact_->bvh_edge_build(
                                            vertex_->ctx().view().data(),
                                            surface_->ctx().view().data());
                                        if(bvh_edge_tpl)
                                            cp.subgraph(bvh_edge_tpl, {n_reset});

                                        if(contact_->has_codim_point_bvh())
                                        {
                                            auto codim_bvh_tpl = contact_->bvh_codim_point_build(
                                                vertex_->ctx().view().data(),
                                                surface_->ctx().view().data());
                                            cp.subgraph(codim_bvh_tpl, {n_reset});
                                        }
                                    }
                                }
                            });

                        if(contact_)
                        {
                            // ==========================================================
                            // Checkpoint 1: BVH query + overflow check + yield
                            // ==========================================================
                            newton.checkpoint(
                                [&](CheckpointScope& cp)
                                {
                                    cp.subgraph(contact_->trajectory_query(
                                                    vertex_->ctx().view().data(),
                                                    surface_->ctx().view().data(),
                                                    body_mgr_->ctx().view().data(),
                                                    overflow_var),
                                                {});
                                },
                                overflow_var,
                                [this](YieldContext& ctx)
                                {
                                    // when bvh overflows, we need to handle it by reallocating the contact buffers
                                    handle_bvh_overflow(ctx);
                                },
                                // we need to resume from this checkpoint after handling the overflow
                                // because we need to rerun the bvh query to get all the contacts
                                YieldResume::This);

                            // ==========================================================
                            // Checkpoint 2: CCD (contact only) + CCD Component Partition
                            // Yields on CCD edge overflow (re-run after realloc).
                            // ==========================================================
                            newton.checkpoint(
                                [&](CheckpointScope& cp)
                                {
                                    const CCDComponentPartitionerContext* d_ccd_part =
                                        ccd_partitioner_ ?
                                            ccd_partitioner_->ctx().view().data() :
                                            nullptr;

                                    auto ccd_tail = cp.subgraph(
                                        contact_->ccd(vertex_->ctx().view().data(),
                                                      surface_->ctx().view().data(),
                                                      d_ccd_part),
                                        {});

                                    if(ccd_partitioner_)
                                    {
                                        cp.subgraph(ccd_partitioner_->partition_and_reduce(
                                                        vertex_->ctx().view().data(),
                                                        body_mgr_->ctx().view().data()),
                                                    {ccd_tail.tail()});
                                    }
                                },
                                ccd_partitioner_ ? ccd_partitioner_->overflow_flag_var() :
                                                   VarView<int32_t>{},
                                [this](YieldContext& ctx)
                                { handle_ccd_edge_overflow(ctx); },
                                YieldResume::This);
                        }

                        // Export yield (legacy contact export)
                        if(export_flag_var.data())
                        {
                            newton.checkpoint([&](CheckpointScope& /*cp*/) {},
                                              export_flag_var,
                                              [this](YieldContext& ctx)
                                              { handle_contact_export(ctx); },
                                              YieldResume::Next);
                            newton.checkpoint([&](CheckpointScope& /*cp*/) {});
                        }

                        // Contact closure yield (docs/contact_closure.md)
                        if(closure_flag_var.data())
                        {
                            newton.checkpoint([&](CheckpointScope& /*cp*/) {},
                                              closure_flag_var,
                                              [this](YieldContext& ctx)
                                              { handle_contact_closure(ctx); },
                                              YieldResume::Next);
                            newton.checkpoint([&](CheckpointScope& /*cp*/) {});
                        }

                        // ==========================================================
                        // Checkpoint 3: Line Search + Convergence
                        // ==========================================================
                        newton.checkpoint(
                            [&](CheckpointScope& cp)
                            {
                                // Inversion-free line-search filter: cap the step so no tet
                                // (FEM) and no particle / lite cell (MPM) inverts
                                // (det F > 0). Required by the log-strain / log-J elastic
                                // energies, which are undefined / blow up at det <= 0. Runs
                                // after dx is final; the results are min'd into the starting
                                // line-search alpha below.
                                std::vector<cudaGraphNode_t> init_ls_deps;
                                if(fem_)
                                    init_ls_deps.push_back(
                                        cp.subgraph(fem_->inversion_filter(), {})
                                            .tail());
                                if(mpm_ && mpm_->n_particles() > 0)
                                    init_ls_deps.push_back(
                                        cp.subgraph(mpm_->inversion_filter(), {})
                                            .tail());

                                cp.kernel(
                                    KernelConfig{"init_line_search"}.grid(1).block(1),
                                    [ctx = sim_config_->ctx().view(),
                                     cctx = contact_ ? contact_->ctx().view() :
                                                       VarView<const GlobalContactContext>{},
                                     fem_ctx = fem_ ? fem_->ctx().view() :
                                                      VarView<const FEMContext>{},
                                     mpm_ctx = (mpm_ && mpm_->n_particles() > 0) ?
                                                   mpm_->ctx().view() :
                                                   VarView<const MpmContext>{},
                                     has_ccd_part = (ccd_partitioner_ != nullptr),
                                     ls_ex_flag = ls_forensics_flag_.data() ?
                                                      ls_forensics_flag_.view() :
                                                      VarView<int32_t>{},
                                     ls_cond = ls_cond_.view()] __device__() mutable
                                    {
                                        // LS forensics: the flag reflects THIS
                                        // iteration's line search only.
                                        if(ls_ex_flag.data())
                                            *ls_ex_flag = 0;

                                        double alpha = 1.0;
                                        if(!has_ccd_part && cctx.data())
                                        {
                                            double ccd_alpha = *cctx->ccd_alpha_view();
                                            if(ccd_alpha > 0.0 && ccd_alpha < 1.0)
                                                alpha = ccd_alpha;
                                        }
                                        if(fem_ctx.data())
                                        {
                                            double inv_alpha =
                                                *fem_ctx->inversion_alpha_view();
                                            if(inv_alpha > 0.0 && inv_alpha < alpha)
                                                alpha = inv_alpha;
                                        }
                                        if(mpm_ctx.data())
                                        {
                                            double inv_alpha =
                                                *mpm_ctx->inversion_alpha_view();
                                            if(inv_alpha > 0.0 && inv_alpha < alpha)
                                                alpha = inv_alpha;
                                        }
                                        *ctx->alpha_view() = alpha;
                                        *ctx->ls_iter_remaining_view() =
                                            *ctx->max_ls_iter_view();

                                        *ls_cond = 1;
                                    },
                                    init_ls_deps);

                                auto ls_while = cp.loop(
                                    VarView<const int32_t>(ls_cond_.view().data()),
                                    [&](LoopScope& ls)
                                    {
                                        const CCDComponentPartitionerContext* d_ccd_part_ls =
                                            ccd_partitioner_ ?
                                                ccd_partitioner_->ctx().view().data() :
                                                nullptr;
                                        std::vector<cudaGraphNode_t> sum_trial_deps;
                                        std::vector<cudaGraphNode_t> trial_pos_deps;
                                        std::vector<cudaGraphNode_t> joint_state_deps;
                                        cudaGraphNode_t n_rigid_step{};

                                        if(abd_)
                                        {
                                            auto n_step =
                                                ls.subgraph(
                                                      abd_->step_forward(
                                                          sim_config_->ctx()
                                                              .view()
                                                              .data(),
                                                          d_ccd_part_ls),
                                                      {})
                                                    .tail();
                                            joint_state_deps.push_back(n_step);
                                            trial_pos_deps.push_back(
                                                ls.subgraph(
                                                      abd_->forward_global_vertices(
                                                          vertex_->ctx().view().data()),
                                                      {n_step})
                                                    .tail());

                                            auto n_abd_ef =
                                                ls.subgraph(abd_->energy(sim_config_
                                                                             ->ctx()
                                                                             .view()
                                                                             .data()),
                                                            {n_step})
                                                    .tail();

                                            sum_trial_deps.push_back(n_abd_ef);
                                        }

                                        if(rigid_)
                                        {
                                            n_rigid_step =
                                                ls.subgraph(
                                                      rigid_->step_forward(
                                                          sim_config_->ctx()
                                                              .view()
                                                              .data(),
                                                          d_ccd_part_ls),
                                                      {})
                                                    .tail();
                                            joint_state_deps.push_back(n_rigid_step);

                                            auto n_rigid_pos =
                                                ls.subgraph(
                                                      rigid_->forward_global_vertices(
                                                          vertex_->ctx().view().data()),
                                                      {n_rigid_step})
                                                    .tail();

                                            auto n_rigid_ef =
                                                ls.subgraph(rigid_->energy(sim_config_
                                                                               ->ctx()
                                                                               .view()
                                                                               .data()),
                                                            {n_rigid_step})
                                                    .tail();

                                            sum_trial_deps.push_back(n_rigid_ef);
                                            trial_pos_deps.push_back(n_rigid_pos);
                                        }

                                        if(fem_)
                                        {
                                            auto n_fem_step =
                                                ls.subgraph(
                                                      fem_->step_forward(
                                                          sim_config_->ctx()
                                                              .view()
                                                              .data(),
                                                          vertex_->ctx().view().data(),
                                                          d_ccd_part_ls),
                                                      {})
                                                    .tail();

                                            auto n_fem_positions =
                                                ls.subgraph(
                                                      fem_->forward_global_vertices(
                                                          vertex_->ctx().view().data()),
                                                      {n_fem_step})
                                                    .tail();
                                            cudaGraphNode_t n_fem_energy_dep = n_fem_step;
                                            if(cluster_manager_ && rigid_)
                                            {
                                                n_fem_positions =
                                                    ls.subgraph(
                                                          cluster_manager_->forward_rigid_rider_vertices(
                                                              vertex_->ctx()
                                                                  .view()
                                                                  .data()),
                                                          {n_rigid_step, n_fem_positions})
                                                        .tail();
                                                n_fem_energy_dep = n_fem_positions;
                                            }
                                            trial_pos_deps.push_back(n_fem_positions);

                                            auto n_fem_ef =
                                                ls.subgraph(fem_->energy(sim_config_
                                                                             ->ctx()
                                                                             .view()
                                                                             .data()),
                                                            {n_fem_energy_dep})
                                                    .tail();

                                            sum_trial_deps.push_back(n_fem_ef);
                                        }

                                        if(bspline_)
                                        {
                                            auto n_bspline_step =
                                                ls.subgraph(
                                                      bspline_->step_forward(
                                                          sim_config_->ctx()
                                                              .view()
                                                              .data(),
                                                          d_ccd_part_ls),
                                                      {})
                                                    .tail();
                                            auto n_bspline_positions =
                                                ls.subgraph(
                                                      bspline_->forward_global_vertices(
                                                          vertex_->ctx().view().data()),
                                                      {n_bspline_step})
                                                    .tail();
                                            trial_pos_deps.push_back(n_bspline_positions);
                                            sum_trial_deps.push_back(
                                                ls.subgraph(bspline_->energy(sim_config_
                                                                                 ->ctx()
                                                                                 .view()
                                                                                 .data()),
                                                            {n_bspline_step})
                                                    .tail());
                                        }

                                        if(mpm_)
                                        {
                                            auto n_mpm_step =
                                                ls.subgraph(
                                                      mpm_->step_forward(
                                                          sim_config_->ctx()
                                                              .view()
                                                              .data()),
                                                      {})
                                                    .tail();
                                            if(mpm_->n_particles() > 0)
                                                trial_pos_deps.push_back(
                                                    ls.subgraph(
                                                          mpm_->forward_global_vertices(
                                                              vertex_->ctx()
                                                                  .view()
                                                                  .data()),
                                                          {n_mpm_step})
                                                        .tail());

                                            auto n_mpm_ef =
                                                ls.subgraph(mpm_->energy(), {n_mpm_step})
                                                    .tail();

                                            sum_trial_deps.push_back(n_mpm_ef);
                                        }

                                        // Joint energy depends on every body state that may
                                        // have been stepped in this trial.
                                        if(has_joints())
                                        {
                                            auto n_joint_ef =
                                                ls.subgraph(
                                                      joint_system_->energy(
                                                          sim_config_->ctx().view()),
                                                      joint_state_deps)
                                                    .tail();
                                            auto n_joint_final =
                                                ls.subgraph(joint_system_->energy_final(),
                                                            {n_joint_ef})
                                                    .tail();
                                            sum_trial_deps.push_back(n_joint_final);
                                        }

                                        if(contact_)
                                        {
                                            auto n_ce =
                                                ls.subgraph(
                                                      contact_->contact_energy(
                                                          vertex_->ctx().view().data(),
                                                          surface_->ctx().view().data(),
                                                          sim_config_->ctx()
                                                              .view()
                                                              .data()),
                                                      trial_pos_deps)
                                                    .tail();
                                            sum_trial_deps.push_back(n_ce);
                                        }

                                        ls.kernel(
                                            KernelConfig{"check_line_search"}
                                                .grid(1)
                                                .block(1),
                                            [ctx = sim_config_->ctx().view(),
                                             abd = abd_ ? abd_->ctx().view() :
                                                          VarView<const ABDContext>{},
                                             rigid = rigid_ ? rigid_->ctx().view() :
                                                              VarView<const RigidContext>{},
                                             fem = fem_ ? fem_->ctx().view() :
                                                          VarView<const FEMContext>{},
                                             bspline = bspline_ ?
                                                           bspline_->ctx().view() :
                                                           VarView<const BSplineContext>{},
                                             jctx = (joint_system_
                                                     && !joint_system_->empty()) ?
                                                        joint_system_->ctx().view() :
                                                        VarView<const GlobalJointContext>{},
                                             cctx = contact_ ? contact_->ctx().view() :
                                                               VarView<const GlobalContactContext>{},
                                             mpm = mpm_ ? mpm_->ctx().view() :
                                                          VarView<const MpmContext>{},
                                             ls_ex_flag = ls_forensics_flag_.data() ?
                                                              ls_forensics_flag_.view() :
                                                              VarView<int32_t>{},
                                             ls_ex_remaining =
                                                 ls_forensics_remaining_.data() ?
                                                     ls_forensics_remaining_.view() :
                                                     VarView<int32_t>{},
                                             ls_ex_dumped =
                                                 ls_forensics_dumped_step_.data() ?
                                                     ls_forensics_dumped_step_.view() :
                                                     VarView<int32_t>{},
                                             ls_ex_e0 = ls_forensics_e0_.data() ?
                                                            ls_forensics_e0_.view() :
                                                            VarView<double>{},
                                             ls_ex_bias =
                                                 ls_forensics_test_bias_.data() ?
                                                     ls_forensics_test_bias_.view() :
                                                     VarView<double>{},
                                             ls_cond = ls_cond_.view()] __device__() mutable
                                            {
                                                // Sum trial energy (computed locally)
                                                double e = 0.0;
                                                if(abd.data())
                                                    e += *abd->abd_energy_view();
                                                if(rigid.data())
                                                    e += *rigid->rigid_energy_view();
                                                if(fem.data())
                                                    e += *fem->fem_energy_view();
                                                if(bspline.data())
                                                    e += bspline->energy_view()[0];
                                                if(jctx.data())
                                                    e += *jctx->joint_energy_view();
                                                if(cctx.data())
                                                    e += *cctx->contact_total_energy_view();
                                                if(mpm.data())
                                                    e += *mpm->energy_view();

                                                // LS-forensics test hook: bias
                                                // every trial sum (E0 stays
                                                // unbiased) => exactly one
                                                // deterministic exhaustion per
                                                // frame. Null in production.
                                                if(ls_ex_bias.data())
                                                    e += *ls_ex_bias;

                                                auto ebuf = ctx->energy_buf_view();

                                                // If already converged, accept step immediately
                                                if(*ctx->converged_view())
                                                {
                                                    ebuf[0]  = e;
                                                    *ls_cond = 0;
                                                    return;
                                                }

                                                // Backtracking line search
                                                int64_t rem =
                                                    *ctx->ls_iter_remaining_view() - 1;
                                                *ctx->ls_iter_remaining_view() = rem;
                                                int64_t max_ls = *ctx->max_ls_iter_view();
                                                int32_t step =
                                                    (int32_t)(max_ls - 1 - rem);

                                                // ebuf[0]=E0, ebuf[1]=E(alpha=1), ebuf[2]=E(alpha=0.5), ...
                                                ebuf[step + 1] = e;

                                                // Diagnostic variant: bound the
                                                // energy-comparison roundoff;
                                                // physical energy/derivatives
                                                // and CCD remain unchanged.
                                                constexpr double comparison_eps =
                                                    16.0 * 2.2204460492503131e-16;
                                                double energy_limit = ebuf[0]
                                                    + comparison_eps * fabs(ebuf[0]);
                                                bool accept =
                                                    (rem <= 0 || e <= energy_limit);

                                                if(accept)
                                                {
                                                    if(rem <= 0 && e > energy_limit)
                                                    {
                                                        printf(
                                                            "[WARN] line search exhausted "
                                                            "(E1-E0=%.3e, Ef-E0=%.3e, "
                                                            "E0=%.6e, conv=%d, "
                                                            "abs_tol=%.3e)\n",
                                                            ebuf[1] - ebuf[0],
                                                            e - ebuf[0],
                                                            ebuf[0],
                                                            (int)*ctx->converged_view(),
                                                            *ctx->abs_tol_view());

                                                        // LS forensics: raise the
                                                        // export-yield flag. Budget
                                                        // gates run device-side so
                                                        // un-staged exhaustions never
                                                        // pay a graph suspension:
                                                        // one dump per step, at most
                                                        // max_dumps per run. E0 is
                                                        // stashed here because the
                                                        // accept below overwrites
                                                        // ebuf[0] with the trial
                                                        // energy.
                                                        if(ls_ex_flag.data()
                                                           && *ls_ex_remaining > 0
                                                           && *ls_ex_dumped == 0)
                                                        {
                                                            *ls_ex_flag = 1;
                                                            *ls_ex_remaining -= 1;
                                                            *ls_ex_dumped = 1;
                                                            *ls_ex_e0 = ebuf[0];
                                                        }
                                                    }
                                                    ebuf[0]  = e;
                                                    *ls_cond = 0;
                                                }
                                                else
                                                {
                                                    *ctx->alpha_view() *= 0.5;
                                                }
                                            },
                                            sum_trial_deps);
                                    });  // End of line search loop

                                cp.kernel(
                                    KernelConfig{"finalize_newton_iter"}.grid(1).block(1),
                                    [ctx = sim_config_->ctx().view(),
                                     cctx = contact_ ? contact_->ctx().view() :
                                                       VarView<const GlobalContactContext>{},
                                     pcg = linear_system_->pcg_solver().context()] __device__() mutable
                                    {
                                        // E0 is already updated in-place by the LS accept path.

                                        // Update PCG/LS iter diagnostics (single-thread kernel, no atomics needed)
                                        int32_t pcg_rem =
                                            (int32_t)*pcg->pcg_iter_remaining_view();
                                        int32_t max_pcg =
                                            (int32_t)*ctx->max_pcg_iter_view();
                                        int32_t pcg_used =
                                            max(0, min(max_pcg - pcg_rem, max_pcg));
                                        int32_t ls_rem =
                                            (int32_t)*ctx->ls_iter_remaining_view();
                                        int32_t max_ls =
                                            (int32_t)*ctx->max_ls_iter_view();
                                        int32_t ls_used =
                                            max(0, min(max_ls - ls_rem, max_ls));
                                        auto& cur_max_pcg = *ctx->max_pcg_used_view();
                                        auto& cur_max_ls = *ctx->max_ls_used_view();
                                        cur_max_pcg = max(cur_max_pcg, pcg_used);
                                        cur_max_ls = max(cur_max_ls, ls_used);
                                        *ctx->total_pcg_used_view() += pcg_used;
                                        *ctx->total_ls_used_view() += ls_used;

                                        // CCD alpha of this iteration's trajectory
                                        // sweep (1.0 when nothing was in the way).
                                        // With the CCD partitioner the step is
                                        // capped per component, so this records the
                                        // tightest cap anywhere in the scene. One
                                        // sample per Newton iteration, so the sum
                                        // divided by the iteration count is the mean.
                                        double ccd_alpha =
                                            cctx.data() ? *cctx->ccd_alpha_view() : 1.0;
                                        auto& min_ccd = *ctx->min_ccd_alpha_view();
                                        min_ccd = fmin(min_ccd, ccd_alpha);
                                        *ctx->total_ccd_alpha_view() += ccd_alpha;

                                        // Exit Newton if converged or max iterations reached
                                        auto newton_iter = *ctx->newton_iter_view();
                                        auto max_newton_iter =
                                            *ctx->max_newton_iter_view();
                                        auto& newton_cond = *ctx->newton_cond_view();
                                        auto converged = *ctx->converged_view();

                                        newton_cond = 1;  // continue Newton loop

                                        if(converged && newton_iter > 1)
                                        {
                                            newton_cond = 0;  // exit Newton loop
                                            return;
                                        }

                                        if(newton_iter >= max_newton_iter)
                                        {
                                            newton_cond = 0;  // exit Newton loop
                                            printf(
                                                "[WARN] Newton loop exhausted "
                                                "(max_iter=%lld, conv=%d, "
                                                "E=%.6e, max_disp=%.6e "
                                                "(free=%.3e, contact=%.3e), "
                                                "abs_tol=%.3e)\n",
                                                max_newton_iter,
                                                (int)converged,
                                                ctx->energy_buf_view()[0],
                                                *ctx->max_disp_view(),
                                                *ctx->max_disp_free_view(),
                                                *ctx->max_disp_contact_view(),
                                                *ctx->abs_tol_view());
                                            return;
                                        }
                                    },
                                    {ls_while});
                            });

                        // ==========================================================
                        // LS-forensics export yield (extras/ls_forensics): fires
                        // when check_line_search flagged an exhausted line search
                        // this iteration. At this point the mid-Newton state is
                        // intact (safe_positions = iterate,
                        // trajectory_end_positions = finalized t=1 endpoint,
                        // alpha = final trial, contact candidates + lagged
                        // friction unchanged); the handler stages the energy
                        // closure to host and the frame resumes bit-identically
                        // at the empty landing checkpoint below.
                        // ==========================================================
                        if(ls_forensics_flag_.data())
                        {
                            newton.checkpoint([&](CheckpointScope& /*cp*/) {},
                                              ls_forensics_flag_.view(),
                                              [this](YieldContext& ctx)
                                              { handle_ls_forensics(ctx); },
                                              YieldResume::Next);
                            // resume point for the above yield
                            newton.checkpoint([&](CheckpointScope& /*cp*/) {});
                        }

                        // ==========================================================
                        // CP ET-detect: edge-triangle intersection check (debug,
                        // config-gated). Yields on pair-buffer overflow; the
                        // handler grows the buffer and this checkpoint re-runs
                        // (This) until every intersection is stored.
                        // ==========================================================
                        if(et_checker_)
                        {
                            newton.checkpoint(
                                [&](CheckpointScope& cp)
                                {
                                    cp.subgraph(
                                        et_checker_->build_detect_subgraph(
                                            vertex_->ctx().view().data(),
                                            surface_->ctx().view().data(),
                                            body_mgr_ ? body_mgr_->ctx().view().data() : nullptr,
                                            contact_ ? contact_->ctx().view().data() : nullptr),
                                        {});
                                },
                                et_checker_->overflow_flag_var(),
                                [this](YieldContext& ctx)
                                { handle_et_pair_overflow(ctx); },
                                YieldResume::This);

                            // ==========================================================
                            // CP ET-report: yields when intersections were found
                            // after step_forward (complete list -- the detect CP
                            // settled within capacity). Informational: the
                            // handler reports + forces Newton exit, resume skips
                            // to the landing checkpoint.
                            // ==========================================================
                            newton.checkpoint(
                                [&](CheckpointScope& cp)
                                {
                                    cp.subgraph(et_checker_->build_report_flag_subgraph(), {});
                                },
                                et_checker_->yield_flag(),
                                [this](YieldContext& ctx)
                                { handle_et_intersection(ctx); },
                                YieldResume::Next);

                            // Empty landing checkpoint for ET yield resume
                            newton.checkpoint([&](CheckpointScope&) {});
                        }
                    });

                // ==========================================================
                // Checkpoint 4: Update velocity and copy previous state
                // ==========================================================
                m.checkpoint(
                    [&](CheckpointScope& cp)
                    {
                        if(abd_)
                        {
                            auto n_update_vel =
                                cp.subgraph(abd_->update_velocity(
                                                sim_config_->ctx().view().data()),
                                            {})
                                    .tail();
                            cp.subgraph(abd_->copy_q_prev(
                                            sim_config_->ctx().view().data()),
                                        {n_update_vel});
                        }

                        if(rigid_)
                        {
                            auto n_rigid_vel =
                                cp.subgraph(rigid_->update_velocity(
                                                sim_config_->ctx().view().data()),
                                            {})
                                    .tail();
                            cp.subgraph(rigid_->copy_prev(
                                            sim_config_->ctx().view().data()),
                                        {n_rigid_vel});
                        }

                        if(fem_)
                        {
                            auto n_fem_vel =
                                cp.subgraph(fem_->update_velocity(
                                                sim_config_->ctx().view().data()),
                                            {})
                                    .tail();
                            cp.subgraph(fem_->copy_x_prev(), {n_fem_vel});
                        }

                        if(bspline_)
                        {
                            auto n_bspline_velocity =
                                cp.subgraph(bspline_->update_velocity(
                                                sim_config_->ctx().view().data()),
                                            {})
                                    .tail();
                            cp.subgraph(bspline_->copy_q_prev(), {n_bspline_velocity});
                        }

                        if(mpm_)
                        {
                            // Node velocity v = u/dt -> G2P + advection ->
                            // Hencky/DP return mapping (lagged plasticity).
                            cp.subgraph(mpm_->finalize_frame(), {});
                        }

                        if(joint_system_ && !joint_system_->empty())
                        {
                            std::vector<cudaGraphNode_t> post_deps;
                            auto theta_sg = joint_system_->theta();
                            if(theta_sg)
                                post_deps.push_back(cp.subgraph(theta_sg, {}).tail());
                            // Joint post-step hooks (e.g. the revolute
                            // elastoplastic spring's rest-angle absorption)
                            // read the committed theta, so they follow the
                            // theta kernels.
                            auto post_sg = joint_system_->post_step();
                            if(post_sg)
                                cp.subgraph(post_sg, post_deps);
                        }

                        // Generic post-step hook (CP4): let the contact constitution
                        // run a post-convergence update over the settled contact set.
                        // Falsy (empty subgraph) for model-agnostic constitutions =>
                        // no-op, so behaviour is unchanged unless a model opts in.
                        if(contact_)
                        {
                            auto sg = contact_->constitution().post_step_update();
                            if(sg)
                                cp.subgraph(sg, {});
                        }

                        // Distance-bond release runs once over the converged
                        // state. Creation is a separate yielding checkpoint
                        // below so an exact-resize resume cannot repeat velocity
                        // updates, previous-state copies, or released-feed writes.
                        if(bond_system_ && contact_)
                        {
                            if(auto sg = bond_system_->bond_break(
                                   sim_config_->ctx().view().data()))
                                cp.subgraph(sg, {});
                        }

                        if(cut_detection_ && cut_detection_->active() && contact_)
                        {
                            auto sg = cut_detection_->build_detection_subgraph(
                                vertex_->ctx().view().data(),
                                surface_->ctx().view().data(),
                                contact_->ctx().view().data());
                            if(sg)
                                cp.subgraph(sg, {});
                        }
                    });

                // ==========================================================
                // Post-Newton bond create: append once; on exact unmet demand
                // resize (demand + grow-factor slack) and resume THIS
                // checkpoint exactly once. PR #356's loop scope gate skips the
                // Newton WHILE on that resume.
                // ==========================================================
                if(bond_system_ && contact_)
                {
                    m.checkpoint(
                        [&](CheckpointScope& cp)
                        {
                            if(auto sg = bond_system_->bond_create(
                                   vertex_->ctx().view().data(),
                                   surface_->ctx().view().data(),
                                   contact_->ctx().view().data()))
                                cp.subgraph(sg, {});
                        },
                        bond_system_->overflow_flag_var(),
                        [this](YieldContext& ctx) { handle_bond_overflow(ctx); },
                        YieldResume::This);
                }

            }  // has_lagrangian_dofs
        });

    return pipe;
}

// ============================================================================
// Canonical elastic-only partition seed.
//
// The init pipeline and the post-topology recovery pipeline share these two
// recipes so static_labels always describe the same production sparsity:
// live extent reports, solve-scope derivation, elastic assemble, BCOO
// sort/reduce, then FastSV. Contact/bond/cluster relocation are deliberately
// absent; matrix-free structural connections enter through partitioner hooks.
// ============================================================================

Subgraph SimEngine::build_elastic_extent_report()
{
    return Subgraph(
        [this](SubgraphScope& sg, const std::vector<cudaGraphNode_t>& deps) -> std::vector<cudaGraphNode_t>
        {
            auto d_lctx = linear_system_->ctx().view().data();
            std::vector<cudaGraphNode_t> report_tails;
            auto add_report = [&](Subgraph report, const std::vector<cudaGraphNode_t>& report_deps)
            {
                if(report)
                    report_tails.push_back(
                        sg.subgraph(std::move(report), report_deps).tail());
            };

            if(abd_)
            {
                add_report(abd_->kinetic()->report_extent_node(d_lctx), deps);
                for(auto* constitution : abd_->constitutions())
                    add_report(constitution->report_extent_node(d_lctx), deps);
            }
            if(rigid_)
                add_report(rigid_->report_extent_node(d_lctx), deps);
            if(joint_system_ && !joint_system_->empty())
            {
                for(auto* constitution : joint_system_->constitutions())
                    add_report(constitution->report_extent_node(d_lctx), deps);
            }

            cudaGraphNode_t fem_report{};
            if(fem_)
            {
                auto report = fem_->report_extent_node(d_lctx);
                if(report)
                {
                    fem_report = sg.subgraph(std::move(report), deps).tail();
                    report_tails.push_back(fem_report);
                }
                for(auto* constitution : fem_->constitutions())
                    add_report(constitution->report_extent_node(d_lctx), deps);
            }
            cudaGraphNode_t bspline_report{};
            if(bspline_)
            {
                std::vector<cudaGraphNode_t> bspline_deps = deps;
                if(fem_report)
                    bspline_deps.push_back(fem_report);
                bspline_report = sg.subgraph(bspline_->report_extent_node(d_lctx), bspline_deps)
                                     .tail();
                report_tails.push_back(bspline_report);
            }
            if(mpm_)
            {
                std::vector<cudaGraphNode_t> mpm_deps = deps;
                if(bspline_report)
                    mpm_deps.push_back(bspline_report);
                else if(fem_report)
                    mpm_deps.push_back(fem_report);
                add_report(mpm_->report_extent_node(d_lctx), mpm_deps);
            }

            return {sg.subgraph(linear_system_->derive_extents(), report_tails).tail()};
        });
}

Subgraph SimEngine::build_static_labels_assemble()
{
    return Subgraph(
        [this](SubgraphScope& sg, const std::vector<cudaGraphNode_t>& deps) -> std::vector<cudaGraphNode_t>
        {
            QIPC_ASSERT(partitioner_, "static-label assemble requires ComponentPartitioner");

            GridStrideKernel<memset_b_rhs_kernel> k_memset_rhs;
            auto                                  n_memset_rhs = sg.kernel(
                k_memset_rhs, std::make_tuple(linear_system_->ctx().view().data()), deps);

            GridStrideKernel<memset_triplet_kernel> k_memset_triplet;
            auto                                    n_memset_triplet =
                sg.kernel(k_memset_triplet,
                          std::make_tuple(linear_system_->ctx().view().data()),
                          deps);

            std::vector<Subgraph> scope_subgraphs;
            if(abd_)
            {
                auto scope =
                    abd_->derive_solve_scope(sim_config_->ctx().view().data(),
                                             linear_system_->ctx().view().data());
                if(scope)
                    scope_subgraphs.push_back(std::move(scope));
            }
            if(rigid_)
            {
                auto scope =
                    rigid_->derive_solve_scope(sim_config_->ctx().view().data());
                if(scope)
                    scope_subgraphs.push_back(std::move(scope));
            }
            if(fem_)
            {
                auto scope =
                    fem_->derive_solve_scope(sim_config_->ctx().view().data(),
                                             linear_system_->ctx().view().data());
                if(scope)
                    scope_subgraphs.push_back(std::move(scope));
            }

            std::vector<cudaGraphNode_t> scope_deps = {n_memset_rhs, n_memset_triplet};
            if(!scope_subgraphs.empty())
            {
                scope_deps = {sg.subgraph(Subgraph::parallel(std::move(scope_subgraphs)), scope_deps)
                                  .tail()};
            }

            std::vector<cudaGraphNode_t> assemble_tails;
            if(abd_)
                assemble_tails.push_back(
                    sg.subgraph(abd_->assemble(sim_config_->ctx().view().data(),
                                               linear_system_->ctx().view().data()),
                                scope_deps)
                        .tail());
            if(rigid_)
                assemble_tails.push_back(
                    sg.subgraph(rigid_->assemble(sim_config_->ctx().view().data(),
                                                 linear_system_->ctx().view().data()),
                                scope_deps)
                        .tail());
            if(fem_)
                assemble_tails.push_back(
                    sg.subgraph(fem_->assemble(sim_config_->ctx().view().data(),
                                               linear_system_->ctx().view().data()),
                                scope_deps)
                        .tail());
            if(bspline_)
                assemble_tails.push_back(
                    sg.subgraph(bspline_->assemble(sim_config_->ctx().view().data(),
                                                   linear_system_->ctx().view().data()),
                                scope_deps)
                        .tail());
            if(has_joints())
                assemble_tails.push_back(
                    sg.subgraph(joint_system_->assemble(sim_config_->ctx().view(),
                                                        linear_system_->ctx().view()),
                                scope_deps)
                        .tail());
            if(mpm_)
                assemble_tails.push_back(
                    sg.subgraph(mpm_->assemble(linear_system_->ctx().view().data()), scope_deps)
                        .tail());

            auto n_bcoo_setup = sg.kernel(
                KernelConfig{"static_labels_bcoo_setup"}.grid(1).block(1),
                [lctx = linear_system_->ctx().view()] __device__() mutable
                {
                    *lctx->bcoo_nnz_device_view() = 0;
                    *lctx->n_triplets_device_view() =
                        static_cast<int32_t>(*lctx->n_elastic_device_view());
                },
                assemble_tails);

            auto n_sort =
                sg.subgraph(linear_system_->body_sort_reduce(), {n_bcoo_setup}).tail();
            return {sg.subgraph(partitioner_->compute_static_labels(), {n_sort}).tail()};
        });
}

Pipeline SimEngine::build_init_pipeline()
{
    PipelineGraphBuilder builder;

    Pipeline pipe = builder.main(
        [&](MainScope& m)
        {
            m.checkpoint(
                [&](CheckpointScope& cp)
                {
                    std::vector<cudaGraphNode_t> compute_pos_tails;

                    if(abd_)
                    {
                        auto n_abd_pos = cp.subgraph(abd_->forward_global_vertices(
                                                         vertex_->ctx().view().data()),
                                                     {})
                                             .tail();
                        compute_pos_tails.push_back(n_abd_pos);
                    }

                    if(rigid_)
                    {
                        auto n_rigid_pos =
                            cp.subgraph(rigid_->forward_global_vertices(
                                            vertex_->ctx().view().data()),
                                        {})
                                .tail();
                        compute_pos_tails.push_back(n_rigid_pos);
                    }

                    if(fem_)
                    {
                        auto n_fem_pos = cp.subgraph(fem_->forward_global_vertices(
                                                         vertex_->ctx().view().data()),
                                                     {})
                                             .tail();
                        compute_pos_tails.push_back(n_fem_pos);
                    }

                    if(bspline_)
                    {
                        auto n_bspline_pos =
                            cp.subgraph(bspline_->forward_global_vertices(
                                            vertex_->ctx().view().data()),
                                        {})
                                .tail();
                        compute_pos_tails.push_back(n_bspline_pos);
                    }

                    if(mpm_ && mpm_->n_particles() > 0)
                    {
                        // u = 0 at init, so this writes the wired particle
                        // positions into the contact-vertex range.
                        auto n_mpm_pos = cp.subgraph(mpm_->forward_global_vertices(
                                                         vertex_->ctx().view().data()),
                                                     {})
                                             .tail();
                        compute_pos_tails.push_back(n_mpm_pos);
                    }

                    if(cluster_manager_ && rigid_ && fem_)
                    {
                        auto n_rider_pos =
                            cp.subgraph(cluster_manager_->forward_rigid_rider_vertices(
                                            vertex_->ctx().view().data()),
                                        compute_pos_tails)
                                .tail();
                        compute_pos_tails = {n_rider_pos};
                    }

                    auto n_trajectory_reset =
                        cp.subgraph(vertex_->reset_trajectory(), compute_pos_tails)
                            .tail();

                    if(contact_)
                    {
                        // Endpoint equals the current position, so the init
                        // BVH is static.

                        auto bvh_face = contact_->bvh_triangle_build(
                            vertex_->ctx().view().data(), surface_->ctx().view().data());
                        auto bvh_edge =
                            contact_->bvh_edge_build(vertex_->ctx().view().data(),
                                                     surface_->ctx().view().data());

                        auto n_reset = cp.kernel(
                            KernelConfig{"reset_collision_count"}.grid(1).block(1),
                            [cctx = contact_->ctx().view()] __device__() mutable
                            {
                                *cctx->buffer_frame_counts_view() =
                                    ContactBufferFrameCounts{};
                                *cctx->n_pairs_pt_view() = 0;
                                *cctx->n_pairs_ee_view() = 0;
                                if(cctx->max_pairs_pe > 0)
                                    *cctx->n_pairs_pe_view() = 0;
                                if(cctx->max_pairs_pp > 0)
                                    *cctx->n_pairs_pp_view() = 0;
                            },
                            {n_trajectory_reset});
                        if(bvh_face)
                            cp.subgraph(bvh_face, {n_reset});
                        if(bvh_edge)
                            cp.subgraph(bvh_edge, {n_reset});

                        if(contact_->has_codim_point_bvh())
                        {
                            auto bvh_codim = contact_->bvh_codim_point_build(
                                vertex_->ctx().view().data(),
                                surface_->ctx().view().data());
                            cp.subgraph(bvh_codim, {n_reset});
                        }
                    }
                });


            m.checkpoint(
                [&](CheckpointScope& cp)
                {
                    std::vector<cudaGraphNode_t> tail_deps;

                    if(partitioner_)
                    {
                        QIPC_ASSERT(masked_pcg_solver_,
                                    "partition checkpoint requires MaskedPCGSolver");
                        auto n_report =
                            cp.subgraph(build_elastic_extent_report(), {}).tail();
                        tail_deps.push_back(
                            cp.subgraph(build_static_labels_assemble(), {n_report})
                                .tail());
                    }

                    if(ccd_partitioner_)
                    {
                        tail_deps.push_back(
                            cp.subgraph(ccd_partitioner_->compute_static_labels(
                                            vertex_->ctx().view().data(),
                                            body_mgr_->ctx().view().data()),
                                        {})
                                .tail());
                    }
                });

            if(contact_)
            {
                m.checkpoint(
                    [&](CheckpointScope& cp)
                    {
                        cp.subgraph(contact_->trajectory_query(
                                        vertex_->ctx().view().data(),
                                        surface_->ctx().view().data(),
                                        body_mgr_->ctx().view().data(),
                                        contact_->overflow_flag_var()),
                                    {});
                    },
                    contact_->overflow_flag_var(),
                    [this](YieldContext& ctx) { handle_init_overflow(ctx); },
                    YieldResume::This);
            }
        });

    return pipe;
}

// ============================================================================
// Contact rebuild pipeline (the pre-step stage's graph).
//
// TWO between-steps processes invalidate the collision pair set: topology
// commits (geometry changed) and affine-cluster membership events (body-id
// flips change the broad phase's same-body culling and the derived
// contact-side pin). Both request a rebuild through the host flag
// (request_contact_rebuild); step() replays this graph BEFORE the friction
// capacity check and the step launch. Built once at init() whenever such
// an invalidator exists (all kernels read live counts through device
// contexts; only the morton-sort onesweep grids are patched via
// reconfigure on topo growth).
//
// It restores the pair state to exactly what the init pipeline establishes
// at t=0: recompute vertex positions from the current FEM/ABD state
// (required after a commit; idempotent for the position-preserving cluster
// events), reset the trajectory endpoint/safe positions to the recomputed
// current positions, rebuild the BVHs, recompute the CCD partitioner's static labels
// (they read body ids), and re-detect all collision pairs (with the same
// overflow-yield handling as init). The step's frame-start friction
// snapshot then derives the friction pairs from these fresh contact pairs.
//
// Dependencies are explicit for maximal parallelism: ABD || FEM position
// recompute, then kinematic reset, with the pair-counter reset alongside;
// the three BVH builds run in parallel; the query fork (PT || EE || PE ||
// PP || halfplane) parallelizes internally.
// ============================================================================

Pipeline SimEngine::build_contact_rebuild_pipeline()
{
    QIPC_ASSERT(contact_ && surface_ && vertex_ && body_mgr_,
                "contact rebuild pipeline requires the contact stack");

    PipelineGraphBuilder builder;

    Pipeline pipe = builder.main(
        [&](MainScope& m)
        {
            m.checkpoint(
                [&](CheckpointScope& cp)
                {
                    std::vector<cudaGraphNode_t> pos_tails;

                    if(abd_)
                        pos_tails.push_back(
                            cp.subgraph(abd_->forward_global_vertices(
                                            vertex_->ctx().view().data()),
                                        {})
                                .tail());
                    if(rigid_)
                        pos_tails.push_back(
                            cp.subgraph(rigid_->forward_global_vertices(
                                            vertex_->ctx().view().data()),
                                        {})
                                .tail());
                    if(fem_)
                        pos_tails.push_back(
                            cp.subgraph(fem_->forward_global_vertices(
                                            vertex_->ctx().view().data()),
                                        {})
                                .tail());
                    // No MPM node: topology events are Lagrangian-only (MPM
                    // takes no part in cut/fracture commits) and between
                    // frames the u state does not match the advected x_p; the
                    // next frame's CP0 refresh rewrites the range anyway.

                    if(cluster_manager_ && rigid_ && fem_)
                        pos_tails = {cp.subgraph(cluster_manager_->forward_rigid_rider_vertices(
                                                     vertex_->ctx().view().data()),
                                                 pos_tails)
                                         .tail()};

                    auto n_kin =
                        cp.subgraph(vertex_->reset_trajectory(), pos_tails).tail();

                    auto n_reset = cp.kernel(
                        KernelConfig{"topo_reset_collision_count"}.grid(1).block(1),
                        [cctx = contact_->ctx().view()] __device__() mutable
                        {
                            *cctx->n_pairs_pt_view() = 0;
                            *cctx->n_pairs_ee_view() = 0;
                            if(cctx->max_pairs_pe > 0)
                                *cctx->n_pairs_pe_view() = 0;
                            if(cctx->max_pairs_pp > 0)
                                *cctx->n_pairs_pp_view() = 0;
                        },
                        {});

                    auto bvh_face =
                        contact_->bvh_triangle_build(vertex_->ctx().view().data(),
                                                     surface_->ctx().view().data());
                    auto bvh_edge =
                        contact_->bvh_edge_build(vertex_->ctx().view().data(),
                                                 surface_->ctx().view().data());
                    if(bvh_face)
                        cp.subgraph(bvh_face, {n_kin, n_reset});
                    if(bvh_edge)
                        cp.subgraph(bvh_edge, {n_kin, n_reset});
                    if(contact_->has_codim_point_bvh())
                    {
                        auto bvh_codim = contact_->bvh_codim_point_build(
                            vertex_->ctx().view().data(), surface_->ctx().view().data());
                        cp.subgraph(bvh_codim, {n_kin, n_reset});
                    }
                });

            if(ccd_partitioner_)
            {
                m.checkpoint(
                    [&](CheckpointScope& cp)
                    {
                        cp.subgraph(ccd_partitioner_->compute_static_labels(
                                        vertex_->ctx().view().data(),
                                        body_mgr_->ctx().view().data()),
                                    {});
                    });
            }

            m.checkpoint(
                [&](CheckpointScope& cp)
                {
                    cp.subgraph(
                        contact_->trajectory_query(vertex_->ctx().view().data(),
                                                   surface_->ctx().view().data(),
                                                   body_mgr_->ctx().view().data(),
                                                   contact_->overflow_flag_var()),
                        {});
                },
                contact_->overflow_flag_var(),
                [this](YieldContext& ctx) { handle_init_overflow(ctx); },
                YieldResume::This);
        });

    return pipe;
}

// ============================================================================
// Exact elastic static-label recovery after MutableTopo commits.
//
// Built once at init and replayed only when a commit marks it pending. CP0
// derives live extents and yields for row/triplet capacity changes; CP1
// reuses the canonical elastic-only assembly recipe and recomputes FastSV
// labels. Back-to-back commits coalesce before this pre-step replay.
// ============================================================================

Pipeline SimEngine::build_static_labels_rebuild_pipeline()
{
    QIPC_ASSERT(partitioner_ && masked_pcg_solver_, "static-label rebuild requires partition_pcg");

    PipelineGraphBuilder builder;
    return builder.main(
        [&](MainScope& m)
        {
            m.checkpoint(
                [&](CheckpointScope& cp)
                {
                    auto n_report =
                        cp.subgraph(build_elastic_extent_report(), {}).tail();
                    cp.kernel(
                        KernelConfig{"static_labels_capacity_check"}.grid(1).block(1),
                        [lctx = linear_system_->ctx().view(),
                         overflow = triplet_overflow_flag_.view()] __device__() mutable
                        {
                            int64_t n_el = *lctx->n_elastic_device_view();
                            *lctx->n_triplets_device_view() = static_cast<int32_t>(n_el);
                            bool triplet_overflow = n_el > lctx->padded_triplets;
                            bool row_change = *lctx->required_block_rows_view()
                                              != lctx->n_block_rows;
                            *overflow = (triplet_overflow || row_change) ? 1 : 0;
                        },
                        {n_report});
                },
                triplet_overflow_flag_.view(),
                [this](YieldContext& ctx) { handle_static_labels_overflow(ctx); },
                YieldResume::Next);

            m.checkpoint([&](CheckpointScope& cp)
                         { cp.subgraph(build_static_labels_assemble(), {}); });
        });
}

// ============================================================================
// Post-topology surface derive pipeline.
//
// Built at init() alongside the recovery pipeline and reused for every
// event. A single checkpoint wrapping the GlobalSurfaceManager derive
// subgraph: it re-derives the FEM segments of the surface tables, area
// weights, and the codim selection on device from the post-commit FEM
// tables, and publishes the derived counts for the host readback. Its
// three exclusive-sum scans repatch their grids in this exec inside the
// reconfigure(n) calls made by ensure_topo_capacity (push model).
// ============================================================================

Pipeline SimEngine::build_topo_derive_pipeline()
{
    QIPC_ASSERT(surface_ && fem_ && vertex_,
                "topo derive pipeline requires the FEM surface stack");

    PipelineGraphBuilder builder;
    return builder.main([&](MainScope& m)
                        { m.subgraph(surface_->derive_topo_surface(), {}); });
}

Pipeline SimEngine::build_ls_replay_pipeline()
{
    // Standalone LS-only pipeline (docs/ls_forensics.md): exactly the trial
    // step_forward + energy subgraphs of the production LS loop body -- no
    // assemble, no PCG, no CCD, no BVH, no accept logic. The host writes
    // *alpha and launches; the tail kernel materializes every energy term
    // into ls_replay_energies_buf_ with the total accumulated in the exact
    // check_line_search order.
    QIPC_ASSERT(vertex_ && sim_config_, "ls replay requires an initialized engine");

    PipelineGraphBuilder builder;
    return builder.main(
        [&](MainScope& m)
        {
            // Stability: the partitioner context is its own Var, allocated
            // exactly once via create_context<T, Mode>() in do_build() and never
            // re-assigned, so baking .data() below (as the production LS
            // body does) cannot dangle.
            VarView<const CCDComponentPartitionerContext> ccd_ctx =
                ccd_partitioner_ ? ccd_partitioner_->ctx().view() :
                                   VarView<const CCDComponentPartitionerContext>{};

            std::vector<cudaGraphNode_t> sum_deps;
            std::vector<cudaGraphNode_t> trial_pos_deps;
            std::vector<cudaGraphNode_t> joint_state_deps;
            cudaGraphNode_t              n_rigid_step{};

            if(abd_)
            {
                auto n_step =
                    m.subgraph(abd_->step_forward(sim_config_->ctx().view().data(),
                                                  ccd_ctx.data()),
                               {})
                        .tail();
                joint_state_deps.push_back(n_step);
                trial_pos_deps.push_back(m.subgraph(abd_->forward_global_vertices(
                                                        vertex_->ctx().view().data()),
                                                    {n_step})
                                             .tail());
                auto n_ef =
                    m.subgraph(abd_->energy(sim_config_->ctx().view().data()), {n_step})
                        .tail();
                sum_deps.push_back(n_ef);
            }

            if(rigid_)
            {
                n_rigid_step =
                    m.subgraph(rigid_->step_forward(sim_config_->ctx().view().data(),
                                                    ccd_ctx.data()),
                               {})
                        .tail();
                joint_state_deps.push_back(n_rigid_step);
                auto n_pos = m.subgraph(rigid_->forward_global_vertices(
                                            vertex_->ctx().view().data()),
                                        {n_rigid_step})
                                 .tail();
                auto n_ef =
                    m.subgraph(rigid_->energy(sim_config_->ctx().view().data()), {n_rigid_step})
                        .tail();
                sum_deps.push_back(n_ef);
                trial_pos_deps.push_back(n_pos);
            }

            if(fem_)
            {
                auto n_step =
                    m.subgraph(fem_->step_forward(sim_config_->ctx().view().data(),
                                                  vertex_->ctx().view().data(),
                                                  ccd_ctx.data()),
                               {})
                        .tail();
                auto n_fem_positions = m.subgraph(fem_->forward_global_vertices(
                                                      vertex_->ctx().view().data()),
                                                  {n_step})
                                           .tail();
                cudaGraphNode_t n_fem_energy_dep = n_step;
                if(cluster_manager_ && rigid_)
                {
                    n_fem_positions = m.subgraph(cluster_manager_->forward_rigid_rider_vertices(
                                                     vertex_->ctx().view().data()),
                                                 {n_rigid_step, n_fem_positions})
                                          .tail();
                    n_fem_energy_dep = n_fem_positions;
                }
                trial_pos_deps.push_back(n_fem_positions);
                auto n_ef = m.subgraph(fem_->energy(sim_config_->ctx().view().data()),
                                       {n_fem_energy_dep})
                                .tail();
                sum_deps.push_back(n_ef);
            }

            if(bspline_)
            {
                auto n_step =
                    m.subgraph(bspline_->step_forward(sim_config_->ctx().view().data(),
                                                      ccd_ctx.data()),
                               {})
                        .tail();
                auto n_positions = m.subgraph(bspline_->forward_global_vertices(
                                                  vertex_->ctx().view().data()),
                                              {n_step})
                                       .tail();
                trial_pos_deps.push_back(n_positions);
                auto n_energy =
                    m.subgraph(bspline_->energy(sim_config_->ctx().view().data()), {n_step})
                        .tail();
                sum_deps.push_back(n_energy);
            }

            if(has_joints())
            {
                auto n_joint_ef =
                    m.subgraph(joint_system_->energy(sim_config_->ctx().view()), joint_state_deps)
                        .tail();
                auto n_joint_final =
                    m.subgraph(joint_system_->energy_final(), {n_joint_ef}).tail();
                sum_deps.push_back(n_joint_final);
            }

            if(contact_)
            {
                auto n_ce = m.subgraph(contact_->contact_energy(
                                           vertex_->ctx().view().data(),
                                           surface_->ctx().view().data(),
                                           sim_config_->ctx().view().data()),
                                       trial_pos_deps)
                                .tail();
                sum_deps.push_back(n_ce);
            }

            m.kernel(
                KernelConfig{"ls_replay_sum"}.grid(1).block(1),
                [abd = abd_ ? abd_->ctx().view() : VarView<const ABDContext>{},
                 rigid = rigid_ ? rigid_->ctx().view() : VarView<const RigidContext>{},
                 fem = fem_ ? fem_->ctx().view() : VarView<const FEMContext>{},
                 bspline = bspline_ ? bspline_->ctx().view() : VarView<const BSplineContext>{},
                 jctx     = (joint_system_ && !joint_system_->empty()) ?
                                joint_system_->ctx().view() :
                                VarView<const GlobalJointContext>{},
                 cctx     = contact_ ? contact_->ctx().view() :
                                       VarView<const GlobalContactContext>{},
                 has_fric = contact_ && contact_->ctx().ro().has_friction(),
                 has_hp   = contact_ && contact_->ctx().ro().n_halfplanes > 0,
                 out      = ls_replay_energies_buf_.view()] __device__() mutable
                {
                    double e_abd = abd.data() ? *abd->abd_energy_view() : 0.0;
                    double e_rigid = rigid.data() ? *rigid->rigid_energy_view() : 0.0;
                    double e_fem = fem.data() ? *fem->fem_energy_view() : 0.0;
                    double e_bs = bspline.data() ? bspline->energy_view()[0] : 0.0;
                    double e_joint = jctx.data() ? *jctx->joint_energy_view() : 0.0;
                    double e_con = cctx.data() ? *cctx->contact_total_energy_view() : 0.0;

                    // EXACT check_line_search accumulation order (bitwise
                    // parity with the production trial sums; MPM absent by
                    // the replay precondition).
                    double e = 0.0;
                    if(abd.data())
                        e += *abd->abd_energy_view();
                    if(rigid.data())
                        e += *rigid->rigid_energy_view();
                    if(fem.data())
                        e += *fem->fem_energy_view();
                    if(bspline.data())
                        e += bspline->energy_view()[0];
                    if(jctx.data())
                        e += *jctx->joint_energy_view();
                    if(cctx.data())
                        e += *cctx->contact_total_energy_view();

                    out[0]  = e;
                    out[1]  = e_abd;
                    out[2]  = e_rigid;
                    out[3]  = e_fem;
                    out[4]  = e_bs;
                    out[5]  = e_joint;
                    out[6]  = e_con;
                    out[7]  = cctx.data() ? *cctx->barrier_energy_view() : 0.0;
                    out[8]  = (cctx.data() && has_fric) ?
                                  *cctx->frictional_contact_energy_view() :
                                  0.0;
                    out[9]  = (cctx.data() && has_hp) ?
                                  *cctx->halfplane_barrier_energy_view() :
                                  0.0;
                    out[10] = (cctx.data() && has_hp && has_fric) ?
                                  *cctx->halfplane_friction_energy_view() :
                                  0.0;
                },
                sum_deps);
        });
}

Pipeline SimEngine::build_ls_replay_grad_pipeline()
{
    // Per-term gradient dots for the offline Taylor twin test
    // (docs/ls_forensics.md M4): at the trial state x(alpha), each
    // assemble-level term's gradient is re-assembled ALONE into the zeroed
    // global rhs and dotted with the packed closure direction
    // (ls_replay_dir_). Terms run SEQUENTIALLY because they share b_rhs.
    // The fused assemble kernels also write Hessian triplets -- extents are
    // derived up front so the writes land in bounds, but nothing reads them
    // (no body_sort_reduce, no PCG). Production solve-scope derivation is
    // skipped: the closure staged the post-derivation is_fixed flags.
    QIPC_ASSERT(vertex_ && sim_config_ && linear_system_,
                "ls replay grad requires an initialized engine");

    PipelineGraphBuilder builder;
    return builder.main(
        [&](MainScope& m)
        {
            VarView<const CCDComponentPartitionerContext> ccd_ctx =
                ccd_partitioner_ ? ccd_partitioner_->ctx().view() :
                                   VarView<const CCDComponentPartitionerContext>{};
            VarView<const GlobalLinearSystemContext> lctx =
                linear_system_->ctx().view();
            VarView<const SimContext> sctx = sim_config_->ctx().view();

            // ---- Trial state at alpha (the energy replay's head) ----
            std::vector<cudaGraphNode_t> trial_pos_deps;
            std::vector<cudaGraphNode_t> state_tails;
            cudaGraphNode_t              n_rigid_step{};
            if(abd_)
            {
                auto n_step =
                    m.subgraph(abd_->step_forward(sctx.data(), ccd_ctx.data()), {})
                        .tail();
                state_tails.push_back(n_step);
                trial_pos_deps.push_back(m.subgraph(abd_->forward_global_vertices(
                                                        vertex_->ctx().view().data()),
                                                    {n_step})
                                             .tail());
            }
            if(rigid_)
            {
                n_rigid_step =
                    m.subgraph(rigid_->step_forward(sctx.data(), ccd_ctx.data()), {})
                        .tail();
                auto n_pos = m.subgraph(rigid_->forward_global_vertices(
                                            vertex_->ctx().view().data()),
                                        {n_rigid_step})
                                 .tail();
                state_tails.push_back(n_rigid_step);
                trial_pos_deps.push_back(n_pos);
            }
            if(fem_)
            {
                auto n_fem_step =
                    m.subgraph(fem_->step_forward(sctx.data(),
                                                  vertex_->ctx().view().data(),
                                                  ccd_ctx.data()),
                               {})
                        .tail();
                state_tails.push_back(n_fem_step);
                auto n_fem_pos = m.subgraph(fem_->forward_global_vertices(
                                                vertex_->ctx().view().data()),
                                            {n_fem_step})
                                     .tail();
                if(cluster_manager_ && rigid_)
                {
                    n_fem_pos = m.subgraph(cluster_manager_->forward_rigid_rider_vertices(
                                               vertex_->ctx().view().data()),
                                           {n_rigid_step, n_fem_pos})
                                    .tail();
                    state_tails.push_back(n_fem_pos);
                }
                trial_pos_deps.push_back(n_fem_pos);
            }
            if(bspline_)
            {
                auto n_bspline_step =
                    m.subgraph(bspline_->step_forward(sctx.data(), ccd_ctx.data()), {})
                        .tail();
                state_tails.push_back(n_bspline_step);
                trial_pos_deps.push_back(m.subgraph(bspline_->forward_global_vertices(
                                                        vertex_->ctx().view().data()),
                                                    {n_bspline_step})
                                             .tail());
            }

            // ---- Extent offsets (triplet write bases; mirrors CP1a) ----
            std::vector<cudaGraphNode_t> derive_deps;
            auto                         add_report = [&](Subgraph sg)
            {
                if(sg)
                    derive_deps.push_back(m.subgraph(std::move(sg), {}).tail());
            };
            if(abd_)
            {
                add_report(abd_->kinetic()->report_extent_node(lctx.data()));
                for(auto* c : abd_->constitutions())
                    add_report(c->report_extent_node(lctx.data()));
            }
            if(rigid_)
                add_report(rigid_->report_extent_node(lctx.data()));
            if(joint_system_ && !joint_system_->empty())
            {
                for(auto* c : joint_system_->constitutions())
                    add_report(c->report_extent_node(lctx.data()));
            }
            cudaGraphNode_t n_fem_report{};
            if(fem_)
            {
                n_fem_report =
                    m.subgraph(fem_->report_extent_node(lctx.data()), {}).tail();
                derive_deps.push_back(n_fem_report);
                for(auto* c : fem_->constitutions())
                    add_report(c->report_extent_node(lctx.data()));
            }
            if(bspline_)
            {
                std::vector<cudaGraphNode_t> bspline_deps;
                if(n_fem_report)
                    bspline_deps.push_back(n_fem_report);
                derive_deps.push_back(
                    m.subgraph(bspline_->report_extent_node(lctx.data()), bspline_deps)
                        .tail());
            }
            auto n_derive =
                m.subgraph(linear_system_->derive_extents(), derive_deps).tail();

            GridStrideKernel<memset_triplet_kernel> k_memset_trip;
            auto                                    n_memset_trip =
                m.kernel(k_memset_trip, std::make_tuple(lctx.data()), {n_derive});

            auto n_zero_gdot =
                m.kernel(KernelConfig{"ls_gdot_zero"}.grid(1).block(1),
                         [out = ls_replay_gdot_buf_.view()] __device__() mutable
                         {
                             for(int32_t i = 0; i < kLsReplayGradTerms; ++i)
                                 out[i] = 0.0;
                         },
                         {});

            // ---- Effective DOF-space direction: EXACTLY d(state)/d(alpha)
            // ---- of the replayed path, mirroring the step_forward
            // ---- semantics (fixed rows zero, per-body CCD-component
            // ---- scaling). Dotting the raw dq/dx instead would disagree
            // ---- with the energy slope on every CCD-clamped scene.
            std::vector<cudaGraphNode_t> pack_tails;
            if(abd_)
                pack_tails.push_back(
                    m.kernel(KernelConfig{"ls_pack_dir_abd"}.grid(64).block(256),
                             [abd = abd_->ctx().view(),
                              ccd = ccd_ctx,
                              dir = ls_replay_dir_.view()] __device__() mutable
                             {
                                 int64_t nb       = abd->n_bodies;
                                 auto    dq       = abd->dq_view();
                                 auto    is_fixed = abd->is_fixed_view();
                                 int64_t base     = abd->dof_offset;
                                 for(int64_t b = blockIdx.x * (int64_t)blockDim.x
                                                 + threadIdx.x;
                                     b < nb;
                                     b += (int64_t)gridDim.x * blockDim.x)
                                 {
                                     double s = is_fixed[(int32_t)b] ? 0.0 : 1.0;
                                     if(ccd.data())
                                         s *= ccd->body_ccd_alpha_view()[(int32_t)b];
                                     for(int32_t j = 0; j < 12; ++j)
                                         dir[base + 12 * b + j] = s * dq[12 * b + j];
                                 }
                             },
                             {}));
            if(rigid_)
                pack_tails.push_back(m.kernel(
                    KernelConfig{"ls_pack_dir_rigid"}.grid(64).block(256),
                    [rigid = rigid_->ctx().view(),
                     ccd   = ccd_ctx,
                     dir   = ls_replay_dir_.view()] __device__() mutable
                    {
                        int64_t nb       = rigid->n_bodies;
                        auto    dq       = rigid->dq_view();
                        auto    is_fixed = rigid->is_fixed_view();
                        int64_t base     = rigid->dof_offset;
                        int32_t gbo      = rigid->global_body_offset;
                        for(int64_t b =
                                blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
                            b < nb;
                            b += (int64_t)gridDim.x * blockDim.x)
                        {
                            double s = is_fixed[(int32_t)b] ? 0.0 : 1.0;
                            if(ccd.data())
                                s *= ccd->body_ccd_alpha_view()[gbo + (int32_t)b];
                            for(int32_t j = 0; j < 6; ++j)
                                dir[base + 6 * b + j] = s * dq[6 * b + j];
                        }
                    },
                    {}));
            if(fem_)
                pack_tails.push_back(m.kernel(
                    KernelConfig{"ls_pack_dir_fem"}.grid(64).block(256),
                    [fem = fem_->ctx().view(),
                     vtx = vertex_->ctx().view(),
                     ccd = ccd_ctx,
                     dir = ls_replay_dir_.view()] __device__() mutable
                    {
                        int64_t nv    = fem->n_fem_verts;
                        auto    dx    = fem->dx_view();
                        auto    fixed = fem->is_fixed_view();
                        int64_t gvo   = fem->global_vert_offset;
                        int64_t base  = fem->dof_offset;
                        for(int64_t v =
                                blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
                            v < nv;
                            v += (int64_t)gridDim.x * blockDim.x)
                        {
                            double s = fixed[v] ? 0.0 : 1.0;
                            if(ccd.data())
                                s *= ccd->body_ccd_alpha_view()[vtx->body_id_view()[(int32_t)(gvo + v)]];
                            dir.as<3>(base / 3 + v) = s * dx.as<3>(v);
                        }
                    },
                    {}));
            if(bspline_)
                pack_tails.push_back(m.kernel(
                    KernelConfig{"ls_pack_dir_bspline"}.grid(64).block(256),
                    [bspline = bspline_->ctx().view(),
                     ccd     = ccd_ctx,
                     dir     = ls_replay_dir_.view()] __device__() mutable
                    {
                        int64_t nc       = bspline->n_controls;
                        auto    dq       = bspline->dq_view();
                        auto    fixed    = bspline->is_fixed_view();
                        auto    body_ids = bspline->control_body_ids_view();
                        int64_t base     = bspline->dof_offset;
                        int32_t gbo      = bspline->global_body_offset;
                        for(int64_t c =
                                blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
                            c < nc;
                            c += (int64_t)gridDim.x * blockDim.x)
                        {
                            double s = fixed[c] ? 0.0 : 1.0;
                            if(ccd.data())
                                s *= ccd->body_ccd_alpha_view()[gbo + body_ids[c]];
                            dir.as<3>(base / 3 + c) = s * dq.as<3>(c);
                        }
                    },
                    {}));

            // ---- Contact pair values at the trial positions ----
            cudaGraphNode_t n_contact_ready{};
            if(contact_)
            {
                auto n_fa =
                    m.subgraph(contact_->filter_assemble(vertex_->ctx().view().data(),
                                                         surface_->ctx().view().data(),
                                                         sctx.data(),
                                                         VarView<int32_t>{}),
                               trial_pos_deps)
                        .tail();
                n_contact_ready =
                    m.subgraph(contact_->sort_reduce(vertex_->ctx().view().data(),
                                                     surface_->ctx().view().data()),
                               {n_fa})
                        .tail();
            }

            // ---- Sequential per-term: memset b_rhs -> assemble_T -> dot ----
            auto memset_b = [&](const std::vector<cudaGraphNode_t>& deps)
            {
                GridStrideKernel<memset_b_rhs_kernel> k;
                return m.kernel(k, std::make_tuple(lctx.data()), deps);
            };
            auto dot_into = [&](int32_t slot, const std::vector<cudaGraphNode_t>& deps)
            {
                return m.kernel(
                    KernelConfig{"ls_gdot"}.grid(64).block(256),
                    [lctx,
                     dir = ls_replay_dir_.view(),
                     out = ls_replay_gdot_buf_.view(),
                     slot] __device__() mutable
                    {
                        auto   b   = lctx->b_rhs_view();
                        double acc = 0.0;
                        for(int64_t i =
                                blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
                            i < dir.size();
                            i += (int64_t)gridDim.x * blockDim.x)
                            acc += b[i] * dir[i];
                        out.atomic_add(slot, acc);
                    },
                    deps);
            };

            std::vector<cudaGraphNode_t> prev = state_tails;
            prev.push_back(n_memset_trip);
            prev.push_back(n_zero_gdot);
            prev.insert(prev.end(), pack_tails.begin(), pack_tails.end());

            if(abd_)
            {
                auto n0 = memset_b(prev);
                auto na =
                    m.subgraph(abd_->assemble(sctx.data(), lctx.data()), {n0}).tail();
                prev = {dot_into(1, {na})};
            }
            if(rigid_)
            {
                auto n0 = memset_b(prev);
                auto nr =
                    m.subgraph(rigid_->assemble(sctx.data(), lctx.data()), {n0}).tail();
                prev = {dot_into(2, {nr})};
            }
            if(fem_)
            {
                auto n0 = memset_b(prev);
                auto nf =
                    m.subgraph(fem_->assemble(sctx.data(), lctx.data()), {n0}).tail();
                prev = {dot_into(3, {nf})};
            }
            if(bspline_)
            {
                auto n0 = memset_b(prev);
                auto nb = m.subgraph(bspline_->assemble(sctx.data(), lctx.data()), {n0})
                              .tail();
                prev = {dot_into(4, {nb})};
            }
            if(has_joints())
            {
                auto n0 = memset_b(prev);
                auto nj = m.subgraph(joint_system_->assemble(sctx, lctx), {n0}).tail();
                prev = {dot_into(5, {nj})};
            }
            if(contact_)
            {
                auto deps = prev;
                deps.push_back(n_contact_ready);
                auto                         n0 = memset_b(deps);
                std::vector<cudaGraphNode_t> dist;
                if(abd_)
                {
                    dist.push_back(AbdGradientContactRoute::append(
                        m,
                        std::make_tuple(abd_->ctx().view().data(),
                                        vertex_->ctx().view().data(),
                                        contact_->ctx().view().data(),
                                        lctx.data()),
                        {n0}));
                }
                if(rigid_)
                {
                    dist.push_back(RigidGradientOnlyContactRoute::append(
                        m,
                        std::make_tuple(rigid_->ctx().view().data(),
                                        vertex_->ctx().view().data(),
                                        contact_->ctx().view().data(),
                                        lctx.data()),
                        {n0}));
                }
                if(fem_)
                {
                    dist.push_back(FemGradientContactRoute::append(
                        m,
                        std::make_tuple(fem_->ctx().view().data(),
                                        contact_->ctx().view().data(),
                                        lctx.data()),
                        {n0}));
                }
                if(bspline_)
                {
                    dist.push_back(BSplineGradientContactRoute::append(
                        m,
                        std::make_tuple(bspline_->ctx().view().data(),
                                        contact_->ctx().view().data(),
                                        lctx.data()),
                        {n0}));
                }
                prev = {dot_into(6, dist)};
            }

            m.kernel(
                KernelConfig{"ls_gdot_total"}.grid(1).block(1),
                [out = ls_replay_gdot_buf_.view()] __device__() mutable
                {
                    double t = 0.0;
                    for(int32_t i = 1; i < kLsReplayGradTerms; ++i)
                        t += out[i];
                    out[0] = t;
                },
                prev);
        });
}

}  // namespace qipc
