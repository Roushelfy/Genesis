/// @file edge_triangle_intersect.h
/// @brief Device-only edge-triangle intersection test and BVH query kernel.
///
/// Ported from libuipc geometry::tri_edge_intersect (plane intersection +
/// point-in-triangle with full coplanar handling).
/// Reference: external/libuipc/src/geometry/intersection.cpp

#pragma once
#include <cuda_runtime.h>
#include <solver/bvh_types.cuh>
#include <solver/global_vertex_context.h>
#include <solver/global_surface_context.h>
#include <solver/global_body_context.h>
#include <solver/contact_context.h>
#include <solver/contact_function/contact_table_query.h>
#include <solver/et_check_context.h>
#include <buffer_view.h>
#include <var.h>
#include <cstdint>

namespace qipc
{

// ============================================================================
// Helper: double3 arithmetic
// ============================================================================

__device__ inline double3 d3_add(double3 a, double3 b)
{
    return make_double3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__device__ inline double3 d3_scale(double3 v, double s)
{
    return make_double3(v.x * s, v.y * s, v.z * s);
}

// ============================================================================
// Point-in-triangle (barycentric coordinates)
// ============================================================================

__device__ inline bool point_in_triangle(double3 P, double3 T0, double3 T1, double3 T2)
{
    double3 u  = d3_sub(T1, T0);
    double3 v  = d3_sub(T2, T0);
    double3 w  = d3_sub(P, T0);
    double  uu = d3_dot(u, u);
    double  uv = d3_dot(u, v);
    double  vv = d3_dot(v, v);
    double  wu = d3_dot(w, u);
    double  wv = d3_dot(w, v);

    double denom = uv * uv - uu * vv;
    if(fabs(denom) < 1e-30)
        return false;

    double s = (uv * wv - vv * wu) / denom;
    double t = (uv * wu - uu * wv) / denom;
    return (s >= -1e-12) && (t >= -1e-12) && (s + t <= 1.0 + 1e-12);
}

// ============================================================================
// Segment-segment intersection (3D, for coplanar case)
//
// Tests whether segments (A, A+dA) and (B, B+dB) intersect.
// Solves: A + u*dA = B + v*dB via the cross-product method.
// ============================================================================

__device__ inline bool segment_segment_intersect(double3 A, double3 dA, double3 B, double3 dB)
{
    double3 AB         = d3_sub(B, A);
    double3 cross_dirs = d3_cross(dA, dB);
    double  denom      = d3_dot(cross_dirs, cross_dirs);

    if(denom < 1e-30)
        return false;

    double3 cross_ab_db = d3_cross(AB, dB);
    double  u           = d3_dot(cross_ab_db, cross_dirs) / denom;

    double3 cross_ab_da = d3_cross(AB, dA);
    double  v           = d3_dot(cross_ab_da, cross_dirs) / denom;

    return (u >= -1e-12 && u <= 1.0 + 1e-12 && v >= -1e-12 && v <= 1.0 + 1e-12);
}

// ============================================================================
// Edge-triangle intersection
//
// Returns true if segment [E0, E1] intersects triangle (T0, T1, T2).
// Handles general case (edge crosses triangle plane) and coplanar case.
// ============================================================================

__device__ inline bool tri_edge_intersect(double3 T0, double3 T1, double3 T2, double3 E0, double3 E1)
{
    constexpr double eps = 1e-14;

    double3 n  = d3_cross(d3_sub(T1, T0), d3_sub(T2, T0));
    double  da = d3_dot(d3_sub(E0, T0), n);
    double  db = d3_dot(d3_sub(E1, T0), n);

    if((da > 0.0 && db > 0.0) || (da < 0.0 && db < 0.0))
        return false;

    if(fabs(da) < eps && fabs(db) < eps)
    {
        if(point_in_triangle(E0, T0, T1, T2))
            return true;
        if(point_in_triangle(E1, T0, T1, T2))
            return true;

        double3 edge_dir = d3_sub(E1, E0);
        if(segment_segment_intersect(E0, edge_dir, T0, d3_sub(T1, T0)))
            return true;
        if(segment_segment_intersect(E0, edge_dir, T1, d3_sub(T2, T1)))
            return true;
        if(segment_segment_intersect(E0, edge_dir, T2, d3_sub(T0, T2)))
            return true;

        return false;
    }

    double t = da / (da - db);
    if(t < 0.0 || t > 1.0)
        return false;

    double3 P = d3_add(E0, d3_scale(d3_sub(E1, E0), t));
    return point_in_triangle(P, T0, T1, T2);
}

// ============================================================================
// ET query kernel: traverse face BVH with edge AABBs
// ============================================================================

constexpr int32_t kETBVHStackCapacity = 64;

/// body / cctx may be null (no body manager / no contact system): the
/// corresponding suppression rules are skipped. With both wired, the checker
/// ignores exactly the pairs the contact stack ignores (same-body with
/// self-collision off, joint-connected bodies, tabular-disabled element
/// pairs), so by-design overlaps -- e.g. URDF links interpenetrating at
/// their joints -- do not drown out real violations.
///
/// The pair buffer and its capacity are read through ETCheckContext so a
/// capacity growth (ET overflow checkpoint) only republishes the context;
/// n_et_pairs counts the TRUE demand (writes are clamped at capacity).
__global__ void et_query_kernel(const GlobalVertexContext*  vtx,
                                const GlobalSurfaceContext* surf,
                                const GlobalBodyContext*    body,
                                const GlobalContactContext* cctx,
                                BufferView<BVHNode>         nodes,
                                BufferView<AABB>            aabbs,
                                const ETCheckContext*       et,
                                VarView<int32_t>            n_et_pairs)
{
    auto pos   = vtx->positions_view();
    auto edges = surf->surf_edges_view();
    auto tris  = surf->surf_triangles_view();

    int64_t       n_edges = surf->n_surf_edges();
    const int32_t stride  = blockDim.x * gridDim.x;

    for(int32_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < n_edges; idx += stride)
    {
        uint32_t e0_gvid = edges[idx * 2 + 0];
        uint32_t e1_gvid = edges[idx * 2 + 1];

        double3 p0 =
            make_double3(pos[e0_gvid * 3], pos[e0_gvid * 3 + 1], pos[e0_gvid * 3 + 2]);
        double3 p1 =
            make_double3(pos[e1_gvid * 3], pos[e1_gvid * 3 + 1], pos[e1_gvid * 3 + 2]);

        AABB query_bv;
        query_bv.upper = p0;
        query_bv.lower = p0;
        query_bv.combine(p1);

        uint32_t stack[kETBVHStackCapacity];
        int32_t  stack_top = 0;
        stack[stack_top++] = 0;

        do
        {
            const uint32_t node_id = stack[--stack_top];
            const uint32_t L_idx   = nodes[node_id].left_idx;
            const uint32_t R_idx   = nodes[node_id].right_idx;

            auto process_child = [&](uint32_t child_idx)
            {
                if(!aabb_overlap(query_bv, aabbs[child_idx]))
                    return;

                const uint32_t obj_idx = nodes[child_idx].element_idx;
                if(obj_idx == 0xFFFFFFFF)
                {
                    if(stack_top < kETBVHStackCapacity)
                        stack[stack_top++] = child_idx;
                    return;
                }

                uint32_t f0 = tris[obj_idx * 3 + 0];
                uint32_t f1 = tris[obj_idx * 3 + 1];
                uint32_t f2 = tris[obj_idx * 3 + 2];

                if(e0_gvid == f0 || e0_gvid == f1 || e0_gvid == f2
                   || e1_gvid == f0 || e1_gvid == f1 || e1_gvid == f2)
                    return;

                // Contact-stack suppression (ContactPTPred rules 1, 2, 4).
                if(body != nullptr)
                {
                    auto    body_ids = vtx->body_id_view();
                    auto    self_col = body->self_collision_view();
                    int32_t bi       = body_ids[e0_gvid];
                    int32_t bj       = body_ids[f0];
                    if(bi == bj && bi >= 0 && !self_col[bi])
                        return;
                    if(bi >= 0 && bj >= 0 && bi != bj
                       && body->is_body_contact_ignored(bi, bj))
                        return;
                }
                if(cctx != nullptr && cctx->n_contact_elements > 0)
                {
                    auto    eids = cctx->vert_contact_element_ids_view();
                    auto    mask = cctx->contact_mask_table_view();
                    int32_t N    = cctx->n_contact_elements;
                    if(!ct_enabled_pt(mask.data(), N, eids[e0_gvid], eids[f0], eids[f1], eids[f2])
                       || !ct_enabled_pt(
                           mask.data(), N, eids[e1_gvid], eids[f0], eids[f1], eids[f2]))
                        return;
                }

                double3 v0 = make_double3(pos[f0 * 3], pos[f0 * 3 + 1], pos[f0 * 3 + 2]);
                double3 v1 = make_double3(pos[f1 * 3], pos[f1 * 3 + 1], pos[f1 * 3 + 2]);
                double3 v2 = make_double3(pos[f2 * 3], pos[f2 * 3 + 1], pos[f2 * 3 + 2]);

                if(tri_edge_intersect(v0, v1, v2, p0, p1))
                {
                    int32_t ci = atomicAdd(n_et_pairs.data(), 1);
                    if(ci < et->max_pairs)
                    {
                        auto pairs        = et->pairs_view();
                        pairs[ci * 2 + 0] = idx;
                        pairs[ci * 2 + 1] = (int32_t)obj_idx;
                    }
                }
            };

            process_child(L_idx);
            process_child(R_idx);
        } while(stack_top > 0);
    }
}

}  // namespace qipc
