import os
import pickle as pkl
from typing import Any

import fast_simplification
import numpy as np
import trimesh

import genesis as gs
import genesis.utils.mesh as mu
import genesis.utils.gltf as gltf_utils
import genesis.utils.particle as pu
from genesis.options.surfaces import Surface
from genesis.repr_base import RBC
from genesis.utils.misc import redirect_libc_stderr


class Mesh(RBC):
    """
    Genesis's own triangle mesh object.

    This is a wrapper of `trimesh.Trimesh` with some additional features and attributes. The internal trimesh object
    can be accessed via `self.trimesh`.

    Parameters
    ----------
    surface : genesis.Surface
        The mesh's surface object.
    uvs : np.ndarray
        The mesh's uv coordinates.
    convexify : bool
        Whether to convexify the mesh.
    decimate : bool
        Whether to decimate the mesh.
    decimate_face_num : int
        The target number of faces after decimation.
    decimate_aggressiveness : int
        How hard the decimation process will try to match the target number of faces, as a integer ranging from 0 to 8.
        0 is losseless. 2 preserves all features of the original geometry. 5 may significantly alters
        the original geometry if necessary. 8 does what needs to be done at all costs. Default to 0.
    metadata : dict
        The metadata of the mesh.
    """

    def __init__(
        self,
        mesh,
        surface: Surface | None = None,
        uvs: "np.typing.NDArray | None" = None,
        scale: "np.typing.NDArray | float | None" = None,
        convexify=False,
        decimate=False,
        decimate_face_num=500,
        decimate_aggressiveness=0,
        metadata=None,
        is_mesh_zup: bool = True,
    ):
        self._uid = gs.UID()
        self._mesh = mesh  # .copy() FIXME: For some reason forcing copy is causing some tests to fails...
        self._surface = surface
        if uvs is not None:
            uvs = uvs.astype(gs.np_float, copy=False)
        self._uvs = uvs
        self._metadata: dict[str, Any] = metadata or {}
        self._color = np.array([1.0, 1.0, 1.0, 1.0], dtype=gs.np_float)

        # By default, all meshes are considered zup, unless the "FileMorph.file_meshes_are_zup" option was set to False
        self._metadata.setdefault("imported_as_zup", True)

        # By default, all meshes are considered having their original visual
        self._metadata.setdefault("is_visual_overwritten", False)

        if not is_mesh_zup:
            if self._metadata["imported_as_zup"]:
                self._mesh.apply_transform(mu.Y_UP_TRANSFORM.T)
            self._metadata["imported_as_zup"] = False

        if scale is not None:
            scale = np.atleast_1d(np.asarray(scale))
            assert scale.ndim == 1 and scale.size in (1, 3)
            self._mesh.apply_scale(scale)

        if self._surface.requires_uv:  # check uvs here
            if self._uvs is None:
                if "mesh_path" in self._metadata:
                    gs.logger.warning(
                        f"Texture given but asset missing uv info (or failed to load): {self._metadata['mesh_path']}"
                    )
                else:
                    gs.logger.warning("Texture given but asset missing uv info (or failed to load).")
        else:
            self._uvs = None

        if convexify:
            self.convexify()

        if decimate:
            self.decimate(decimate_face_num, decimate_aggressiveness)

    def convexify(self):
        """
        Convexify the mesh.
        """
        if self._mesh.vertices.shape[0] > 3:
            self._mesh = trimesh.convex.convex_hull(self._mesh)
            self._metadata["convexified"] = True
        self.clear_visuals()

    def decimate(self, decimate_face_num, decimate_aggressiveness):
        """
        Decimate the mesh.
        """
        if self._mesh.vertices.shape[0] > 3 and len(self._mesh.faces) > decimate_face_num:
            self._mesh.process(validate=True)
            self._mesh = trimesh.Trimesh(
                *fast_simplification.simplify(
                    self._mesh.vertices,
                    self._mesh.faces,
                    target_count=decimate_face_num,
                    agg=decimate_aggressiveness,
                    lossless=(decimate_aggressiveness == 0),
                ),
            )
            self._metadata["decimated"] = True

        self.clear_visuals()

    def remesh(self, edge_len_abs=None, edge_len_ratio=0.01, fix=True):
        """
        Remesh for tetrahedralization.
        """
        rm_file_path = mu.get_remesh_path(self.verts, self.faces, edge_len_abs, edge_len_ratio, fix)

        is_cached_loaded = False
        if os.path.exists(rm_file_path):
            gs.logger.debug("Remeshed file (`.rm`) found in cache.")
            try:
                with open(rm_file_path, "rb") as file:
                    verts, faces = pkl.load(file)
                is_cached_loaded = True
            except (EOFError, ModuleNotFoundError, pkl.UnpicklingError, TypeError, MemoryError):
                gs.logger.info("Ignoring corrupted cache.")

        if not is_cached_loaded:
            # Importing pymeshlab is very slow and not used very often. Let's delay import.
            with open(os.devnull, "w") as stderr, redirect_libc_stderr(stderr):
                import pymeshlab

            gs.logger.info("Remeshing for tetrahedralization...")
            ms = pymeshlab.MeshSet()
            ms.add_mesh(pymeshlab.Mesh(vertex_matrix=self.verts, face_matrix=self.faces))
            if edge_len_abs is not None:
                ms.meshing_isotropic_explicit_remeshing(targetlen=pymeshlab.PureValue(edge_len_abs))
            else:
                ms.meshing_isotropic_explicit_remeshing(targetlen=pymeshlab.PercentageValue(edge_len_ratio * 100))
            m = ms.current_mesh()
            verts, faces = m.vertex_matrix(), m.face_matrix()
            # Maybe we need to fix the mesh in some extreme cases with open3d
            # if fix:
            #     verts, faces = pymeshfix.clean_from_arrays(verts, faces)
            os.makedirs(os.path.dirname(rm_file_path), exist_ok=True)
            with open(rm_file_path, "wb") as file:
                pkl.dump((verts, faces), file)

        self._mesh = trimesh.Trimesh(vertices=verts, faces=faces)
        self.clear_visuals()

    def tetrahedralize(self, tet_cfg):
        """
        Tetrahedralize the mesh.
        """
        return mu.tetrahedralize_mesh(self._mesh, tet_cfg)

    def particlize(
        self,
        p_size=0.01,
        sampler="random",
    ):
        """
        Sample particles using the mesh volume.
        """
        if "pbs" in sampler:
            return pu.trimesh_to_particles_pbs(self._mesh, p_size, sampler)
        return pu.trimesh_to_particles_simple(self._mesh, p_size, sampler)

    def clear_visuals(self):
        """
        Clear the mesh's visual attributes by resetting the surface to gs.surfaces.Default().
        """
        self._surface = gs.surfaces.Default()
        self._surface.update_texture()

    def get_unique_edges(self):
        """
        Get the unique edges of the mesh.
        """
        r_face = np.roll(self.faces, 1, axis=1)
        edges = np.concatenate(np.array([self.faces, r_face]).T)

        # do a first pass to remove duplicates
        edges.sort(axis=1)
        edges = np.unique(edges, axis=0)
        edges = edges[edges[:, 0] != edges[:, 1]]

        return edges

    def copy(self):
        """
        Copy the mesh.
        """
        return Mesh(
            mesh=self._mesh.copy(**(dict(include_cache=True) if isinstance(self._mesh, trimesh.Trimesh) else {})),
            surface=self._surface.model_copy(),
            uvs=self._uvs.copy() if self._uvs is not None else None,
            metadata=self._metadata.copy(),
        )

    @classmethod
    def from_trimesh(
        cls,
        mesh,
        scale=None,
        convexify=False,
        decimate=False,
        decimate_face_num=500,
        decimate_aggressiveness=2,
        metadata=None,
        surface=None,
        is_mesh_zup=True,
        diffuse_texture_path=None,
    ):
        """
        Create a genesis.Mesh from a trimesh.Trimesh object.
        """
        if surface is None:
            surface = gs.surfaces.Default()
            surface.update_texture()
        else:
            surface = surface.model_copy()

        mesh = mesh.copy(**(dict(include_cache=True) if isinstance(mesh, trimesh.Trimesh) else {}))

        # Always parse uvs if available because roughness and normal map also need uvs.
        # Note that some visual may not have uv, e.g. ColorVisuals.
        uvs = None
        if isinstance(mesh.visual, trimesh.visual.texture.TextureVisuals) and mesh.visual.uv is not None:
            # Note that 'trimesh' uses uvs starting from top left corner.
            uvs = mesh.visual.uv.copy()
            uvs[:, 1] = 1.0 - uvs[:, 1]

        metadata = metadata or {}
        must_update_surface = True
        roughness_factor = None
        metallic_factor = None
        color_image = None
        color_factor = None
        opacity = 1.0

        visual = mesh.visual
        if isinstance(visual, trimesh.visual.texture.TextureVisuals) and visual.defined:
            if visual.kind == "texture":
                material = visual.material

                # TODO: Parsing PBR in obj or not
                # trimesh from .obj file will never use PBR material, but that from .glb file will
                if isinstance(material, trimesh.visual.material.PBRMaterial):
                    if material.baseColorTexture is not None:
                        color_image = mu.PIL_to_array(material.baseColorTexture)
                    if material.baseColorFactor is not None:
                        color_factor = tuple(np.array(material.baseColorFactor, dtype=np.float32) / 255.0)

                    if material.roughnessFactor is not None:
                        roughness_factor = (material.roughnessFactor,)

                    if material.metallicFactor is not None:
                        metallic_factor = (material.metallicFactor,)

                elif isinstance(material, trimesh.visual.material.SimpleMaterial):
                    if material.image is not None:
                        color_image = mu.PIL_to_array(material.image)
                    elif material.diffuse is not None:
                        color_factor = tuple(np.array(material.diffuse, dtype=np.float32) / 255.0)

                    if material.glossiness is not None:
                        roughness_factor = (mu.glossiness_to_roughness(material.glossiness),)

                    opacity = float(material.kwargs.get("d", [1.0])[0])
                    if opacity < 1.0:
                        if color_factor is None:
                            color_factor = (1.0, 1.0, 1.0, opacity)
                        else:
                            color_factor = (*color_factor[:3], color_factor[3] * opacity)
                else:
                    gs.raise_exception(f"Unsupported Trimesh material type '{type(material)}'.")
            else:
                # TODO: support vertex/face colors in luisa
                color_factor = tuple(np.array(visual.main_color, dtype=np.float32) / 255.0)
        elif isinstance(surface.texture, gs.textures.ColorTexture):
            color_factor = surface.texture.color
        elif (isinstance(visual, trimesh.visual.color.ColorVisuals) and visual.defined) or (
            isinstance(visual, trimesh.visual.color.VertexColor) and visual.vertex_colors.size > 0
        ):
            # Color is already vertex-based. It is not only necessary to create a new visual.
            must_update_surface = False
        else:
            # use white color as default
            color_factor = (1.0, 1.0, 1.0, 1.0)

        if must_update_surface:
            metadata["is_visual_overwritten"] = isinstance(surface.texture, gs.textures.ColorTexture)

            color_texture = mu.create_texture(color_image, color_factor, "srgb")
            # Attach the resolved file path so downstream renderers (e.g. Nyx)
            # can load the texture from disk without metadata workarounds.
            if (
                diffuse_texture_path is not None
                and isinstance(color_texture, gs.textures.ImageTexture)
                and not color_texture.image_path
            ):
                color_texture.image_path = diffuse_texture_path
            opacity_texture = None
            if color_texture is not None:
                opacity_texture = color_texture.check_dim(3)
            roughness_texture = mu.create_texture(None, roughness_factor, "linear")
            metallic_texture = mu.create_texture(None, metallic_factor, "linear")

            # Force update when mesh has its own texture or color factor (from OBJ/glTF material),
            # otherwise the surface's pre-existing default ColorTexture takes priority.
            has_mesh_texture = color_image is not None or color_factor is not None
            surface.update_texture(
                color_texture=color_texture,
                opacity_texture=opacity_texture,
                roughness_texture=roughness_texture,
                metallic_texture=metallic_texture,
                force=has_mesh_texture,
            )
            mesh.visual = mu.surface_uvs_to_trimesh_visual(surface, uvs, len(mesh.vertices))

        return cls(
            mesh=mesh,
            surface=surface,
            uvs=uvs,
            scale=scale,
            convexify=convexify,
            decimate=decimate,
            decimate_face_num=decimate_face_num,
            decimate_aggressiveness=decimate_aggressiveness,
            metadata=metadata,
            is_mesh_zup=is_mesh_zup,
        )

    @classmethod
    def from_attrs(
        cls, verts, faces, normals=None, surface=None, uvs=None, scale=None, metadata=None, is_mesh_zup=True
    ):
        """
        Create a genesis.Mesh from mesh attributes including vertices, faces, and normals.
        """
        if surface is None:
            surface = gs.surfaces.Default()

        metadata = metadata or {}
        metadata["is_visual_overwritten"] = metadata.get("is_visual_overwritten") or (surface.texture is not None)
        visual = mu.surface_uvs_to_trimesh_visual(surface, uvs, len(verts))

        tmesh = trimesh.Trimesh(
            vertices=verts,
            faces=faces,
            vertex_normals=normals,
            visual=visual,
            process=False,
        )

        return cls(
            mesh=tmesh,
            surface=surface,
            uvs=uvs,
            scale=scale,
            metadata=metadata,
            is_mesh_zup=is_mesh_zup,
        )

    @classmethod
    def from_morph_surface(cls, morph, surface=None) -> "list[gs.Mesh] | gs.Mesh":
        """
        Create a genesis.Mesh from morph and surface options.

        If the morph is a Mesh morph (morphs.Mesh), it could contain multiple sub-meshes, so we return a list.
        """
        if isinstance(morph, gs.options.morphs.Mesh):
            if morph.is_format(gs.options.morphs.MESH_FORMATS):
                if morph.is_format(gs.options.morphs.GLTF_FORMATS):
                    meshes = gltf_utils.parse_mesh_glb(
                        morph.file, morph.group_by_material, morph.scale, morph.file_meshes_are_zup, surface
                    )
                else:
                    meshes = mu.parse_mesh_trimesh(
                        morph.file, morph.group_by_material, morph.scale, morph.file_meshes_are_zup, surface
                    )
            elif isinstance(morph, gs.options.morphs.MeshSet):
                assert all(isinstance(mesh, trimesh.Trimesh) for mesh in morph.files)
                meshes = [mu.trimesh_to_mesh(mesh, morph.scale, surface) for mesh in morph.files]
            else:
                gs.raise_exception(f"File type not supported: {morph.file}")

            return meshes

        if isinstance(morph, gs.options.morphs.Box):
            tmesh = mu.create_box(extents=morph.size)
        elif isinstance(morph, gs.options.morphs.Cylinder):
            tmesh = mu.create_cylinder(radius=morph.radius, height=morph.height)
        elif isinstance(morph, gs.options.morphs.Sphere):
            tmesh = mu.create_sphere(radius=morph.radius)
        else:
            gs.raise_exception(f"Morph {morph} not supported by this method.")

        return [cls.from_trimesh(tmesh, surface=surface)]

    def set_color(self, color):
        """
        Set the mesh's color.
        """
        self._color = color
        color_texture = gs.textures.ColorTexture(color=tuple(color))
        opacity_texture = color_texture.check_dim(3)
        self._surface.update_texture(color_texture=color_texture, opacity_texture=opacity_texture, force=True)
        self.update_trimesh_visual()

    def update_trimesh_visual(self):
        """
        Update the trimesh obj's visual attributes using its surface and uvs.
        """
        self._mesh.visual = mu.surface_uvs_to_trimesh_visual(self.surface, self.uvs, len(self.verts))
        self._metadata["is_visual_overwritten"] = True

    def apply_transform(self, T):
        """
        Apply a 4x4 transformation matrix (translation on the right column) to the mesh.
        """
        self._mesh.apply_transform(T)

    @property
    def uid(self):
        """
        Return the mesh's uid.
        """
        return self._uid

    @property
    def trimesh(self):
        """
        Return the mesh's trimesh object.
        """
        return self._mesh

    @property
    def is_convex(self) -> bool:
        """
        Whether the mesh is convex.
        """
        return self.metadata.get("convexified", self._mesh.is_convex)

    @property
    def metadata(self):
        """
        Metadata of the mesh.
        """
        return self._metadata

    @property
    def verts(self):
        """
        Vertices of the mesh.
        """
        return self._mesh.vertices

    @verts.setter
    def verts(self, verts):
        """
        Set the vertices of the mesh.
        """
        assert len(verts) == len(self.verts)
        self._mesh.vertices = verts

    @property
    def faces(self):
        """
        Faces of the mesh.
        """
        return self._mesh.faces

    @property
    def normals(self):
        """
        Normals of the mesh.
        """
        return self._mesh.vertex_normals

    @property
    def surface(self):
        """
        Surface of the mesh.
        """
        return self._surface

    @property
    def uvs(self):
        """
        UVs of the mesh.
        """
        return self._uvs

    @property
    def area(self):
        """
        Surface area of the mesh.
        """
        return self._mesh.area

    @property
    def volume(self):
        """
        Volume of the mesh.
        """
        return self._mesh.volume


class LineMesh(RBC):
    """A 1D line mesh (vertices + edges) for rope/string simulation.

    Loaded from OBJ files with 'l' (line) elements. Handles scale and
    Y-up→Z-up conversion, matching the transform convention of gs.Mesh.
    """

    def __init__(self, vertices, edges, radius, surface=None, is_mesh_zup=True, scale=None, n_tube_sides=6):
        self._vertices = np.array(vertices, dtype=gs.np_float)
        self._edges = np.array(edges, dtype=np.int32)
        self._surface = surface or gs.surfaces.Default()
        self._n_tube_sides = n_tube_sides
        self._tube_radius = radius
        self._tube_faces = None
        self._tube_angles = None

        if not is_mesh_zup:
            self._vertices = (mu.Y_UP_TRANSFORM.T[:3, :3] @ self._vertices.T).T

        if scale is not None:
            scale = np.atleast_1d(np.asarray(scale))
            self._vertices *= scale

        self._build_tube_topology()

    @classmethod
    def from_morph_surface(cls, morph, radius, surface=None):
        """Create LineMesh(es) from a Mesh morph.

        Currently only supports OBJ format (with 'l' line elements).
        Returns a list of LineMesh objects (one per geometry group in the file).
        """
        if not isinstance(morph, gs.options.morphs.Mesh):
            gs.raise_exception(f"LineMesh only supports Mesh morph. Got: {morph}.")
        if not morph.is_format(".obj"):
            gs.raise_exception(f"LineMesh only supports OBJ format. Got: {morph.file}")
        if surface is None:
            surface = gs.surfaces.Default()
        return mu.parse_mesh_linemesh(morph.file, morph.scale, radius, morph.file_meshes_are_zup, surface)

    @property
    def verts(self):
        """Vertex positions."""
        return self._vertices

    @property
    def edges(self):
        """Edge connectivity (N, 2) array."""
        return self._edges

    @property
    def uvs(self):
        """UVs (always None for line meshes)."""
        return None

    @property
    def surface(self):
        """Surface material."""
        return self._surface

    def _build_tube_topology(self):
        """Pre-build tube face topology for rendering.

        Uses ``n_tube_sides`` and ``tube_radius`` from init. Each vertex
        becomes a ring of vertices; each edge becomes a cylinder segment.
        Tube vertex positions are rebuilt each frame via ``build_tube_verts()``.
        """
        n_sides = self._n_tube_sides
        n_edges = len(self._edges)

        tube_faces = np.zeros((n_edges * 2 * n_sides, 3), dtype=np.int32)
        for ei, (v0, v1) in enumerate(self._edges):
            for si in range(n_sides):
                sn = (si + 1) % n_sides
                a = v0 * n_sides + si
                b = v0 * n_sides + sn
                c = v1 * n_sides + si
                d = v1 * n_sides + sn
                tube_faces[ei * 2 * n_sides + 2 * si] = [a, b, c]
                tube_faces[ei * 2 * n_sides + 2 * si + 1] = [b, d, c]

        self._tube_faces = tube_faces
        self._tube_angles = np.linspace(0, 2 * np.pi, n_sides, endpoint=False)

    def build_tube_verts(self, sim_verts):
        """Rebuild tube vertex positions from current sim vertex positions.

        Returns (n_verts * n_sides, 3) array. Call ``build_tube_topology()``
        first to set up the face topology.

        Uses parallel transport to propagate the local frame along the
        curve, avoiding the ring-twist self-intersections that occur when
        each vertex computes its frame independently.
        """
        n_sides = self._n_tube_sides
        radius = self._tube_radius
        angles = self._tube_angles
        n_verts = len(sim_verts)

        # Per-vertex tangent
        tangents = np.zeros_like(sim_verts)
        for e in self._edges:
            d = sim_verts[e[1]] - sim_verts[e[0]]
            tangents[e[0]] += d
            tangents[e[1]] += d
        norms = np.linalg.norm(tangents, axis=1, keepdims=True)
        norms[norms < 1e-12] = 1.0
        tangents /= norms

        # Build vertex ordering along the curve from edge list
        order = self._tube_vertex_order(n_verts)

        # Seed frame at the first vertex
        t0 = tangents[order[0]]
        up = np.array([0.0, 0.0, 1.0])
        if abs(np.dot(t0, up)) > 0.9:
            up = np.array([1.0, 0.0, 0.0])
        n1 = np.cross(t0, up)
        n1 /= np.linalg.norm(n1) + 1e-15
        n2 = np.cross(t0, n1)

        # Parallel-transport the frame along the curve
        frames_n1 = np.empty((n_verts, 3), dtype=sim_verts.dtype)
        frames_n2 = np.empty((n_verts, 3), dtype=sim_verts.dtype)
        frames_n1[order[0]] = n1
        frames_n2[order[0]] = n2

        for i in range(1, len(order)):
            vi_prev = order[i - 1]
            vi = order[i]
            t_prev = tangents[vi_prev]
            t_cur = tangents[vi]

            # Rotation axis and angle between consecutive tangents
            b = np.cross(t_prev, t_cur)
            b_len = np.linalg.norm(b)
            if b_len < 1e-12:
                # Tangents are parallel — keep the same frame
                frames_n1[vi] = frames_n1[vi_prev]
                frames_n2[vi] = frames_n2[vi_prev]
            else:
                b /= b_len
                cos_a = np.clip(np.dot(t_prev, t_cur), -1.0, 1.0)
                sin_a = b_len
                # Rodrigues rotation of n1 around b
                prev_n1 = frames_n1[vi_prev]
                rotated = prev_n1 * cos_a + np.cross(b, prev_n1) * sin_a + b * np.dot(b, prev_n1) * (1.0 - cos_a)
                frames_n1[vi] = rotated
                frames_n2[vi] = np.cross(t_cur, rotated)

        # Build tube ring vertices
        tube_verts = np.zeros((n_verts * n_sides, 3), dtype=sim_verts.dtype)
        for vi in range(n_verts):
            n1_vi = frames_n1[vi]
            n2_vi = frames_n2[vi]
            for si in range(n_sides):
                tube_verts[vi * n_sides + si] = sim_verts[vi] + radius * (
                    np.cos(angles[si]) * n1_vi + np.sin(angles[si]) * n2_vi
                )
        return tube_verts

    def _tube_vertex_order(self, n_verts):
        """Return vertex indices in curve order from the edge list."""
        if not hasattr(self, "_cached_tube_order") or self._cached_tube_order is None:
            from collections import defaultdict

            adj = defaultdict(list)
            for e in self._edges:
                adj[e[0]].append(e[1])
                adj[e[1]].append(e[0])

            # Find an endpoint (degree 1) to start from
            start = 0
            for v in range(n_verts):
                if len(adj[v]) == 1:
                    start = v
                    break

            order = [start]
            visited = {start}
            cur = start
            while len(order) < n_verts:
                found = False
                for nb in adj[cur]:
                    if nb not in visited:
                        order.append(nb)
                        visited.add(nb)
                        cur = nb
                        found = True
                        break
                if not found:
                    # Walk got stuck (branch or disconnected component).
                    # Find any unvisited vertex connected to a visited one
                    # to continue with a smooth frame transition.
                    restarted = False
                    for v_visited in reversed(order):
                        for nb in adj[v_visited]:
                            if nb not in visited:
                                order.append(nb)
                                visited.add(nb)
                                cur = nb
                                restarted = True
                                break
                        if restarted:
                            break
                    if not restarted:
                        break
            self._cached_tube_order = order
        return self._cached_tube_order

    @property
    def tube_faces(self):
        """Pre-built tube face topology. None if ``build_tube_topology()`` not called."""
        return self._tube_faces
