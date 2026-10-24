"""CPU automatic Hertz size and conservative projected-DPR shell loading.

The Hertz disk is defined in the supplied contact-normal plane, centred on
the original rigid application point. Samples project onto the actual offset
Q4 surface ALONG THE RESULTANT FORCE. Consequently this is a labelled
projected-Hertz DPR load law, not an independent shell-contact solution and
not exactly Genesis's surface-ball pressure law on curved surfaces.

The force direction, normal force, physical radius and point wrench are kept.
Rotational nodal loads account for the actual inner/outer surface offset.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import numpy as np
from scipy.spatial import cKDTree
from scipy.special import roots_jacobi


@dataclass
class Contact:
    point: np.ndarray
    force: np.ndarray
    normal: np.ndarray  # OUTWARD material-surface normal; compressive Fn=-F.n.
    curvatures: np.ndarray | tuple = (50.0, 50.0)  # Combined relative principal curvatures, /m.
    side: int = 1
    radius: float | None = None
    friction: float | None = None
    face_hint: int | None = None  # Current collision witness, checked before reuse.


class ContactIntegrationError(ValueError):
    pass


def hertz_radius(normal_force, E=1e10, nu=.3, curvatures=(50., 50.), contact_modulus=None):
    k = np.asarray(curvatures, dtype=float)
    if k.shape != (2,) or not np.isfinite(k).all() or np.min(k) <= 0:
        raise ValueError("Hertz estimate requires two positive combined principal curvatures")
    if not np.isfinite(normal_force) or normal_force < 0:
        raise ValueError("Normal load must be finite and compressive")
    modulus = E / (1-nu*nu) if contact_modulus is None else float(contact_modulus)
    if modulus <= 0:
        raise ValueError("Contact modulus must be positive")
    return (3*normal_force/(4*modulus*np.mean(k))) ** (1/3)


def _basis(normal):
    n = np.asarray(normal, dtype=float)
    n = n / np.linalg.norm(n)
    e1 = np.cross(n, np.eye(3)[np.argmin(np.abs(n))])
    e1 /= np.linalg.norm(e1)
    return np.column_stack((e1, np.cross(n, e1)))


def _shape(x, y):
    x, y = np.broadcast_arrays(x, y)
    N = .25*np.stack(((1-x)*(1-y), (1+x)*(1-y), (1+x)*(1+y), (1-x)*(1+y)), axis=-1)
    dx = .25*np.stack((-(1-y), 1-y, 1+y, -(1+y)), axis=-1)
    dy = .25*np.stack((-(1-x), -(1+x), 1+x, 1-x), axis=-1)
    return N, dx, dy


@lru_cache(maxsize=32)
def _gauss(n):
    return np.polynomial.legendre.leggauss(int(n))


@lru_cache(maxsize=8)
def _disk_cubature(n):
    # t=r²/a² has normalized density (3/2)*sqrt(1-t), t in [0,1].
    x, w = roots_jacobi(int(n), .5, 0.)
    r = np.sqrt(.5*(1+x))
    angular = 4*int(n)
    theta = 2*np.pi*(np.arange(angular)+.5)/angular
    points = np.stack((r[:, None]*np.cos(theta), r[:, None]*np.sin(theta)), axis=-1).reshape(-1, 2)
    weights = np.broadcast_to((w/w.sum())[:, None]/angular, (len(r), angular)).ravel().copy()
    return points, weights


class ShellDPRTransfer:
    def __init__(self, model, contact_modulus=None):
        self.model = model
        self.vertices = np.asarray(model.vertices)
        self.quads = np.asarray(model.quads)
        self.directors = np.asarray(model.directors)
        self.h = float(model.h)
        self.E, self.nu = float(model.E), float(model.nu)
        self.ndof = 6*len(self.vertices)
        self.contact_modulus = contact_modulus
        self.faces = {}
        for side in (-1, 1):
            points = self.vertices + side*.5*self.h*self.directors
            face_points = points[self.quads]
            center = face_points.mean(axis=1)
            radius = np.linalg.norm(face_points-center[:, None], axis=2).max(axis=1)
            self.faces[side] = (face_points, center, radius, cKDTree(center))
        edge_faces = {}
        self.adjacent = [set() for _ in self.quads]
        for face, quad in enumerate(self.quads):
            for i in range(4):
                key = tuple(sorted((int(quad[i]), int(quad[(i+1) % 4]))))
                edge_faces.setdefault(key, []).append(face)
        for entries in edge_faces.values():
            for f in entries:
                self.adjacent[f].update(set(entries)-{f})

    @staticmethod
    def _inverse(projected_vertices, points):
        """Vectorized inversion of one projected bilinear Q4."""
        points = np.atleast_2d(points)
        x, y = np.zeros(len(points)), np.zeros(len(points))
        origin = projected_vertices.mean(axis=0)
        W = projected_vertices-origin
        target = points-origin
        good = np.ones(len(points), dtype=bool)
        for _ in range(12):
            N, dx, dy = _shape(x, y)
            residual = N @ W-target
            jx, jy = dx @ W, dy @ W
            determinant = jx[:, 0]*jy[:, 1]-jx[:, 1]*jy[:, 0]
            ok = np.abs(determinant) > 1e-30
            good &= ok
            det = np.where(ok, determinant, 1.)
            sx = (jy[:, 1]*residual[:, 0]-jy[:, 0]*residual[:, 1])/det
            sy = (-jx[:, 1]*residual[:, 0]+jx[:, 0]*residual[:, 1])/det
            x -= sx
            y -= sy
            if np.max(np.abs(np.r_[sx[good], sy[good]]), initial=0.) < 3e-14:
                break
        N, _, _ = _shape(x, y)
        scale = max(np.linalg.norm(W, axis=1).max(), 1e-15)
        good &= np.linalg.norm(N @ W-target, axis=1) < 2e-11*scale
        good &= (np.abs(x) <= 1+1e-9) & (np.abs(y) <= 1+1e-9)
        return N, np.column_stack((x, y)), good

    @staticmethod
    def _projected_polygon(face_points, point, normal, force_direction, plane_basis):
        points = face_points-point
        points = points - np.outer(points @ normal/(force_direction @ normal), force_direction)
        return points @ plane_basis

    def _candidates(self, point, radius, side):
        _, centers, extent, tree = self.faces[side]
        faces = tree.query_ball_point(point, radius+extent.max()+1e-12)
        if not faces:
            _, index = tree.query(point, k=min(12, len(centers)))
            faces = np.atleast_1d(index).tolist()
        return np.asarray(sorted(faces), dtype=int)

    @staticmethod
    def _analytic_weights(poly, radius):
        """Exact Q4 moment integration for a complete affine disk."""
        scale = max(np.linalg.norm(poly-poly.mean(axis=0), axis=1).max(), 1e-15)
        if np.linalg.norm(poly[0]+poly[2]-poly[1]-poly[3]) > 1e-11*scale:
            return None
        N, xy, good = ShellDPRTransfer._inverse(poly, [[0., 0.]])
        if not good[0]:
            return None
        _, dx, dy = _shape(xy[0, 0], xy[0, 1])
        J = np.column_stack((dx @ poly, dy @ poly))
        if abs(np.linalg.det(J)) < 1e-30:
            return None
        A = np.linalg.inv(J)
        if np.any(np.abs(xy[0])+radius*np.linalg.norm(A, axis=1) > 1-2e-12):
            return None
        covariance = radius*radius/5.
        cross = xy[0, 0]*xy[0, 1]+covariance*(A[0] @ A[1])
        sx = np.array([-1, 1, 1, -1])
        sy = np.array([-1, -1, 1, 1])
        return .25*(1+sx*xy[0, 0]+sy*xy[0, 1]+sx*sy*cross)

    @staticmethod
    def _full_disk(poly, radius):
        area = np.sum(poly[:, 0]*np.roll(poly[:, 1], -1)-poly[:, 1]*np.roll(poly[:, 0], -1))
        p = poly if area > 0 else poly[::-1]
        edges = np.roll(p, -1, axis=0)-p
        distances = (edges[:, 0]*(-p[:, 1])-edges[:, 1]*(-p[:, 0]))/np.linalg.norm(edges, axis=1)
        return bool(np.min(distances) >= radius*(1+2e-12))

    @staticmethod
    def _taylor_weights(poly, radius):
        """Second-moment approximation for a complete non-affine Q4 disk.

        This opt-in path retains the disk's exact mass, centroid and covariance,
        but approximates the inverse bilinear shape functions to second order.
        It must be compared against the independent higher-order clipped rule;
        it is never silently selected by method='auto'.
        """
        if not ShellDPRTransfer._full_disk(poly, radius):
            return None
        N, xy, good = ShellDPRTransfer._inverse(poly, [[0., 0.]])
        if not good[0]:
            return None
        _, dx, dy = _shape(xy[0, 0], xy[0, 1])
        J = np.column_stack((dx @ poly, dy @ poly))
        A = np.linalg.inv(J)
        mixed = .25*(poly[0]-poly[1]+poly[2]-poly[3])
        s = A[0] @ A[1]
        laplacian_coordinates = -2*(A @ mixed)*s
        sx = np.array([-1., 1., 1., -1.])
        sy = np.array([-1., -1., 1., 1.])
        gradient = .25*np.column_stack((sx+sx*sy*xy[0, 1], sy+sx*sy*xy[0, 0]))
        laplacian_shape = gradient @ laplacian_coordinates+.5*sx*sy*s
        return N[0]+radius*radius/10*laplacian_shape

    @staticmethod
    def _embedded_weights(poly, radius, tolerance=1e-10):
        """Embedded degree-7/11 cubature on a complete disk.

        Degree labels refer to polynomial exactness; inverse Q4 shapes are not
        polynomial. Their 16/36-point difference is an error indicator, not a
        mathematical error certificate. Failed indicators retain full polar
        quadrature. This path is opt-in and checked against independent rules.
        """
        if not ShellDPRTransfer._full_disk(poly, radius):
            return None, None
        p2, w2 = _disk_cubature(2)
        p3, w3 = _disk_cubature(3)
        N, _, valid = ShellDPRTransfer._inverse(poly, radius*np.vstack((p2, p3)))
        if not valid.all():
            return None, None
        lower, upper = w2 @ N[:len(w2)], w3 @ N[len(w2):]
        difference = float(np.linalg.norm(upper-lower)/np.linalg.norm(upper))
        return (upper if difference <= tolerance else None), difference

    @staticmethod
    def _radial_interval(poly, angle, radius):
        d = np.column_stack((np.cos(angle), np.sin(angle)))
        low = np.zeros(len(angle))
        high = np.full(len(angle), radius)
        for a, b in zip(poly, np.roll(poly, -1, axis=0)):
            edge = b-a
            A = edge[0]*d[:, 1]-edge[1]*d[:, 0]
            B = edge[0]*a[1]-edge[1]*a[0]
            positive, negative = A > 1e-18, A < -1e-18
            low = np.where(positive, np.maximum(low, B/np.where(positive, A, 1)), low)
            high = np.where(negative, np.minimum(high, B/np.where(negative, A, -1)), high)
            if B > 1e-16:
                high = np.where(~positive & ~negative, -1., high)
        return np.maximum(low, 0), np.minimum(high, radius)

    @staticmethod
    def _face_polar_samples(poly, radius, order):
        """Face-clipped polar rule, split at vertices and circle intersections."""
        area = np.sum(poly[:, 0]*np.roll(poly[:, 1], -1)-poly[:, 1]*np.roll(poly[:, 0], -1))
        if area < 0:
            poly = poly[::-1]
        if ShellDPRTransfer._full_disk(poly, radius):
            gr, wr = _gauss(max(6, int(order)//2))
            phi = np.pi/4*(1+gr)
            theta = 2*np.pi*(np.arange(max(16, 2*int(order)))+.5)/max(16, 2*int(order))
            r = radius*np.sin(phi)
            points = np.stack((r[:, None]*np.cos(theta), r[:, None]*np.sin(theta)), axis=-1).reshape(-1, 2)
            weights = np.broadcast_to((3*np.sin(phi)*np.cos(phi)**2*np.pi/4*wr)[:, None]/len(theta), (len(phi), len(theta)))
            return points, weights.ravel().copy()
        angles = [0., 2*np.pi]
        for a, b in zip(poly, np.roll(poly, -1, axis=0)):
            for p in (a, b):
                if np.linalg.norm(p) > 1e-18:
                    angles.append(float(np.arctan2(p[1], p[0]) % (2*np.pi)))
            edge = b-a
            aa = edge @ edge
            if aa > 0:
                bb, cc = 2*(a @ edge), a @ a-radius*radius
                disc = bb*bb-4*aa*cc
                if disc > 0:
                    for t in ((-bb-np.sqrt(disc))/(2*aa), (-bb+np.sqrt(disc))/(2*aa)):
                        if 0 < t < 1:
                            p = a+t*edge
                            angles.append(float(np.arctan2(p[1], p[0]) % (2*np.pi)))
        angles = np.unique(np.round(angles, 14))
        ga, wa = _gauss(max(8, int(order)))
        gr, wr = _gauss(max(6, int(order)//2))
        pts, weights = [], []
        for left, right in zip(angles[:-1], angles[1:]):
            if right-left < 1e-14:
                continue
            # Circle/edge intersections produce endpoint square-root terms.
            # Quadratic flattening regularizes both ends without changing
            # the physical domain or contact radius.
            s = .5*(ga+1.)
            theta = left+(right-left)*np.sin(.5*np.pi*s)**2
            lo, hi = ShellDPRTransfer._radial_interval(poly, theta, radius)
            valid = hi > lo+radius*1e-14
            if not valid.any():
                continue
            theta, lo, hi = theta[valid], lo[valid], hi[valid]
            wtheta = (right-left)*np.pi/4*np.sin(np.pi*s[valid])*wa[valid]
            phi0, phi1 = np.arcsin(np.clip(lo/radius, 0, 1)), np.arcsin(np.clip(hi/radius, 0, 1))
            phi = .5*(phi0[:, None]+phi1[:, None])+.5*(phi1-phi0)[:, None]*gr
            r = radius*np.sin(phi)
            w = 3/(2*np.pi)*np.sin(phi)*np.cos(phi)**2*.5*(phi1-phi0)[:, None]*wr*wtheta[:, None]
            pts.append(np.stack((r*np.cos(theta[:, None]), r*np.sin(theta[:, None])), axis=-1).reshape(-1, 2))
            weights.append(w.ravel())
        if not pts:
            return np.empty((0, 2)), np.empty(0)
        return np.vstack(pts), np.concatenate(weights)

    def _transfer_once(self, contact: Contact, order=16, method="auto"):
        point = np.asarray(contact.point, dtype=float)
        force = np.asarray(contact.force, dtype=float)
        normal = np.array(contact.normal, dtype=float, copy=True)
        if point.shape != (3,) or force.shape != (3,) or normal.shape != (3,):
            raise ValueError("Contact point/force/normal must have three components")
        if not np.isfinite(np.r_[point, force, normal]).all() or np.linalg.norm(normal) == 0:
            raise ValueError("Contact data must be finite with a nonzero normal")
        if contact.side not in (-1, 1):
            raise ValueError("side must be +1 or -1")
        normal /= np.linalg.norm(normal)
        Fn = -float(force @ normal)
        load = np.zeros(self.ndof)
        if np.linalg.norm(force) == 0:
            return load, dict(valid=True, radius_m=0., normal_force=0., method="unloaded", faces=0, n_samples=0,
                              force_error=0., moment_error=0., a_over_h=0.)
        if Fn <= 0:
            raise ValueError("Nonzero contact force must have a compressive normal component")
        shear = force+Fn*normal
        if contact.friction is not None and np.linalg.norm(shear) > contact.friction*Fn*(1+1e-12):
            raise ValueError("Supplied traction exceeds the declared friction cone")
        radius = float(contact.radius) if contact.radius is not None else hertz_radius(Fn, self.E, self.nu, contact.curvatures, self.contact_modulus)
        if not np.isfinite(radius) or radius <= 0:
            raise ValueError("Physical radius must be positive and finite")
        plane = _basis(normal)
        direction = force/np.linalg.norm(force)
        face_points = self.faces[contact.side][0]
        hint = contact.face_hint
        hint_complete = False
        # A current collision material-face witness can bypass spatial search
        # only when this complete physical disk is on that same valid face.
        # This is not an unchecked prior-frame contact-slot cache.
        if hint is not None and 0 <= int(hint) < len(self.quads):
            hint = int(hint)
            hint_polygon = self._projected_polygon(face_points[hint], point, normal, direction, plane)
            N_hint, _, valid_hint = self._inverse(hint_polygon, [[0., 0.]])
            if valid_hint[0] and self._full_disk(hint_polygon, radius):
                hit = N_hint[0] @ face_points[hint]
                surface_normal = contact.side*(N_hint[0] @ self.directors[self.quads[hint]])
                local_bound = radius+self.faces[contact.side][2][hint]
                hint_complete = bool(np.linalg.norm(hit-point) < local_bound and surface_normal @ normal > .25)
        faces = np.array([hint], dtype=int) if hint_complete else self._candidates(point, radius, contact.side)
        polygons = {int(f): self._projected_polygon(face_points[f], point, normal, direction, plane) for f in faces}
        # Seed from the closest force-line intersection, then retain only its local connected sheet.
        seeds = []
        hint = contact.face_hint
        if hint is not None and int(hint) in polygons:
            N, _, valid = self._inverse(polygons[int(hint)], [[0., 0.]])
            if valid[0]:
                hit = N[0] @ face_points[int(hint)]
                surface_normal = contact.side*(N[0] @ self.directors[self.quads[int(hint)]])
                local_bound = radius+self.faces[contact.side][2][int(hint)]
                if np.linalg.norm(hit-point) < local_bound and surface_normal @ normal > .25:
                    seeds.append((np.linalg.norm(hit-point), int(hint)))
        if not seeds:
            for f in faces:
                N, _, valid = self._inverse(polygons[int(f)], [[0., 0.]])
                if valid[0]:
                    hit = N[0] @ face_points[f]
                    seeds.append((np.linalg.norm(hit-point), int(f)))
        if not seeds:
            raise ValueError("No local offset-surface intersection on the original force line")
        seed = min(seeds)[1]
        local, todo = {seed}, [seed]
        candidate_set = set(map(int, faces))
        while todo:
            f = todo.pop()
            for g in self.adjacent[f] & candidate_set-local:
                local.add(g)
                todo.append(g)
        node_weights = np.zeros(len(self.vertices))
        count = 0
        selected = []
        used_method = "face_polar"
        embedded_error = None
        if method in ("auto", "analytic", "taylor", "fast"):
            w = self._analytic_weights(polygons[seed], radius)
            if w is None and method == "fast":
                w, embedded_error = self._embedded_weights(polygons[seed], radius)
                if w is not None:
                    used_method = "embedded_disk_cubature"
                    count = 52
            if w is None and method == "taylor":
                w = self._taylor_weights(polygons[seed], radius)
                if w is not None:
                    used_method = "taylor_complete_disk"
            if w is not None:
                np.add.at(node_weights, self.quads[seed], w)
                if used_method not in ("taylor_complete_disk", "embedded_disk_cubature"):
                    used_method = "analytic_affine_disk"
                selected = [seed]
        if not selected:
            if method == "analytic":
                raise ValueError("Analytic affine complete-disk path is not applicable")
            for f in sorted(local):
                poly = polygons[f]
                samples, weights = self._face_polar_samples(poly, radius, order)
                if not len(weights):
                    continue
                N, _, inside = self._inverse(poly, samples)
                if not inside.all():
                    raise ContactIntegrationError("Projected Q4 inversion failed inside the clipped patch")
                np.add.at(node_weights, self.quads[f], weights @ N)
                count += len(weights)
                selected.append(f)
        mass = node_weights.sum()
        if abs(mass-1) > 1e-9:
            raise ContactIntegrationError(f"Finite footprint not resolved/covered: normalized mass {mass:.16g}")
        # Correct only roundoff/quadrature-level resultant drift, never discard uncovered loads.
        raw_mass = mass
        node_weights /= mass
        load = load.reshape(-1, 6)
        load[:, :3] = node_weights[:, None]*force
        load[:, 3:] = node_weights[:, None]*(contact.side*.5*self.h*np.cross(self.directors, force))
        resultant = load[:, :3].sum(axis=0)
        moment = np.cross(self.vertices, load[:, :3]).sum(axis=0)+load[:, 3:].sum(axis=0)
        force_error = np.linalg.norm(resultant-force)/np.linalg.norm(force)
        moment_error = np.linalg.norm(moment-np.cross(point, force))/(np.linalg.norm(force)*radius)
        info = dict(valid=bool(force_error < 1e-10 and moment_error < 1e-8), radius_m=radius, normal_force=Fn,
                    a_over_h=radius/self.h, method=used_method, faces=len(selected), face_ids=selected,
                    n_samples=count, force_error=float(force_error), moment_error=float(moment_error),
                    raw_mass_error=float(raw_mass-1), min_nodal_weight=float(node_weights.min()),
                    embedded_relative_error=embedded_error, witness_complete_disk=hint_complete)
        if not info["valid"]:
            raise ContactIntegrationError(f"Contact wrench preservation failed: {info}")
        return load.ravel(), info

    def transfer(self, contact: Contact, order=16, method="auto", adaptive=True, max_order=64):
        """Transfer a footprint, refining only failed numerical integration.

        No force/radius/material/gate is changed on retry. All declined attempts
        are returned in the record. Unsupported contact geometry is not retried
        as a different load model. Set adaptive=False to reproduce a given rule.
        """
        requested = int(order)
        errors = []
        while True:
            try:
                force, info = self._transfer_once(contact, order=order, method=method)
                info.update(requested_order=requested, quadrature_order=int(order),
                            adaptive_attempts=len(errors), retained_failures=errors)
                return force, info
            except ContactIntegrationError as exc:
                errors.append(dict(order=int(order), message=str(exc)))
                if not adaptive or order >= max_order:
                    raise ContactIntegrationError(f"{exc}; integration attempts: {errors}") from exc
                order = min(2*int(order), int(max_order))

    def transfer_many(self, contacts, **kwargs):
        total = np.zeros(self.ndof)
        records = []
        for contact in contacts:
            force, record = self.transfer(contact, **kwargs)
            total += force
            records.append(record)
        return total, records

