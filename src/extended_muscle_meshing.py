"""
Extended muscle meshing with rotation-minimizing (parallel transport) frames.

Pipeline per muscle:
    1. resample_centerline        smooth + equal arc length, end points kept in place
    2. parallel_transport_frames  one frame per node, no twist; shared by measuring and meshing
    3. radial_profiles_from_mask  first-exit ray marching in the (smoothed) muscle mask
    4. smooth_radii               fill gaps, smooth along the muscle and around each ring
    5. clamp_radii_by_curvature   avoid ring self-intersections on the inside of bends
    6. generate_extended_muscle_mesh
                                  body rings from measured radii, then a blend that morphs the
                                  tail cross-section into the eye contact ring (same angles,
                                  same frame), closed with caps

Coordinates: centerlines, masks and radii are in voxels of the isotropic grid.
Eye contact ring / contact centre come in mm (mesh coordinates, = voxel * pixel_spacing).
Returned meshes are in mm.
"""

import numpy
import trimesh
from scipy.interpolate import splprep, splev
from scipy.ndimage import gaussian_filter, map_coordinates, median_filter, uniform_filter1d


# -----------------------------------------------------------------------------
# Centerline and frames
# -----------------------------------------------------------------------------
def resample_centerline(centerline, step=1.0, smooth=0.25, end_weight=50.0):
    """
    Smooth with a spline and resample at equal arc length (step in voxels).
    The first and last point get a high weight so they stay (almost) where they are:
    the last point is the touchdown on the eye and must not drift.
    """
    centerline = numpy.asarray(centerline, dtype=float)
    d = numpy.linalg.norm(numpy.diff(centerline, axis=0), axis=1)
    cl = centerline[numpy.concatenate([[True], d > 1e-6])]          # drop duplicates
    if len(cl) < 4:
        raise ValueError("Centerline needs at least 4 distinct points")

    w = numpy.ones(len(cl))
    w[0] = w[-1] = end_weight
    tck, _ = splprep(cl.T, w=w, s=smooth * len(cl))

    u_dense = numpy.linspace(0, 1, 4000)
    pts = numpy.array(splev(u_dense, tck)).T
    arc = numpy.concatenate([[0], numpy.cumsum(numpy.linalg.norm(numpy.diff(pts, axis=0), axis=1))])

    n = max(int(round(arc[-1] / step)) + 1, 4)
    u_new = numpy.interp(numpy.linspace(0, arc[-1], n), arc, u_dense)
    out = numpy.array(splev(u_new, tck)).T
    out[0], out[-1] = cl[0], cl[-1]                                   # pin ends exactly
    return out


def parallel_transport_frames(centerline, v1_start=None):
    """Rotation-minimizing frames. Returns tangents, v1, v2 (each N x 3), right-handed (v1, v2, t)."""
    t = numpy.gradient(centerline, axis=0)
    t /= numpy.linalg.norm(t, axis=1, keepdims=True)

    if v1_start is None:
        ref = numpy.array([1.0, 0.0, 0.0]) if abs(t[0, 0]) < 0.9 else numpy.array([0.0, 1.0, 0.0])
        v1_start = numpy.cross(t[0], ref)
    v1 = numpy.zeros_like(t)
    v = v1_start - (v1_start @ t[0]) * t[0]
    v1[0] = v / numpy.linalg.norm(v)

    for i in range(1, len(t)):
        v = v1[i - 1] - (v1[i - 1] @ t[i]) * t[i]
        v1[i] = v / numpy.linalg.norm(v)

    v2 = numpy.cross(t, v1)
    return t, v1, v2


def theta_steps(n_theta):
    """The one angle convention used everywhere (measuring, blending, meshing)."""
    return numpy.linspace(-numpy.pi, numpy.pi, n_theta, endpoint=False)


# -----------------------------------------------------------------------------
# Measuring the cross-sections
# -----------------------------------------------------------------------------
def radial_profiles_from_mask(centerline, mask, v1, v2, n_theta=40, r_max=40.0, dr=0.25, sigma=1.0):
    """
    March outward along each ray and stop at the FIRST exit from the muscle.
    The mask is Gaussian-smoothed and the 0.5 crossing is interpolated, so radii are sub-voxel
    and free of staircase steps. Nodes outside the muscle give a row of NaN.
    Returns radii (N x n_theta) in voxels.
    """
    field = gaussian_filter(mask.astype(numpy.float32), sigma)
    th = theta_steps(n_theta)
    r = numpy.arange(0.0, r_max + dr, dr)

    dirs = numpy.cos(th)[None, :, None] * v1[:, None, :] + numpy.sin(th)[None, :, None] * v2[:, None, :]
    pts = centerline[:, None, None, :] + r[None, None, :, None] * dirs[:, :, None, :]
    vals = map_coordinates(field, pts.reshape(-1, 3).T, order=1, mode="constant", cval=0.0)
    vals = vals.reshape(len(centerline), n_theta, len(r))

    below = vals < 0.5
    has_exit = below.any(axis=2)
    i1 = numpy.argmax(below, axis=2)
    i0 = numpy.maximum(i1 - 1, 0)
    f0 = numpy.take_along_axis(vals, i0[..., None], axis=2)[..., 0]
    f1 = numpy.take_along_axis(vals, i1[..., None], axis=2)[..., 0]
    frac = numpy.clip((f0 - 0.5) / numpy.maximum(f0 - f1, 1e-9), 0.0, 1.0)
    radii = r[i0] + frac * dr

    node_inside = vals[:, :, 0] >= 0.5                     # same value for all angles of a node
    radii[~node_inside | ~has_exit] = numpy.nan
    return radii


def smooth_radii(radii, long_size=5, ring_size=5):
    """Fill NaN gaps, then smooth along the muscle (per angle) and around each ring (cyclic)."""
    out = numpy.array(radii, dtype=float)
    n, n_theta = out.shape
    idx = numpy.arange(n)

    for j in range(n_theta):                               # longitudinal gap filling
        col = out[:, j]
        ok = ~numpy.isnan(col)
        out[:, j] = numpy.interp(idx, idx[ok], col[ok]) if ok.any() else 1.0

    for i in range(n):                                     # in-ring gap filling (rare after the above)
        row = out[i]
        ok = ~numpy.isnan(row)
        if not ok.all():
            row[~ok] = numpy.interp(numpy.flatnonzero(~ok), numpy.flatnonzero(ok), row[ok], period=n_theta)

    out = uniform_filter1d(out, size=long_size, axis=0, mode="nearest")
    out = median_filter(out, size=(1, ring_size), mode="wrap")
    out = uniform_filter1d(out, size=ring_size, axis=1, mode="wrap")
    return numpy.maximum(out, 0.1)


def clamp_radii_by_curvature(centerline, radii, v1, v2, safety=0.8):
    """Limit radii on the inside of bends so neighbouring rings cannot cross."""
    t = numpy.gradient(centerline, axis=0)
    ds = numpy.linalg.norm(t, axis=1)
    t /= ds[:, None]
    dt = numpy.gradient(t, axis=0) / ds[:, None]
    kappa = numpy.linalg.norm(dt, axis=1)
    normal = dt / numpy.maximum(kappa, 1e-12)[:, None]

    th = theta_steps(radii.shape[1])
    out = radii.copy()
    for i in range(len(centerline)):
        if kappa[i] < 1e-9:
            continue
        rays = numpy.cos(th)[:, None] * v1[i] + numpy.sin(th)[:, None] * v2[i]
        toward_inside = numpy.maximum(rays @ normal[i], 1e-6)
        out[i] = numpy.minimum(out[i], safety / (kappa[i] * toward_inside))
    return out


def extrapolate_profile(rows, n_out, min_frac=0.3, max_frac=1.0):
    """
    Continue the per-angle linear trend of `rows` (W x n_theta, ordered towards the junction,
    last row = junction) for n_out further nodes. Output row 0 is the node right after the junction.
    Radii are clamped to [min_frac, max_frac] x the median profile of `rows`, so a steep trend
    cannot run away to zero or blow up.
    """
    rows = numpy.asarray(rows, float)
    w = len(rows)
    x = numpy.arange(w)
    xm = x.mean()
    slope = ((x - xm)[:, None] * (rows - rows.mean(axis=0))).sum(axis=0) / ((x - xm) ** 2).sum()
    intercept = rows.mean(axis=0) - slope * xm
    x_out = (w - 1) + numpy.arange(1, n_out + 1)
    out = intercept[None, :] + slope[None, :] * x_out[:, None]
    ref = numpy.median(rows, axis=0)
    return numpy.clip(out, min_frac * ref, max_frac * ref)


def taper_scale(s, shape="dome"):
    """s: 0 at the tip, 1 at full size. 'dome' = round cap, 'cone' = pointed, 'ogive' = in between."""
    s = numpy.clip(s, 0.0, 1.0)
    if shape == "dome":
        return numpy.sqrt(s * (2.0 - s))
    if shape == "cone":
        return s
    if shape == "ogive":
        return numpy.sqrt(s)
    raise ValueError(f"Unknown taper shape '{shape}'")


# -----------------------------------------------------------------------------
# Eye contact ring in the mesh frame
# -----------------------------------------------------------------------------
def order_ring_in_frame(ring_pts, center, v1, v2, n_theta):
    """
    Resample a closed 3D ring (any order) at the standard angles of frame (v1, v2) around `center`.
    Returns n_theta points whose index j sits at angle theta_steps[j]; this makes ring vertex j
    line up with vertex j of the tube rings, so the last band of triangles does not twist.
    """
    rel = ring_pts - center
    ang = numpy.arctan2(rel @ v2, rel @ v1)
    order = numpy.argsort(ang)
    ang, pts = ang[order], ring_pts[order]
    keep = numpy.concatenate([[True], numpy.diff(ang) > 1e-9])    # interp needs increasing x
    ang, pts = ang[keep], pts[keep]

    th = theta_steps(n_theta)
    return numpy.stack([numpy.interp(th, ang, pts[:, k], period=2 * numpy.pi) for k in range(3)], axis=1)


# -----------------------------------------------------------------------------
# Mesh
# -----------------------------------------------------------------------------
def _hermite_path(p0, t0, p1, t1, step, tangent_scale=1.0):
    """Cubic Hermite curve p0 -> p1 with end tangents t0, t1, resampled at equal arc length."""
    L = numpy.linalg.norm(p1 - p0)
    m0, m1 = tangent_scale * L * t0, tangent_scale * L * t1
    s = numpy.linspace(0, 1, 2000)[:, None]
    h00, h10 = 2 * s**3 - 3 * s**2 + 1, s**3 - 2 * s**2 + s
    h01, h11 = -2 * s**3 + 3 * s**2, s**3 - s**2
    pts = h00 * p0 + h10 * m0 + h01 * p1 + h11 * m1
    arc = numpy.concatenate([[0], numpy.cumsum(numpy.linalg.norm(numpy.diff(pts, axis=0), axis=1))])
    n = max(int(round(arc[-1] / step)) + 1, 3)
    out = numpy.stack([numpy.interp(numpy.linspace(0, arc[-1], n), arc, pts[:, k]) for k in range(3)], axis=1)
    out[0], out[-1] = p0, p1
    return out


def generate_extended_muscle_mesh(
    centerline, v1, v2, body_radii, eye_ring_mm, contact_center_mm, pixel_spacing,
    taper_start_len=12, blend_power=2.0, tail_window=5, taper_shape="dome",
    tail_extrapolate=True, tail_fit_window=15, tail_end_scale=1.0, tail_power=1.0,
    tail_min_frac=0.3, tail_max_frac=1.0,
    reroute_transition=False, entry_angle_deg=60.0, eye_normal=None, tangent_scale=1.3,
    blend_length_mm=2.0,
):
    """
    centerline : (N, 3) voxel nodes, from the muscle origin to the touchdown on the eye (last node)
    v1, v2     : (N, 3) parallel transport frames of that centerline
    body_radii : (M, n_theta) smoothed radii for the first M nodes (the measured muscle body);
                 nodes M..N-1 form the transition to the eye
    eye_ring_mm, contact_center_mm : contact ring points and centre on the eyeball, in mm

    Transition (all blending happens in each node's own frame, so rings stay perpendicular to the
    centerline and keep their size; only the eye curvature is added out of plane towards the end):
        blend_length_mm    : the tube keeps the muscle's own (tail) cross-section along the centerline
                             and morphs into the eye ring only over this final length. The eye ring is
                             inclined to the tube axis where the centerline meets the eye at a shallow
                             angle, so a long morph would make the whole transition thin.
                             None = morph over the entire transition (old behaviour).
        reroute_transition : False (default) = follow the extended centerline.
                             True = replace the transition by a smooth Hermite curve from the end of
                             the body to the contact centre (ignores the centerline's path there)
        entry_angle_deg    : arrival direction relative to the eye surface at the contact:
                             0 = tangential (lies on the sclera, gentlest path for a muscle wrapping
                             the globe), 90 = perpendicular (last ring lies flat in the patch plane,
                             but needs a sharp turn if the muscle wraps around the eye)
        eye_normal         : OUTWARD surface normal of the eyeball at the contact (any length), e.g. the
                             eyeball mesh vertex normal. Strongly recommended; without it the normal
                             is fitted to the eye ring and its sign guessed from the centerline.
        tangent_scale      : Hermite tangent length relative to the chord; larger = rounder,
                             wider turn, smaller = tighter turn near the ends
        blend_power        : shape of the morph within the blend window (higher = later)
    Tail profile along the transition (before blending into the eye ring):
        tail_extrapolate : continue the per-angle radius trend of the last `tail_fit_window` body
                           rings (False = constant median profile of the last `tail_window` rings)
        tail_end_scale, tail_power : extra narrowing on top:
                           scale = 1 - (1 - tail_end_scale) * s ** tail_power, s = 0..1 along the transition
        tail_min_frac, tail_max_frac : clamp of the extrapolated radii relative to the tail profile
    taper_shape : origin tip shape, 'dome' (round), 'ogive', or 'cone' (pointed)

    Returns (mesh_in_mm, eye_ring_vertex_indices). The last ring is exactly the eye ring.
    """
    centerline = numpy.asarray(centerline, float)
    v1, v2 = numpy.asarray(v1, float), numpy.asarray(v2, float)
    m_body, n_theta = body_radii.shape
    if m_body >= len(centerline):
        raise ValueError("Need at least one transition node after the body")
    th = theta_steps(n_theta)
    cos_t, sin_t = numpy.cos(th), numpy.sin(th)

    eye_ring = numpy.asarray(eye_ring_mm, float) / pixel_spacing
    contact = numpy.asarray(contact_center_mm, float) / pixel_spacing

    # ---- Transition path and frames ---------------------------------------------------------
    if reroute_transition:
        p0 = centerline[m_body - 1]
        t0 = numpy.cross(v1[m_body - 1], v2[m_body - 1])

        if eye_normal is not None:
            n_in = -numpy.asarray(eye_normal, float)
            n_in /= numpy.linalg.norm(n_in)
        else:
            approach = centerline[-1] - centerline[-2]
            _, _, vh = numpy.linalg.svd(eye_ring - eye_ring.mean(axis=0))
            n_in = vh[-1] if vh[-1] @ approach > 0 else -vh[-1]
            print("   Note: no eye_normal given; inward normal guessed from the centerline approach")

        travel = contact - p0                                     # overall direction of travel
        tangential = travel - (travel @ n_in) * n_in              # its part along the eye surface
        if numpy.linalg.norm(tangential) < 1e-9:
            tangential = t0 - (t0 @ n_in) * n_in
        tangential /= numpy.linalg.norm(tangential)
        alpha = numpy.radians(entry_angle_deg)
        t1 = numpy.cos(alpha) * tangential + numpy.sin(alpha) * n_in
        t1 /= numpy.linalg.norm(t1)

        step = numpy.median(numpy.linalg.norm(numpy.diff(centerline[:m_body], axis=0), axis=1))
        path = _hermite_path(p0, t0, contact, t1, step, tangent_scale)
        _, pv1, pv2 = parallel_transport_frames(path, v1_start=v1[m_body - 1])
        centerline = numpy.vstack([centerline[:m_body], path[1:]])
        v1 = numpy.vstack([v1[:m_body], pv1[1:]])
        v2 = numpy.vstack([v2[:m_body], pv2[1:]])
        end_shift = numpy.zeros(3)                                 # path already ends at the contact
    else:
        end_shift = contact - centerline[-1]

    n_nodes = len(centerline)
    n_trans = n_nodes - m_body
    t_all = numpy.cross(v1, v2)

    # ---- Tail profile along the transition ------------------------------------------------------
    if tail_extrapolate:
        rows = body_radii[max(m_body - tail_fit_window, 0):m_body]
        r_trans = extrapolate_profile(rows, n_trans, tail_min_frac, tail_max_frac)
    else:
        r_tail = numpy.median(body_radii[max(m_body - tail_window, 0):m_body], axis=0)
        r_trans = numpy.repeat(r_tail[None, :], n_trans, axis=0)
    s_tr = numpy.arange(1, n_trans + 1) / n_trans
    r_trans = r_trans * (1.0 - (1.0 - tail_end_scale) * s_tr ** tail_power)[:, None]

    # ---- Eye ring in local coordinates of the last frame -------------------------------------
    eye_ring = order_ring_in_frame(eye_ring, contact, v1[-1], v2[-1], n_theta)
    eye_off = eye_ring - contact
    ex, ey, ez = eye_off @ v1[-1], eye_off @ v2[-1], eye_off @ t_all[-1]

    rings = []

    # Body: measured radii, taper at the origin
    start_point = centerline[0]
    for i in range(m_body):
        p = centerline[i]
        r = body_radii[i].copy()
        if i < taper_start_len:
            s = (i + 1) / taper_start_len                         # first ring is small, not zero
            r *= taper_scale(s, taper_shape)
            p = (1.0 - s) * start_point + s * p
        r = numpy.maximum(r, 0.05)
        rings.append(p + r[:, None] * (cos_t[:, None] * v1[i] + sin_t[:, None] * v2[i]))

    # Transition: tail cross-section -> eye ring, blended in each node's own frame
    if blend_length_mm is None:
        n_blend = n_trans
    else:
        step = numpy.median(numpy.linalg.norm(numpy.diff(centerline, axis=0), axis=1))
        n_blend = int(numpy.clip(round(blend_length_mm / (step * pixel_spacing)), 1, n_trans))
    for k, i in enumerate(range(m_body, n_nodes), start=1):
        kb = k - (n_trans - n_blend)                              # position inside the blend window
        b = (kb / n_blend) ** blend_power if kb > 0 else 0.0
        centre = centerline[i] + b * end_shift
        x = (1.0 - b) * r_trans[k - 1] * cos_t + b * ex
        y = (1.0 - b) * r_trans[k - 1] * sin_t + b * ey
        z = b * ez                                                # eye curvature, out of plane
        rings.append(centre + x[:, None] * v1[i] + y[:, None] * v2[i] + z[:, None] * t_all[i])

    vertices = numpy.vstack(rings + [start_point[None, :], contact[None, :]])
    n_rings = len(rings)
    start_tip, end_tip = n_rings * n_theta, n_rings * n_theta + 1

    faces = []
    j = numpy.arange(n_theta)
    j2 = (j + 1) % n_theta
    faces += numpy.stack([numpy.full(n_theta, start_tip), j, j2], axis=1).tolist()   # same winding as the tube
    for i in range(n_rings - 1):
        b0, b1 = i * n_theta, (i + 1) * n_theta
        faces += numpy.stack([b0 + j, b1 + j, b1 + j2], axis=1).tolist()
        faces += numpy.stack([b0 + j, b1 + j2, b0 + j2], axis=1).tolist()
    last = (n_rings - 1) * n_theta
    faces += numpy.stack([last + j, numpy.full(n_theta, end_tip), last + j2], axis=1).tolist()

    # Curvature check on the transition: rings cross where the radius exceeds the bend radius
    tg = numpy.gradient(centerline, axis=0)
    ds = numpy.linalg.norm(tg, axis=1)
    kappa = numpy.linalg.norm(numpy.gradient(tg / ds[:, None], axis=0), axis=1) / ds
    r_bend = 1.0 / max(kappa[m_body:].max(), 1e-9)
    r_max = max(r_trans.max(), numpy.abs(numpy.r_[ex, ey]).max())
    if r_max > 0.9 * r_bend:
        print(f"   Warning: transition bend radius {r_bend * pixel_spacing:.2f} mm vs muscle radius "
              f"{r_max * pixel_spacing:.2f} mm; rings may cross. Increase tangent_scale or the "
              f"transition length.")

    mesh = trimesh.Trimesh(vertices=vertices * pixel_spacing, faces=numpy.asarray(faces), process=False)
    if mesh.volume < 0:
        mesh.invert()
    eye_ring_idx = numpy.arange(last, last + n_theta)
    return mesh, eye_ring_idx


def taubin_pinned(mesh, pinned, lamb=0.5, nu=-0.53, iterations=6):
    """Taubin smoothing that keeps `pinned` vertices (e.g. the eye contact ring) exactly in place."""
    L = trimesh.smoothing.laplacian_calculation(mesh, equal_weight=True)
    v = mesh.vertices.copy()
    free = numpy.ones(len(v), dtype=bool)
    free[numpy.asarray(pinned)] = False
    for _ in range(iterations):
        for f in (lamb, nu):
            dv = L @ v - v
            v[free] += f * dv[free]
    mesh.vertices = v
    return mesh


def mesh_report(mesh):
    print(f"   vertices={len(mesh.vertices)}, faces={len(mesh.faces)}, "
          f"watertight={mesh.is_watertight}, winding_ok={mesh.is_winding_consistent}, "
          f"volume={mesh.volume:.1f} mm^3")


# -----------------------------------------------------------------------------
# Convenience: centerline -> frames -> measured, smoothed body radii
# -----------------------------------------------------------------------------
def measure_muscle_along_centerline(trimmed_centerline, muscle_mask, n_theta=40, step=1.0,
                                    end_trim=6, min_transition=5,
                                    start_extension_mm=0.0, voxel_mm=None, start_trim=6,
                                    profile_window=5, extension_end_scale=1.0, extension_power=1.0,
                                    extension_extrapolate=True, fit_window=15,
                                    min_frac=0.3, max_frac=1.0):
    """
    Returns centerline, v1, v2 (from the muscle origin - or the end of the start extension - to the
    touchdown on the eye) and body_radii for the first M nodes.

    end_trim           : nodes dropped from the end of the measured body; the rounded end of the
                         segmentation would otherwise make the tail profile too thin (a "neck")
    min_transition     : minimum number of transition nodes before the eye ring
    start_extension_mm : length of a straight extension added before the muscle origin (0 = none)
    voxel_mm           : voxel size in mm, needed to convert start_extension_mm to voxels
    start_trim         : with an extension, the first measured nodes (rounded segmentation tip) get the
                         head profile instead of their own, thinner radii
    profile_window     : number of rings averaged (median) for the head profile
    extension_end_scale: cross-section size at the tip of the extension relative to the head profile
                         (1.0 = constant, 0.5 = half the radius at the tip)
    extension_power    : how the size changes along the extension, s = 0 at the muscle, 1 at the tip:
                         scale = 1 - (1 - extension_end_scale) * s ** extension_power
                         1 = linear (cone), > 1 = stays thick longer then narrows near the tip,
                         < 1 = narrows quickly right after the muscle
    extension_extrapolate : continue the per-angle radius trend of the first `fit_window` well-measured
                         rings into the extension (False = constant head profile)
    min_frac, max_frac : clamp of the extrapolated radii relative to the head profile
    """
    cl = resample_centerline(trimmed_centerline, step=step)
    _, v1, v2 = parallel_transport_frames(cl)
    radii = radial_profiles_from_mask(cl, muscle_mask, v1, v2, n_theta=n_theta)

    valid = ~numpy.all(numpy.isnan(radii), axis=1)
    if valid.sum() < 4:
        raise ValueError("Fewer than 4 centerline nodes lie inside the muscle mask")
    first, last = numpy.flatnonzero(valid)[[0, -1]]

    cl, v1, v2, radii = cl[first:], v1[first:], v2[first:], radii[first:]
    m_body = last - first + 1 - end_trim
    m_body = min(m_body, len(cl) - min_transition)
    if m_body < 4:
        raise ValueError("Measured muscle body too short after trimming")

    body = smooth_radii(radii[:m_body])
    body = clamp_radii_by_curvature(cl[:m_body], body, v1[:m_body], v2[:m_body])

    n_ext = 0
    if start_extension_mm > 0:
        if voxel_mm is None:
            raise ValueError("voxel_mm is required for start_extension_mm")
        k0 = min(start_trim, m_body - 1)
        n_ext = int(round(start_extension_mm / (voxel_mm * step)))

        if extension_extrapolate:
            rows = body[k0:k0 + fit_window][::-1]           # ordered towards the junction (node k0)
            extra = extrapolate_profile(rows, k0 + n_ext, min_frac, max_frac)
            body[:k0] = extra[:k0][::-1]                    # the thin segmentation tip follows the trend
            ext_profile = extra[k0:][::-1]                  # far tip first
        else:
            head = numpy.median(body[k0:k0 + profile_window], axis=0)
            body[:k0] = head
            ext_profile = numpy.repeat(head[None, :], n_ext, axis=0)
        t0 = numpy.cross(v1[0], v2[0])                     # tangent of the frame at the origin
        k = numpy.arange(n_ext, 0, -1)[:, None]            # far end first
        ext_pts = cl[0] - k * step * t0
        cl = numpy.vstack([ext_pts, cl])
        v1 = numpy.vstack([numpy.repeat(v1[:1], n_ext, axis=0), v1])
        v2 = numpy.vstack([numpy.repeat(v2[:1], n_ext, axis=0), v2])
        s_ext = k[:, 0] / n_ext                            # 1 at the far tip -> 1/n_ext next to the muscle
        scale = 1.0 - (1.0 - extension_end_scale) * s_ext ** extension_power
        ext_radii = scale[:, None] * ext_profile
        body = numpy.vstack([ext_radii, body])
        m_body += n_ext

    print(f"   nodes: {len(cl)} total, start extension {n_ext}, body {m_body - n_ext}, "
          f"transition {len(cl) - m_body} (skipped {first} before the muscle)")
    return cl, v1, v2, body


# -----------------------------------------------------------------------------
# Re-extract a clean surface from the ring tube (removes self-intersections)
# -----------------------------------------------------------------------------
def remesh_tube_as_solid(tube_mesh, n_theta, pixel_spacing, resolution_mm=0.15, smooth_sigma=1.0,
                         eye_mask=None, eye_clearance_mm=0.2, taubin_iterations=10, target_faces=None,
                         close_iterations=2, eye_surface=None):
    """
    Turn the ring tube from generate_extended_muscle_mesh into a solid and extract a new surface.

    Each band between two consecutive rings becomes the convex hull of both rings (the first and
    last band also include the cap tips); the muscle is the union of all bands, rasterised on a grid
    with `resolution_mm` spacing. Overlaps of the tube with itself are counted once, so the new
    surface cannot self-intersect. Marching cubes gives a watertight, manifold surface.

    tube_mesh      : mesh in mm, vertices = n_rings * n_theta ring vertices + start tip + end tip
    eye_mask       : optional eyeball mask on the voxel grid of the centerlines (voxel = mm / spacing);
                     everything inside the eye is removed, so the muscle ends exactly on the eye surface
    eye_clearance_mm : with eye_mask, also remove a layer of this thickness outside the eye mask, so the
                     smoothed surface stays outside the eyeball (the mask boundary is a voxel staircase).
                     With eye_surface, the same clearance from the eyeball mesh.
    eye_surface    : optional eyeball surface mesh (mm, closed). Everything inside it (plus the clearance)
                     is removed, i.e. the muscle is cut by the actual eyeball mesh rather than the mask.
    smooth_sigma   : Gaussian smoothing of the occupancy, in grid cells (removes voxel steps)
    target_faces   : optional quadric decimation to about this many faces (needs `fast_simplification`)
    close_iterations : morphological closing (in grid cells) that removes small tunnels and cavities
                     created where the tube overlaps itself; such handles make gmsh's surface
                     reparametrization fail ("Wrong topology of boundary mesh")
    Returns a trimesh.Trimesh in mm.
    """
    from scipy.spatial import Delaunay
    from scipy.spatial.qhull import QhullError
    from scipy.ndimage import label
    from skimage import measure

    V = numpy.asarray(tube_mesh.vertices, float)
    n_rings = (len(V) - 2) // n_theta
    rings = V[:n_rings * n_theta].reshape(n_rings, n_theta, 3)
    start_tip, end_tip = V[-2], V[-1]

    res = float(resolution_mm)
    pad = 3 * res + 3 * smooth_sigma * res
    lo = V.min(axis=0) - pad
    hi = V.max(axis=0) + pad
    shape = numpy.ceil((hi - lo) / res).astype(int) + 1
    occ = numpy.zeros(shape, dtype=bool)

    for i in range(n_rings - 1):
        pts = numpy.vstack([rings[i], rings[i + 1]])
        if i == 0:
            pts = numpy.vstack([pts, start_tip])
        if i == n_rings - 2:
            pts = numpy.vstack([pts, end_tip])
        try:
            hull = Delaunay(pts)
        except QhullError:
            hull = Delaunay(pts, qhull_options="QJ")          # joggle nearly flat bands
        i0 = numpy.maximum(numpy.floor((pts.min(axis=0) - lo) / res).astype(int), 0)
        i1 = numpy.minimum(numpy.ceil((pts.max(axis=0) - lo) / res).astype(int) + 1, shape)
        sub = numpy.stack(numpy.meshgrid(*[numpy.arange(a, b) for a, b in zip(i0, i1)], indexing="ij"), -1)
        coords = lo + sub.reshape(-1, 3) * res
        inside = (hull.find_simplex(coords) >= 0).reshape(sub.shape[:3])
        occ[i0[0]:i1[0], i0[1]:i1[1], i0[2]:i1[2]] |= inside

    if eye_mask is not None:
        from scipy.ndimage import distance_transform_edt
        # signed distance to the eye boundary in mm (negative inside), smooth enough to interpolate
        outside = distance_transform_edt(eye_mask == 0) * pixel_spacing
        inside = distance_transform_edt(eye_mask != 0) * pixel_spacing
        sdist = numpy.where(eye_mask != 0, -inside, outside) - 0.5 * pixel_spacing * numpy.sign(outside - inside)
        grid = numpy.stack(numpy.meshgrid(*[lo[k] + numpy.arange(shape[k]) * res for k in range(3)],
                                          indexing="ij"), -1).reshape(-1, 3)
        d = map_coordinates(sdist.astype(numpy.float32), (grid / pixel_spacing).T, order=1,
                            mode="constant", cval=1e3)
        occ &= (d >= eye_clearance_mm).reshape(shape)

    if eye_surface is not None:
        eye_m = eye_surface if isinstance(eye_surface, trimesh.Trimesh) else trimesh.load(eye_surface)
        lo_e, hi_e = eye_m.bounds[0] - eye_clearance_mm - res, eye_m.bounds[1] + eye_clearance_mm + res
        cand = numpy.argwhere(occ)
        pts = lo + cand * res
        near = numpy.all((pts >= lo_e) & (pts <= hi_e), axis=1)       # only test points near the eye
        sd = signed_distance_to_mesh(eye_m, pts[near])
        cut = cand[near][sd < eye_clearance_mm]
        occ[cut[:, 0], cut[:, 1], cut[:, 2]] = False
        eye_keep = occ.copy()                                          # closing must not refill the cut

    if close_iterations > 0:
        from scipy.ndimage import binary_closing, binary_fill_holes
        p_ = close_iterations + 1
        occ = numpy.pad(occ, p_)
        occ = binary_closing(occ, iterations=close_iterations)
        occ = binary_fill_holes(occ)[p_:-p_, p_:-p_, p_:-p_]
        if eye_mask is not None:                             # closing must not refill the eye cut
            occ &= (d >= eye_clearance_mm).reshape(shape)
        if eye_surface is not None:
            refilled = occ & ~eye_keep
            if refilled.any():
                rp = lo + numpy.argwhere(refilled) * res
                bad = numpy.argwhere(refilled)[signed_distance_to_mesh(eye_m, rp) < eye_clearance_mm]
                occ[bad[:, 0], bad[:, 1], bad[:, 2]] = False

    lab, n = label(occ)
    if n == 0:
        raise ValueError("Empty solid after rasterisation")
    if n > 1:                                                # keep the largest piece
        sizes = numpy.bincount(lab.ravel())[1:]
        occ = lab == (numpy.argmax(sizes) + 1)

    field = gaussian_filter(occ.astype(numpy.float32), smooth_sigma)
    verts, faces, _, _ = measure.marching_cubes(field, level=0.5, spacing=(res, res, res))
    mesh = trimesh.Trimesh(vertices=verts + lo, faces=faces, process=True)
    if mesh.volume < 0:
        mesh.invert()
    if taubin_iterations > 0:
        trimesh.smoothing.filter_taubin(mesh, iterations=taubin_iterations)
    if target_faces is not None and len(mesh.faces) > target_faces:
        mesh = mesh.simplify_quadric_decimation(face_count=int(target_faces))
        if mesh.volume < 0:
            mesh.invert()

    genus = surface_genus(mesh)
    if genus != 0:
        print(f"   Warning: surface has genus {genus} (tunnels/handles). Increase close_iterations "
              f"or check the tube for strong self-overlap.")
    return mesh


def surface_genus(mesh):
    """Genus of a closed surface (0 = sphere-like, 1 = one tunnel, ...); -1 if not a single closed body."""
    if not mesh.is_watertight or mesh.body_count != 1:
        return -1
    chi = len(mesh.vertices) - len(mesh.edges_unique) + len(mesh.faces)
    return (2 - chi) // 2


# -----------------------------------------------------------------------------
# Quality surface / volume meshing with gmsh
# -----------------------------------------------------------------------------
def gmsh_mesh_from_surface(
        surface, output_file, element_size=0.4, size_min=None,
        curvature_points=0, surface_angle=60.0, volume=True, report=True,
        retry_angles=(40.0, 30.0, 80.0), keep_surface_fallback=True, uniform=True,
        calibrate_size=True, algorithm_3d=1):
    """
    Re-mesh a clean, watertight surface with gmsh and optionally fill it with tets.

    surface         : trimesh.Trimesh or path to an STL (mm)
    output_file     : path without extension; writes <output_file>.stl (surface) and, with volume=True,
                      <output_file>.vtk / .msh (tetrahedra)
    element_size    : target edge length in mm (maximum)
    size_min        : smallest allowed edge length (default element_size / 2); only used with
                      curvature refinement
    curvature_points: elements per 2*pi of curvature (0 = uniform size, default). Around 6 refines thin
                      tips and tight bends; larger values over-refine because the small wrinkles of
                      a marching-cubes surface also count as curvature
    surface_angle   : feature angle for classifying the input into patches. 60 deg gave the fewest
                      short patch boundaries (and so the fewest undersized surface elements) on
                      marching-cubes surfaces in tests; larger values made no further difference
    retry_angles    : further angles tried if reparametrization fails ("Wrong topology of boundary mesh")
    keep_surface_fallback : if all angles fail, tet-mesh the input surface as it is (no surface
                      re-meshing); the surface triangles are then those of the input
    uniform         : True (default) = element size fixed at element_size everywhere: no curvature
                      refinement, no size propagation from patch boundaries (size_min and
                      curvature_points are ignored). False = allow local refinement.
    calibrate_size  : gmsh's 3D edges come out ~20 % longer than the size target. With True, the mesh
                      is generated once, the mean edge length measured, and the target rescaled so
                      the final mean edge length matches element_size (costs a second run)
    algorithm_3d    : gmsh 3D algorithm; 1 = Delaunay (default; best worst-element quality with uniform
                      sizing in tests), 4 = Frontal, 10 = HXT (fastest)
    Returns (surface_mesh_trimesh, quality dict). quality["mode"] tells which route succeeded.
    """
    import os, tempfile
    import gmsh

    tmp = None
    if isinstance(surface, trimesh.Trimesh):
        tmp = tempfile.NamedTemporaryFile(suffix=".stl", delete=False).name
        surface.export(tmp)
        stl_path = tmp
    else:
        stl_path = surface
    if size_min is None:
        size_min = element_size / 2.0

    def _set_options(size):
        if uniform:
            gmsh.option.setNumber("Mesh.MeshSizeMax", size)
            gmsh.option.setNumber("Mesh.MeshSizeMin", size)
            gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
            gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
            gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
        else:
            gmsh.option.setNumber("Mesh.MeshSizeMax", size)
            gmsh.option.setNumber("Mesh.MeshSizeMin", size_min * size / element_size)
            gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
            gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 1)
            gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", curvature_points)
        gmsh.option.setNumber("Mesh.Algorithm", 6)                # Frontal-Delaunay: well-shaped triangles
        gmsh.option.setNumber("Mesh.Algorithm3D", algorithm_3d)
        gmsh.option.setNumber("Mesh.Optimize", 1)
        gmsh.option.setNumber("Mesh.OptimizeNetgen", 1)
        gmsh.option.setNumber("Mesh.Smoothing", 10)

    def _attempt(angle, size, write=True):
        """angle = None -> keep the input surface triangles, only mesh the volume."""
        gmsh.initialize()
        try:
            gmsh.option.setNumber("General.Terminal", 0)
            gmsh.model.add("muscle")
            gmsh.merge(stl_path)
            if angle is not None:
                gmsh.model.mesh.classifySurfaces(numpy.radians(angle), True, True, numpy.pi)
                gmsh.model.mesh.createGeometry()
            gmsh.model.geo.synchronize()
            if volume or angle is None:
                tags = [e[1] for e in gmsh.model.getEntities(2)]
                loop = gmsh.model.geo.addSurfaceLoop(tags)
                gmsh.model.geo.addVolume([loop])
                gmsh.model.geo.synchronize()
            _set_options(size)
            gmsh.model.mesh.generate(3 if (volume or angle is None) else 2)


            quality = {"mode": "remeshed surface" if angle is not None else "input surface kept",
                       "angle": angle, "size_target_used": size}
            if report or calibrate_size:
                _, coords, _ = gmsh.model.mesh.getNodes()
                xyz = numpy.asarray(coords).reshape(-1, 3)
                ntags, _, _ = gmsh.model.mesh.getNodes()
                pos = {t: k for k, t in enumerate(ntags)}
                _, _, enodes = gmsh.model.mesh.getElements(3 if (volume or angle is None) else 2)
                if len(enodes):
                    nper = 4 if (volume or angle is None) else 3
                    el = numpy.vectorize(pos.get)(numpy.asarray(enodes[0]).reshape(-1, nper))
                    pairs = numpy.array([(a, b) for a in range(nper) for b in range(a + 1, nper)])
                    L = numpy.linalg.norm(xyz[el[:, pairs[:, 0]]] - xyz[el[:, pairs[:, 1]]], axis=2).ravel()
                    quality["edge_length"] = dict(mean=float(L.mean()), p5=float(numpy.percentile(L, 5)),
                                                  p95=float(numpy.percentile(L, 95)))
            if report:
                for dim, name in [(2, "triangles"), (3, "tets")]:
                    _, etags, _ = gmsh.model.mesh.getElements(dim)
                    if len(etags) and len(etags[0]):
                        q = numpy.asarray(gmsh.model.mesh.getElementQualities(etags[0], "gamma"))
                        quality[name] = dict(n=len(q), min=float(q.min()), mean=float(q.mean()),
                                             below_0_1=int((q < 0.1).sum()))
            if write:
                gmsh.write(output_file + ".stl")
                if volume:
                    gmsh.write(output_file + ".msh")
                    gmsh.write(output_file + ".vtk")
            return quality
        finally:
            gmsh.finalize()

    try:
        quality, errors = None, []
        def _run(angle):
            if not calibrate_size:
                return _attempt(angle, element_size)
            size = element_size
            for _ in range(3):                                    # converge mean tet edge to element_size
                trial = _attempt(angle, size, write=False)
                mean_edge = trial.get("edge_length", {}).get("mean")
                if not mean_edge or abs(mean_edge / element_size - 1.0) < 0.03:
                    break
                size *= element_size / mean_edge
            return _attempt(angle, size)

        for angle in [surface_angle, *retry_angles]:
            try:
                quality = _run(angle)
                break
            except Exception as e:
                errors.append(f"angle {angle}: {str(e).splitlines()[0][:80]}")
        if quality is None and keep_surface_fallback and volume:
            quality = _run(None)
        if quality is None:
            raise RuntimeError("gmsh failed for all settings:\n  " + "\n  ".join(errors))
    finally:
        if tmp is not None:
            os.remove(tmp)

    if report:
        note = f" (after: {'; '.join(errors)})" if errors else ""
        print(f"   gmsh: {quality['mode']}" + (f", angle {quality['angle']}" if quality["angle"] else "") + note)
        if "edge_length" in quality:
            e = quality["edge_length"]
            print(f"   edge length: mean {e['mean']:.3f} mm (target {element_size}), "
                  f"5-95 %: {e['p5']:.3f} - {e['p95']:.3f} mm"
                  + (f"  [gmsh size used: {quality['size_target_used']:.3f}]" if calibrate_size else ""))
        for name in ("triangles", "tets"):
            if name in quality:
                q = quality[name]
                print(f"   {name}: {q['n']}, gamma min {q['min']:.3f}, mean {q['mean']:.3f}, "
                      f"< 0.1: {q['below_0_1']}")

    surf = trimesh.load(output_file + ".stl", process=True)
    return surf, quality


# =============================================================================
# Optic nerve: reliable profile, flare into the eye, exact contact with the eyeball
# =============================================================================
def _signed_eye_distance(eye_mask, pixel_spacing):
    """Signed distance to the eye boundary in mm on the voxel grid (negative inside the eye)."""
    from scipy.ndimage import distance_transform_edt
    outside = distance_transform_edt(eye_mask == 0) * pixel_spacing
    inside = distance_transform_edt(eye_mask != 0) * pixel_spacing
    return (numpy.where(eye_mask != 0, -inside, outside)
            - 0.5 * pixel_spacing * numpy.sign(outside - inside)).astype(numpy.float32)


def build_ring_tube(centerline, v1, v2, radii, pixel_spacing, taper_start_len=0, taper_shape="dome"):
    """Closed ring tube (caps at both ends) from radii (N x n_theta) along a centerline; returns mesh in mm."""
    n, n_theta = radii.shape
    th = theta_steps(n_theta)
    cos_t, sin_t = numpy.cos(th), numpy.sin(th)
    rings = []
    for i in range(n):
        p, r = centerline[i], radii[i].copy()
        if i < taper_start_len:
            s = (i + 1) / taper_start_len
            r *= taper_scale(s, taper_shape)
            p = (1.0 - s) * centerline[0] + s * p
        rings.append(p + numpy.maximum(r, 0.05)[:, None] * (cos_t[:, None] * v1[i] + sin_t[:, None] * v2[i]))
    vertices = numpy.vstack(rings + [centerline[:1], centerline[-1:]])
    j = numpy.arange(n_theta)
    j2 = (j + 1) % n_theta
    faces = [numpy.stack([numpy.full(n_theta, n * n_theta), j, j2], 1)]
    for i in range(n - 1):
        b0, b1 = i * n_theta, (i + 1) * n_theta
        faces += [numpy.stack([b0 + j, b1 + j, b1 + j2], 1), numpy.stack([b0 + j, b1 + j2, b0 + j2], 1)]
    last = (n - 1) * n_theta
    faces.append(numpy.stack([last + j, numpy.full(n_theta, n * n_theta + 1), last + j2], 1))
    mesh = trimesh.Trimesh(vertices * pixel_spacing, numpy.vstack(faces), process=False)
    if mesh.volume < 0:
        mesh.invert()
    return mesh


def optic_nerve_tube(
        centerline, on_mask, eye_mask, pixel_spacing, n_theta=40, step=1.0,
        unreliable_mm=3.0, ref_window_mm=3.0,
        flare_length_mm=2.0, flare_amount_mm=0.8, flare_power=2.0, overlap_mm=1.5,
        taper_start_len=0, taper_shape="dome", target_point=None, tangent_scale=1.0):
    """
    Ring tube for the optic nerve that ignores the poorly segmented part near the eye, flares into the
    eyeball and overlaps it (the overlap is removed later by subtracting the eyeball mesh).

    centerline      : ON centerline in voxels (either direction; it is oriented towards the eye)
    unreliable_mm   : along this last stretch before the eye the measured radii are ignored ...
    ref_window_mm   : ... and replaced by the median cross-section of this window just before it
    flare_length_mm : the flare starts this far (along the centerline) from the eye surface
    flare_amount_mm : radius increase at the eye surface
    flare_power     : flare profile  r += flare_amount * u ** flare_power, u = 0..1 over the flare length.
                      >= 2 leaves the nerve smoothly; the surface meets the eye at a finite contact angle
                      (printed). A tangential fillet (angle -> 0) would leave a zero-thickness wedge at the
                      rim that cannot be meshed with good elements.
    overlap_mm      : how far the tube continues into the eye (must be > 0 so the boolean cut is clean)
    target_point    : optional point in voxels, typically the eyeball centre. The centerline is kept up
                      to `unreliable_mm` before the eye; from there a smooth (Hermite) curve leaves along
                      the nerve's direction and heads for the target, so the nerve reaches the eye pointing
                      towards its centre, i.e. roughly perpendicular to the surface. Without it, the nerve
                      is extended straight along the direction of the last centerline points.
    tangent_scale   : Hermite tangent length relative to the chord (larger = keeps the nerve direction
                      longer before turning towards the target)
    Returns (tube_mesh_mm, info dict).
    """
    sd = _signed_eye_distance(eye_mask, pixel_spacing)
    def eye_dist(p):
        return map_coordinates(sd, numpy.asarray(p, float).T, order=1, mode="nearest")

    cl = resample_centerline(centerline, step=step)
    if eye_dist(cl[[0]])[0] < eye_dist(cl[[-1]])[0]:          # orient: last point closest to the eye
        cl = cl[::-1].copy()

    d = eye_dist(cl)
    if target_point is not None:
        target = numpy.asarray(target_point, float)
        reliable = numpy.flatnonzero(d >= unreliable_mm)
        if len(reliable) < 4:
            raise ValueError("Centerline has fewer than 4 nodes outside the unreliable zone")
        k_rel = reliable[-1]
        k_back = max(k_rel - int(round(2.0 / (step * pixel_spacing))), 0)   # direction over ~2 mm
        t0 = cl[k_rel] - cl[k_back]
        t0 /= numpy.linalg.norm(t0)
        t1 = target - cl[k_rel]
        t1 /= numpy.linalg.norm(t1)
        path = _hermite_path(cl[k_rel], t0, target, t1, step, tangent_scale)
        cl = numpy.vstack([cl[:k_rel], path])
        d = eye_dist(cl)
    inside = numpy.flatnonzero(d < 0)
    if len(inside):                                            # centerline already enters the eye
        cl = cl[:inside[0] + 1]
    t_end = cl[-1] - cl[-min(6, len(cl))]
    t_end /= numpy.linalg.norm(t_end)
    ext = []
    p = cl[-1].copy()
    for _ in range(int(60 / (step * pixel_spacing))):         # extend straight until overlap is reached
        p = p + step * t_end
        ext.append(p.copy())
        if eye_dist(p[None])[0] < -overlap_mm:
            break
    cl = numpy.vstack([cl, ext])

    _, v1, v2 = parallel_transport_frames(cl)
    d = eye_dist(cl)
    k0 = int(numpy.argmax(d < 0))                              # first node inside the eye
    seg = numpy.linalg.norm(numpy.diff(cl, axis=0), axis=1) * pixel_spacing
    arc = numpy.concatenate([[0], numpy.cumsum(seg)])
    s = arc[k0] - arc                                          # mm before the eye surface (<0 inside)
    if k0 > 0:                                                 # interpolate the exact crossing
        f = d[k0 - 1] / (d[k0 - 1] - d[k0])
        s -= (1 - f) * seg[k0 - 1]

    radii = radial_profiles_from_mask(cl, on_mask, v1, v2, n_theta=n_theta)
    radii[s < unreliable_mm] = numpy.nan
    valid = ~numpy.all(numpy.isnan(radii), axis=1)
    if valid.sum() < 4:
        raise ValueError("Too few reliable ON cross-sections; reduce unreliable_mm or check the mask")
    first = numpy.flatnonzero(valid)[0]
    cl, v1, v2, radii, s = cl[first:], v1[first:], v2[first:], radii[first:], s[first:]

    ref_sel = (s >= unreliable_mm) & (s < unreliable_mm + ref_window_mm) & ~numpy.all(numpy.isnan(radii), 1)
    if not ref_sel.any():
        raise ValueError("No reliable cross-sections in the reference window")
    ref = numpy.nanmedian(radii[ref_sel], axis=0)
    last_valid = numpy.flatnonzero(~numpy.all(numpy.isnan(radii), axis=1))[-1]
    radii[last_valid + 1:] = ref                               # hold the reliable cross-section
    radii = smooth_radii(radii)

    u = numpy.clip(1.0 - s / flare_length_mm, 0.0, 1.0)       # 0 before the flare, 1 at the eye
    flare = flare_amount_mm * u ** flare_power / pixel_spacing
    radii = radii + flare[:, None]

    radii = clamp_radii_by_curvature(cl, radii, v1, v2)
    tube = build_ring_tube(cl, v1, v2, radii, pixel_spacing, taper_start_len, taper_shape)
    info = dict(n_nodes=len(cl), ref_radius_mm=float(numpy.mean(ref) * pixel_spacing),
                contact_radius_mm=float(numpy.mean(radii[s <= 0][0]) * pixel_spacing) if (s <= 0).any() else None,
                extension_nodes=len(ext))
    k_in = int(numpy.argmax(s <= 0)) if (s <= 0).any() else len(cl) - 1
    t_arr = cl[min(k_in + 1, len(cl) - 1)] - cl[max(k_in - 1, 0)]
    t_arr /= numpy.linalg.norm(t_arr)
    info["arrival_point_vox"] = cl[k_in]
    info["arrival_direction"] = t_arr
    slope = flare_amount_mm * flare_power / flare_length_mm          # dr/ds at the eye
    info["contact_angle_deg"] = float(90.0 - numpy.degrees(numpy.arctan(slope)))
    print(f"   ON: {len(cl)} nodes ({len(ext)} added towards/into the eye), reliable radius "
          f"{info['ref_radius_mm']:.2f} mm, at the eye {info['contact_radius_mm']:.2f} mm, "
          f"contact angle ~{info['contact_angle_deg']:.0f} deg")
    return tube, info


def cut_by_eyeball(solid_mm, eyeball_mesh):
    """Subtract the eyeball (exact mesh boolean). The contact face becomes part of the eyeball surface."""
    eye = eyeball_mesh if isinstance(eyeball_mesh, trimesh.Trimesh) else trimesh.load(eyeball_mesh)
    if eye.volume < 0:
        eye = eye.copy()
        eye.invert()
    out = trimesh.boolean.difference([solid_mm, eye], engine="manifold")
    if out.body_count > 1:                                    # keep the main piece
        out = max(out.split(only_watertight=False), key=lambda m: abs(m.volume))
    return out


def closest_points_on_mesh(mesh, points, k=16, return_faces=False):
    """Closest points on a triangle mesh without rtree: candidate faces by centroid k-NN, exact distance."""
    from scipy.spatial import cKDTree
    from trimesh.triangles import closest_point as tri_closest
    points = numpy.asarray(points, float)
    k = min(k, len(mesh.faces))
    _, cand = cKDTree(mesh.triangles_center).query(points, k=k)
    cand = cand.reshape(len(points), k)
    cp = numpy.empty((len(points), k, 3))
    for start in range(0, len(points), 200000):                       # chunks keep memory bounded
        sl = slice(start, start + 200000)
        tris = mesh.triangles[cand[sl].ravel()]
        cp[sl] = tri_closest(tris, numpy.repeat(points[sl], k, axis=0)).reshape(-1, k, 3)
    d = numpy.linalg.norm(cp - points[:, None, :], axis=2)
    best = numpy.argmin(d, axis=1)
    idx = numpy.arange(len(points))
    if return_faces:
        return cp[idx, best], d[idx, best], cand[idx, best]
    return cp[idx, best], d[idx, best]


def signed_distance_to_mesh(mesh, points):
    """Signed distance (mm) to a closed, outward-oriented surface: negative inside, positive outside."""
    if mesh.volume < 0:
        mesh = mesh.copy()
        mesh.invert()
    closest, dist, faces = closest_points_on_mesh(mesh, points, return_faces=True)
    sign = numpy.sign(numpy.einsum("ij,ij->i", numpy.asarray(points, float) - closest, mesh.face_normals[faces]))
    sign[sign == 0] = 1.0
    return sign * dist


def snap_nodes_to_surface(points, target_mesh, tol_mm=0.05):
    """Project points within tol_mm of target_mesh onto it. Returns (new_points, snapped_mask)."""
    closest, dist = closest_points_on_mesh(target_mesh, points)
    mask = dist <= tol_mm
    out = numpy.array(points, float)
    out[mask] = closest[mask]
    return out, mask


# =============================================================================
# Muscles: continue along the extended centerline into the eye, cut afterwards
# =============================================================================
def muscle_tube_into_eye(
        extended_centerline, muscle_mask, eye_surface, pixel_spacing, n_theta=40, step=1.0,
        inside_mm=2.0, end_trim=6, tail_fit_window=15, tail_min_frac=0.5, tail_max_frac=1.0,
        tail_end_scale=1.0, tail_power=1.0, taper_start_len=12, taper_shape="cone",
        **measure_kwargs):
    """
    Ring tube of a muscle that follows the extended centerline until `inside_mm` inside the eyeball and
    keeps its own (extrapolated) cross-section all the way. Cut it afterwards with the eyeball mesh,
    e.g. remesh_tube_as_solid(..., eye_surface=eye_mesh); the eye surface then forms the end face.

    extended_centerline : extended centerline in voxels (from the muscle origin towards the eye)
    eye_surface         : eyeball surface mesh (mm), closed; defines "inside the eye"
    inside_mm           : the centerline is followed until this depth inside the eye. If it does not get
                          that deep, it is extended straight along its end direction.
    end_trim            : measured rings dropped at the end of the segmentation (rounded tip)
    tail_fit_window     : rings used for the per-angle trend that is continued towards the eye
    tail_min_frac, tail_max_frac : clamp of the extrapolated radii relative to the tail profile
    tail_end_scale, tail_power   : optional extra narrowing along the extrapolated part
    measure_kwargs      : passed on to measure_muscle_along_centerline (start_extension_mm, voxel_mm,
                          fit_window, min_frac, max_frac, extension_end_scale, ...)
    Returns (tube_mesh_mm, info dict).
    """
    eye = eye_surface if isinstance(eye_surface, trimesh.Trimesh) else trimesh.load(eye_surface)
    ext = numpy.asarray(extended_centerline, float)

    # 1. Point inside_mm deep in the eye along the centerline (or a straight extension of it)
    sd = signed_distance_to_mesh(eye, ext * pixel_spacing)
    deep = numpy.flatnonzero(sd < -inside_mm)
    if len(deep):
        path = ext[:deep[0] + 1]
        n_straight = 0
    else:
        k_back = max(len(ext) - 1 - int(round(2.0 / pixel_spacing)), 0)
        t_end = ext[-1] - ext[k_back]
        t_end /= numpy.linalg.norm(t_end)
        cand = ext[-1] + numpy.arange(1, int(60.0 / (step * pixel_spacing)))[:, None] * step * t_end
        sd_c = signed_distance_to_mesh(eye, cand * pixel_spacing)
        hit = numpy.flatnonzero(sd_c < -inside_mm)
        if not len(hit):
            raise ValueError("Extended centerline does not reach the eye, even when extended straight")
        path = numpy.vstack([ext, cand[:hit[0] + 1]])
        n_straight = hit[0] + 1

    # 2. Frames and measured body (same measurement as before)
    cl, v1, v2, body = measure_muscle_along_centerline(
        path, muscle_mask, n_theta=n_theta, step=step, end_trim=end_trim, **measure_kwargs)
    m_body = len(body)
    n_trans = len(cl) - m_body

    # 3. Continue the muscle's own cross-section trend to the end (no contact ring)
    rows = body[max(m_body - tail_fit_window, 0):m_body]
    trans = extrapolate_profile(rows, n_trans, tail_min_frac, tail_max_frac)
    s_tr = numpy.arange(1, n_trans + 1) / max(n_trans, 1)
    trans *= (1.0 - (1.0 - tail_end_scale) * s_tr ** tail_power)[:, None]
    radii = numpy.vstack([body, trans])
    radii[m_body - 2:] = uniform_filter1d(radii[m_body - 2:], size=3, axis=0, mode="nearest")
    radii = clamp_radii_by_curvature(cl, radii, v1, v2)

    tube = build_ring_tube(cl, v1, v2, radii, pixel_spacing, taper_start_len, taper_shape)
    depth = -signed_distance_to_mesh(eye, cl[-1:] * pixel_spacing)[0]
    info = dict(n_nodes=len(cl), body_nodes=m_body, extrapolated_nodes=n_trans,
                straight_extension_nodes=n_straight, end_depth_mm=float(depth))
    print(f"   tube: {len(cl)} nodes (body {m_body}, extrapolated {n_trans}"
          + (f", {n_straight} straight beyond the centerline" if n_straight else "")
          + f"), ends {depth:.2f} mm inside the eye")
    return tube, info
