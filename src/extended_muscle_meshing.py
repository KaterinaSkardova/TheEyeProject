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
def radial_profiles_from_mask(centerline, mask, v1, v2, n_theta=40, r_max=40.0, dr=0.25, sigma=0.5):
    """
    March outward along each ray and stop at the FIRST exit from the muscle.
    The mask is Gaussian-smoothed (sigma in voxels) and the 0.5 crossing is interpolated, so radii are
    sub-voxel and free of staircase steps. Wider smoothing makes thin muscles measure too small
    (sigma 1.0: ~7 % less area, 0.5: ~2.5 %). Nodes outside the muscle give a row of NaN.
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
    blend_length_mm=2.0, follow_centerline=False,
):
    """
    centerline : (N, 3) voxel nodes, from the muscle origin to the touchdown on the eye (last node)
    v1, v2     : (N, 3) parallel transport frames of that centerline
    body_radii : (M, n_theta) smoothed radii for the first M nodes (the measured muscle body);
                 nodes M..N-1 form the transition to the eye
    eye_ring_mm, contact_center_mm : contact ring points and centre on the eyeball, in mm

    Transition (all blending happens in each node's own frame, so rings stay perpendicular to the
    centerline and keep their size; only the eye curvature is added out of plane towards the end):
        follow_centerline  : True = ring centres stay exactly on the centerline; the eye ring's shape is
                             placed around the centerline end instead of the contact centre (use when
                             the muscle is cut by the eyeball afterwards). False = centres drift towards
                             the contact centre over the blend length, so the last ring is exactly the
                             contact ring.
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
        end_shift = numpy.zeros(3) if follow_centerline else contact - centerline[-1]

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
                                    min_frac=0.3, max_frac=1.0, centerline_smooth=0.25):
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
    centerline_smooth  : spline smoothing of the input centerline (0 = pass exactly through its points;
                         use 0 for an already fitted / extended centerline that must be followed closely)
    """
    cl = resample_centerline(trimmed_centerline, step=step, smooth=centerline_smooth)
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
                         close_iterations=2, eye_surface=None, contact_open_mm=0.3, contact_zone_mm=1.0,
                         contact_fill_mm=0.0):
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
    contact_open_mm: with eye_surface, a morphological opening of this radius is applied within
                     contact_zone_mm of the eye. It removes thin flaps, wedges and spikes ("fingers") left
                     by the cut; parts thicker than ~2x the radius are not changed. 0 = off.
    contact_zone_mm: thickness of the zone next to the eye where the opening acts; the rest of the
                     muscle is left exactly as it was
    contact_fill_mm: with eye_surface, fill narrow gaps between muscle and eye (morphological closing of
                     muscle + eyeball with this radius, eyeball removed again). Where a muscle lies along
                     the eye and meets it at a shallow angle, this replaces the knife edge (which meshes
                     into a sawtooth of spikes) by a fillet and a larger, well-defined contact surface.
                     Only material inside the gap within 2x this radius of the eye is added. 0 = off.
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

        if contact_fill_mm > 0:
            from scipy.ndimage import binary_closing as _bclose, binary_dilation as _bdil
            r_f = max(int(round(contact_fill_mm / res)), 1)
            gf = numpy.arange(-r_f, r_f + 1)
            ball_f = (gf[:, None, None] ** 2 + gf[None, :, None] ** 2 + gf[None, None, :] ** 2) <= r_f ** 2
            region = _bdil(occ, structure=ball_f, iterations=2) & ~occ     # where a fillet could go
            rc = numpy.argwhere(region)
            rpts = lo + rc * res
            rnear = numpy.all((rpts >= lo_e) & (rpts <= hi_e), axis=1)
            rc, rpts = rc[rnear], rpts[rnear]
            sd_r = signed_distance_to_mesh(eye_m, rpts)
            eye_loc = numpy.zeros(shape, dtype=bool)                       # eyeball (+clearance) nearby
            ec = rc[sd_r < eye_clearance_mm]
            eye_loc[ec[:, 0], ec[:, 1], ec[:, 2]] = True
            gap_zone = numpy.zeros(shape, dtype=bool)                      # close enough to the eye to fill
            gz = rc[(sd_r >= eye_clearance_mm) & (sd_r < 2.0 * contact_fill_mm)]
            gap_zone[gz[:, 0], gz[:, 1], gz[:, 2]] = True
            pf = r_f + 1
            closed = _bclose(numpy.pad(occ | eye_loc, pf), structure=ball_f)[pf:-pf, pf:-pf, pf:-pf]
            added = closed & gap_zone
            occ |= added
            print(f"   contact fill: {added.sum() * res ** 3:.2f} mm^3 added between muscle and eye")

        if contact_open_mm > 0:
            from scipy.ndimage import binary_opening, binary_dilation
            zone = numpy.zeros(shape, dtype=bool)
            zc = cand[near][sd < contact_zone_mm]
            zone[zc[:, 0], zc[:, 1], zc[:, 2]] = True
            r_c = max(int(round(contact_open_mm / res)), 1)
            g_ = numpy.arange(-r_c, r_c + 1)
            ball = (g_[:, None, None] ** 2 + g_[None, :, None] ** 2 + g_[None, None, :] ** 2) <= r_c ** 2
            zone = binary_dilation(zone, structure=ball)               # cover whole flaps at the zone edge
            opened = binary_opening(numpy.pad(occ, r_c + 1), structure=ball)[
                r_c + 1:-r_c - 1, r_c + 1:-r_c - 1, r_c + 1:-r_c - 1]
            occ = numpy.where(zone, opened, occ)
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
def remesh_surface_except(surface, fixed_faces, element_size, refine_near=None, h_near=None,
                          near_distance_mm=0.3, transition_mm=3.0, surface_angle=60.0):
    """
    Re-mesh a closed surface with gmsh, keeping the faces in `fixed_faces` (bool mask) exactly as they are.
    Everything else is re-meshed with the given (optionally adaptive) element size; the outline of the
    fixed faces keeps its nodes, so the original fixed triangles are put back afterwards and connect exactly.
    Returns the new surface (trimesh, double precision).
    """
    import gmsh
    from scipy.spatial import cKDTree
    V = numpy.asarray(surface.vertices, float)
    F = numpy.asarray(surface.faces)
    fixed_faces = numpy.asarray(fixed_faces, bool)
    if fixed_faces.sum() == 0:
        raise ValueError("fixed_faces is empty")
    fixed_nodes = numpy.unique(F[fixed_faces])
    other_nodes = numpy.setdiff1d(numpy.arange(len(V)), fixed_nodes)
    fixed_cent = cKDTree(V[F[fixed_faces]].mean(axis=1))

    if refine_near is not None:
        ref_m = refine_near if isinstance(refine_near, trimesh.Trimesh) else trimesh.load(refine_near)
        ref_tree = cKDTree(numpy.vstack([ref_m.vertices, ref_m.triangles_center]))

        def size_at(x, y, z):
            d, _ = ref_tree.query([x, y, z])
            t = min(max((d - near_distance_mm) / max(transition_mm, 1e-9), 0.0), 1.0)
            return h_near + t * (element_size - h_near)
        s_min, s_max = min(h_near, element_size), max(h_near, element_size)
    else:
        size_at = lambda x, y, z: element_size
        s_min = s_max = element_size

    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("remesh_except")
        sf = gmsh.model.addDiscreteEntity(2)
        so = gmsh.model.addDiscreteEntity(2)
        gmsh.model.mesh.addNodes(2, sf, (fixed_nodes + 1).tolist(), V[fixed_nodes].ravel().tolist())
        gmsh.model.mesh.addNodes(2, so, (other_nodes + 1).tolist(), V[other_nodes].ravel().tolist())
        gmsh.model.mesh.addElementsByType(sf, 2, [], (F[fixed_faces] + 1).ravel().tolist())
        gmsh.model.mesh.addElementsByType(so, 2, [], (F[~fixed_faces] + 1).ravel().tolist())
        gmsh.model.mesh.classifySurfaces(numpy.radians(surface_angle), True, True, numpy.pi)
        gmsh.model.mesh.createGeometry()
        gmsh.model.geo.synchronize()

        keep_s, clear_s = [], []
        for _, tag in gmsh.model.getEntities(2):
            _, _, en = gmsh.model.mesh.getElements(2, tag)
            nt, xyz, _ = gmsh.model.mesh.getNodes(2, tag, True)
            pos = dict(zip(nt, numpy.asarray(xyz).reshape(-1, 3)))
            tri = numpy.asarray(en[0]).reshape(-1, 3)
            cent = numpy.array([(pos[a] + pos[b] + pos[c]) / 3 for a, b, c in tri])
            frac = (fixed_cent.query(cent)[0] < 1e-7).mean()
            if frac > 0.999:
                keep_s.append(tag)
            elif frac > 0.001:
                raise RuntimeError("gmsh mixed fixed and free triangles in one patch; try another surface_angle")
            else:
                clear_s.append(tag)
        if not keep_s:
            raise RuntimeError("gmsh did not keep the fixed faces as a separate surface")
        keep_c = set(abs(c) for s_ in keep_s for _, c in gmsh.model.getBoundary([(2, s_)], oriented=False))
        clear_c = [c for _, c in gmsh.model.getEntities(1) if c not in keep_c]
        gmsh.model.mesh.clear([(2, t) for t in clear_s] + [(1, c) for c in clear_c])

        for k, v in [("Mesh.MeshOnlyEmpty", 1), ("Mesh.MeshSizeMin", s_min), ("Mesh.MeshSizeMax", s_max),
                     ("Mesh.MeshSizeFromPoints", 0), ("Mesh.MeshSizeExtendFromBoundary", 0),
                     ("Mesh.MeshSizeFromCurvature", 0), ("Mesh.Algorithm", 6)]:
            gmsh.option.setNumber(k, v)
        gmsh.model.mesh.setSizeCallback(lambda dim, tag, x, y, z, lc: float(size_at(x, y, z)))
        gmsh.model.mesh.generate(2)

        # collect the re-meshed free surfaces
        tris, pts = [], {}
        for t in clear_s:
            nt, xyz, _ = gmsh.model.mesh.getNodes(2, t, True)
            pts.update(zip(nt, numpy.asarray(xyz).reshape(-1, 3)))
            _, _, en = gmsh.model.mesh.getElements(2, t)
            tris.append(numpy.asarray(en[0]).reshape(-1, 3))
        tris = numpy.vstack(tris)
    finally:
        gmsh.finalize()

    # assemble: re-meshed free part + ORIGINAL fixed faces, joined on the outline nodes (exact coordinates)
    tags = numpy.array(sorted(pts))
    idx_of = {t: i for i, t in enumerate(tags)}
    P = numpy.array([pts[t] for t in tags])
    free_faces = numpy.vectorize(idx_of.get)(tris)
    fixed_V = V[fixed_nodes]
    d, j = cKDTree(fixed_V).query(P)
    on_outline = d < 1e-9
    # vertices: fixed nodes first (original coordinates), then the free nodes not on the outline
    free_new = numpy.flatnonzero(~on_outline)
    vmap = numpy.empty(len(P), int)
    vmap[on_outline] = j[on_outline]
    vmap[free_new] = len(fixed_V) + numpy.arange(len(free_new))
    fixed_local = {n: i for i, n in enumerate(fixed_nodes)}
    faces_fixed = numpy.vectorize(fixed_local.get)(F[fixed_faces])
    verts = numpy.vstack([fixed_V, P[free_new]])
    faces = numpy.vstack([faces_fixed, vmap[free_faces]])
    out = trimesh.Trimesh(verts, faces, process=False)
    trimesh.repair.fix_normals(out)
    if out.volume < 0:
        out.invert()
    if not out.is_watertight:
        raise RuntimeError("re-meshed surface is not closed (outline nodes did not match)")
    return out


def gmsh_mesh_from_surface(
        surface, output_file, element_size=0.4, size_min=None,
        curvature_points=0, surface_angle=60.0, volume=True, report=True,
        retry_angles=(40.0, 30.0, 80.0), keep_surface_fallback=True, uniform=True,
        calibrate_size=True, algorithm_3d=1,
        refine_near=None, h_near=None, near_distance_mm=0.3, transition_mm=3.0, keep_surface=False,
        fixed_faces=None):
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
    refine_near     : optional surface mesh (mm), e.g. the eyeball. Elements get size h_near within
                      near_distance_mm of it and grow linearly to element_size over transition_mm.
                      Use it to match the eyeball's element size at the contact.
    h_near          : element size close to refine_near (e.g. the eyeball's h)
    near_distance_mm: distance from refine_near over which the size stays at h_near
    transition_mm   : distance over which the size grows from h_near to element_size; the size changes by
                      (element_size - h_near) / transition_mm per mm, keep that below ~0.3 for smooth grading
    keep_surface    : True = do not re-mesh the surface, only fill the volume (sizes apply to the interior).
    fixed_faces     : bool mask over the surface's faces (e.g. the conforming contact face). These faces are
                      kept exactly; everything else is re-meshed with the (adaptive) element size, connecting
                      to the fixed faces' outline nodes. Then the volume is meshed.
    Returns (surface_mesh_trimesh, quality dict). quality["mode"] tells which route succeeded.
    """
    import os, tempfile
    import gmsh

    if fixed_faces is not None:
        if not isinstance(surface, trimesh.Trimesh):
            surface = trimesh.load(surface, process=False)
        surface = remesh_surface_except(surface, fixed_faces, element_size, refine_near=refine_near,
                                        h_near=h_near, near_distance_mm=near_distance_mm,
                                        transition_mm=transition_mm, surface_angle=surface_angle)
        keep_surface = True
        if report:
            print(f"   surface re-meshed except {int(numpy.sum(fixed_faces))} fixed faces "
                  f"({len(surface.faces)} triangles now)")

    adaptive = refine_near is not None
    if adaptive:
        from scipy.spatial import cKDTree
        if h_near is None:
            raise ValueError("h_near is required with refine_near")
        ref_m = refine_near if isinstance(refine_near, trimesh.Trimesh) else trimesh.load(refine_near)
        surf_m = surface if isinstance(surface, trimesh.Trimesh) else trimesh.load(surface)
        reach = near_distance_mm + transition_mm + element_size
        lo_b, hi_b = surf_m.bounds[0] - reach, surf_m.bounds[1] + reach
        ref_pts = numpy.vstack([ref_m.vertices, ref_m.triangles_center])
        ref_pts = ref_pts[numpy.all((ref_pts >= lo_b) & (ref_pts <= hi_b), axis=1)]
        if len(ref_pts) == 0:
            print("   Note: refine_near surface is far from this object; using uniform size")
            adaptive = False
        else:
            ref_tree = cKDTree(ref_pts)

    def base_size(points):
        """Target size (before calibration) at the given points."""
        points = numpy.atleast_2d(points)
        if not adaptive:
            return numpy.full(len(points), element_size)
        d, _ = ref_tree.query(points)
        t = numpy.clip((d - near_distance_mm) / max(transition_mm, 1e-9), 0.0, 1.0)
        return h_near + t * (element_size - h_near)

    tmp = None
    surf_in = surface if isinstance(surface, trimesh.Trimesh) else None
    if surf_in is not None:
        # STL for re-meshing (gmsh merges near-coincident nodes, removing slivers from boolean seams);
        # kept surfaces are passed directly in double precision so their nodes stay exact
        tmp = tempfile.NamedTemporaryFile(suffix=".stl", delete=False).name
        surf_in.export(tmp)
        stl_path = tmp
    else:
        stl_path = surface
    if size_min is None:
        size_min = element_size / 2.0

    def _set_options(size):
        if adaptive:
            f = size / element_size
            gmsh.option.setNumber("Mesh.MeshSizeMax", f * max(element_size, h_near))
            gmsh.option.setNumber("Mesh.MeshSizeMin", f * min(element_size, h_near))
            gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
            gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
            gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
            gmsh.model.mesh.setSizeCallback(
                lambda dim, tag, x, y, z, lc: float(f * base_size([[x, y, z]])[0]))
        elif uniform:
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
            if surf_in is not None and angle is None:
                stag = gmsh.model.addDiscreteEntity(2)
                n_v = len(surf_in.vertices)
                gmsh.model.mesh.addNodes(2, stag, list(range(1, n_v + 1)),
                                         numpy.asarray(surf_in.vertices, float).ravel().tolist())
                gmsh.model.mesh.addElementsByType(stag, 2, [],
                                                  (numpy.asarray(surf_in.faces) + 1).ravel().tolist())
            else:
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
                    A, B = xyz[el[:, pairs[:, 0]]], xyz[el[:, pairs[:, 1]]]
                    L = numpy.linalg.norm(A - B, axis=2).ravel()
                    target = base_size(((A + B) / 2).reshape(-1, 3))
                    ratio = L / target
                    quality["edge_length"] = dict(mean=float(L.mean()), p5=float(numpy.percentile(L, 5)),
                                                  p95=float(numpy.percentile(L, 95)),
                                                  ratio_mean=float(ratio.mean()),
                                                  ratio_p5=float(numpy.percentile(ratio, 5)),
                                                  ratio_p95=float(numpy.percentile(ratio, 95)))
                    if adaptive:
                        near = target <= h_near + 1e-9
                        far = target >= element_size - 1e-9
                        quality["edge_length"]["near_mean"] = float(L[near].mean()) if near.any() else None
                        quality["edge_length"]["far_mean"] = float(L[far].mean()) if far.any() else None
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
            for _ in range(3):                                    # converge edges to their local target
                trial = _attempt(angle, size, write=False)
                ratio = trial.get("edge_length", {}).get("ratio_mean")
                if not ratio or abs(ratio - 1.0) < 0.03:
                    break
                size /= ratio
            return _attempt(angle, size)

        if keep_surface:
            quality = _run(None)
        else:
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
            if adaptive:
                nm = f"{e['near_mean']:.3f}" if e.get("near_mean") else "-"
                fm = f"{e['far_mean']:.3f}" if e.get("far_mean") else "-"
                print(f"   edge length: near the surface {nm} mm (target {h_near}), far {fm} mm "
                      f"(target {element_size}); edge/target 5-95 %: {e['ratio_p5']:.2f} - {e['ratio_p95']:.2f}")
            else:
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


def find_eye_intersection_index_mesh(centerline_extended, eye_mesh, pixel_spacing, min_depth_mm=0.0,
                                     start_index=0):
    """
    Index of the first centerline point (from start_index on) that lies at least min_depth_mm inside the
    eyeball mesh. A small depth (e.g. 0.3-0.5 mm) ignores places where a centerline running along the
    sclera only grazes the eye surface before the real insertion.
    """
    sd = signed_distance_to_mesh(eye_mesh, numpy.asarray(centerline_extended, float) * pixel_spacing)
    inside = numpy.flatnonzero(sd[start_index:] < -min_depth_mm) + start_index
    return int(inside[0]) if len(inside) else len(centerline_extended) - 1


def centerline_deviation(tube_mesh, n_theta, reference_centerline, pixel_spacing):
    """Max / mean distance (mm) of the tube's ring centres from a reference centerline (in voxels)."""
    from scipy.spatial import cKDTree
    rings = tube_mesh.vertices[:-2].reshape(-1, n_theta, 3).mean(axis=1)
    ref = numpy.asarray(reference_centerline, float) * pixel_spacing
    dense = numpy.vstack([numpy.linspace(ref[i], ref[i + 1], 10, endpoint=False) for i in range(len(ref) - 1)]
                         + [ref[-1:]])
    d, _ = cKDTree(dense).query(rings)
    return float(d.max()), float(d.mean())


# =============================================================================
# Transfer the measured muscle shape from the original centerline to the extended one
# =============================================================================
def _arc(points):
    return numpy.concatenate([[0.0], numpy.cumsum(numpy.linalg.norm(numpy.diff(points, axis=0), axis=1))])


def _project_to_polyline_arc(polyline, point):
    """Arc length (same units as the polyline) of the closest point on a polyline to `point`."""
    a, b = polyline[:-1], polyline[1:]
    ab = b - a
    t = numpy.clip(numpy.einsum("ij,ij->i", point - a, ab) / numpy.maximum((ab ** 2).sum(1), 1e-12), 0, 1)
    proj = a + t[:, None] * ab
    k = int(numpy.argmin(numpy.linalg.norm(proj - point, axis=1)))
    arc = _arc(polyline)
    return arc[k] + t[k] * numpy.linalg.norm(ab[k])


def _rotate_profile(profile, phi):
    """Profile r(theta) measured relative to some v1 -> profile in a frame rotated by -phi: r'(psi) = r(psi - phi)."""
    th = theta_steps(len(profile))
    return numpy.interp(th - phi, th, profile, period=2 * numpy.pi)


def transfer_muscle_profiles(
        short_centerline, muscle_mask, extended_centerline, pixel_spacing, n_theta=40, step=1.0,
        short_smooth=0.25, extended_smooth=0.0, start_trim=6, end_trim=6, match="start",
        start_extension_mm=0.0, fit_window=15, min_frac=0.5, max_frac=1.0,
        extension_end_scale=1.0, extension_power=1.0, measure_sigma=0.5, ring_smooth=3, long_smooth=5):
    """
    1. Measure the cross-sections along the ORIGINAL (short) centerline in the segmented muscle.
    2. Filter them.
    3. Map them onto the EXTENDED centerline by distance along the muscle, rotated so each profile keeps
       its anatomical orientation. The new muscle has the original shape but the extended centerline's
       position. Past the measured part, generate_extended_muscle_mesh extrapolates the tail trend.

    short_centerline    : original muscle centerline (voxels), runs through the segmentation
    extended_centerline : extended centerline (voxels), trimmed at the eye; it may differ in position
                          from the short one everywhere, including the start
    short_smooth        : spline smoothing of the short centerline before measuring (skeleton noise)
    extended_smooth     : smoothing of the extended centerline (0 = follow it exactly)
    start_trim, end_trim: measured rings dropped at both rounded tips of the segmentation; they are
                          replaced by the extrapolated trend
    match               : how positions along the two centerlines correspond:
                          "start" = same distance from the muscle start (keeps the shape's length),
                          "both"  = start and end of the measured muscle are matched to their closest
                                    points on the extended centerline (stretches/compresses slightly)
    start_extension_mm  : extra length before the muscle start (straight if the extended centerline
                          does not reach back that far), with the head trend extrapolated
    fit_window, min_frac, max_frac, extension_end_scale, extension_power : head extrapolation, as in
                          measure_muscle_along_centerline
    measure_sigma       : mask smoothing (voxels) when finding the boundary; smaller = closer to the
                          segmented size, larger = smoother but thinner
    ring_smooth, long_smooth : filter sizes around each cross-section (angles) and along the muscle
                          (nodes); larger ring filters flatten the ends of strongly flattened sections
    Returns centerline, v1, v2, body_radii, info: drop-in replacement for measure_muscle_along_centerline,
    to be passed to generate_extended_muscle_mesh (with follow_centerline=True).
    """
    sp = float(pixel_spacing)

    # ---- 1. Measure along the original centerline -----------------------------------------------
    cs = resample_centerline(short_centerline, step=step, smooth=short_smooth)
    _, v1s, v2s = parallel_transport_frames(cs)
    rs = radial_profiles_from_mask(cs, muscle_mask, v1s, v2s, n_theta=n_theta, sigma=measure_sigma)
    valid = ~numpy.all(numpy.isnan(rs), axis=1)
    if valid.sum() < start_trim + end_trim + 4:
        raise ValueError("Too few cross-sections along the original centerline lie inside the muscle mask")
    first, last = numpy.flatnonzero(valid)[[0, -1]]
    cs, v1s, rs = cs[first:last + 1], v1s[first:last + 1], rs[first:last + 1]

    # ---- 2. Filter ---------------------------------------------------------------------------------
    rs = smooth_radii(rs, long_size=long_smooth, ring_size=ring_smooth)
    s_short = _arc(cs) * sp                                     # mm from the muscle start
    core = slice(start_trim, len(cs) - end_trim)                # reliable part (rounded tips removed)
    u_core, r_core, v1_core = s_short[core], rs[core], v1s[core]

    # ---- 3. Extended centerline, frames, correspondence ------------------------------------------
    ce = resample_centerline(extended_centerline, step=step, smooth=extended_smooth)
    s_ext = _arc(ce) * sp
    s0 = _project_to_polyline_arc(ce, cs[0]) * sp               # muscle start on the extended centerline
    if match == "both":
        s1 = _project_to_polyline_arc(ce, cs[-1]) * sp
        scale = (s1 - s0) / s_short[-1]
    elif match == "start":
        scale = 1.0
    else:
        raise ValueError("match must be 'start' or 'both'")

    # optional start extension: keep/extend the extended centerline before the muscle start
    need_back = s0 - start_extension_mm
    if need_back < 0:                                            # extended centerline too short: straight
        t_start = ce[min(5, len(ce) - 1)] - ce[0]
        t_start /= numpy.linalg.norm(t_start)
        n_add = int(numpy.ceil(-need_back / (step * sp)))
        pre = ce[0] - numpy.arange(n_add, 0, -1)[:, None] * step * t_start
        ce = numpy.vstack([pre, ce])
        s_ext = _arc(ce) * sp
        s0 += n_add * step * sp
    keep = s_ext >= s0 - start_extension_mm - 1e-9
    ce, s_ext = ce[keep], s_ext[keep]
    _, v1e, v2e = parallel_transport_frames(ce)
    te = numpy.cross(v1e, v2e)

    u = (s_ext - s0) / scale                                     # position in "short muscle" coordinates
    in_core = (u >= u_core[0]) & (u <= u_core[-1])
    head = u < u_core[0]
    body_end = int(numpy.flatnonzero(u <= u_core[-1])[-1]) + 1  # nodes after this = transition

    # ---- 4. Map the profiles, with orientation alignment ----------------------------------------
    radii = numpy.zeros((body_end, n_theta))
    phis = numpy.zeros(body_end)
    for j in range(body_end):
        uj = min(max(u[j], u_core[0]), u_core[-1])
        k = numpy.searchsorted(u_core, uj)
        k0, k1 = max(k - 1, 0), min(k, len(u_core) - 1)
        w = 0.0 if k1 == k0 else (uj - u_core[k0]) / (u_core[k1] - u_core[k0])
        prof = (1 - w) * r_core[k0] + w * r_core[k1]
        ref = (1 - w) * v1_core[k0] + w * v1_core[k1]           # measuring direction of angle 0
        ref = ref - (ref @ te[j]) * te[j]                        # into the extended cross-section plane
        phis[j] = numpy.arctan2(ref @ v2e[j], ref @ v1e[j])
        radii[j] = _rotate_profile(prof, phis[j])

    # head (trimmed tip + start extension): continue the trend of the first reliable rings backwards
    n_head = int(head[:body_end].sum())
    if n_head:
        first_core = n_head
        rows = radii[first_core:first_core + fit_window][::-1]
        extra = extrapolate_profile(rows, n_head, min_frac, max_frac)[::-1]       # far tip first
        u_head = u[:n_head]
        s_tip = numpy.clip((u_core[0] - u_head) / max(u_core[0] - u_head[0], 1e-9), 0, 1)
        is_ext = u_head < 0                                       # beyond the segmented muscle start
        scale_ext = numpy.where(is_ext, 1.0 - (1.0 - extension_end_scale) *
                                numpy.clip(-u_head / max(start_extension_mm, 1e-9), 0, 1) ** extension_power, 1.0)
        radii[:n_head] = extra * scale_ext[:, None]

    radii = clamp_radii_by_curvature(ce[:body_end], radii, v1e[:body_end], v2e[:body_end])

    info = dict(muscle_start_on_extended_mm=float(s0), length_scale=float(scale),
                measured_length_mm=float(s_short[-1]), body_nodes=body_end, head_nodes=n_head,
                transition_nodes=len(ce) - body_end,
                rotation_deg=(float(numpy.degrees(phis.min())), float(numpy.degrees(phis.max()))))
    print(f"   measured along original centerline: {s_short[-1]:.1f} mm; mapped onto extended centerline "
          f"from {s0:.1f} mm (scale {scale:.3f})")
    print(f"   nodes: {len(ce)} (head {n_head}, body {body_end - n_head}, transition {len(ce) - body_end}), "
          f"profile rotation {info['rotation_deg'][0]:.0f} to {info['rotation_deg'][1]:.0f} deg")
    return ce, v1e, v2e, radii, info


# =============================================================================
# Conforming contact: use the eyeball's own triangles as the contact face
# =============================================================================
def _boundary_loops(mesh):
    """Closed boundary loops of a mesh as ordered vertex index arrays."""
    edges = mesh.edges_sorted
    uniq, counts = numpy.unique(edges, axis=0, return_counts=True)
    bnd = uniq[counts == 1]
    adj = {}
    for a, b in bnd:
        adj.setdefault(a, []).append(b)
        adj.setdefault(b, []).append(a)
    loops, seen = [], set()
    for start in adj:
        if start in seen:
            continue
        loop, prev, cur = [start], None, start
        seen.add(start)
        while True:
            nxt = [n for n in adj[cur] if n != prev]
            if not nxt or nxt[0] == start:
                break
            prev, cur = cur, nxt[0]
            if cur in seen:
                break
            loop.append(cur)
            seen.add(cur)
        loops.append(numpy.array(loop))
    return loops


def _zipper_by_angle(A_pts, B_pts, centre, e1, e2):
    """
    Triangulate the strip between two closed loops by marching around `centre` in the plane (e1, e2):
    always advance the loop whose next point comes first in angle. Both loops must already run in the
    same (increasing-angle) direction. Avoids the crossings a greedy shortest-diagonal zipper can make
    when the loops are offset from each other.
    """
    def ang(P):
        q = P - centre
        return numpy.unwrap(numpy.arctan2(q @ e2, q @ e1))
    n, m = len(A_pts), len(B_pts)
    aA = ang(A_pts)
    aB0 = numpy.arctan2((B_pts - centre) @ e2, (B_pts - centre) @ e1)
    j0 = int(numpy.argmin(numpy.abs(numpy.angle(numpy.exp(1j * (aB0 - aA[0]))))))
    order_B = numpy.roll(numpy.arange(m), -j0)
    aB = numpy.unwrap(aB0[order_B])
    aB += 2 * numpy.pi * numpy.round((aA[0] - aB[0]) / (2 * numpy.pi))
    aA_ext = numpy.append(aA, aA[0] + 2 * numpy.pi)
    aB_ext = numpy.append(aB, aB[0] + 2 * numpy.pi)
    tris, i, j = [], 0, 0
    while i < n or j < m:
        a_i, a_n = i % n, (i + 1) % n
        b_j, b_n = order_B[j % m], order_B[(j + 1) % m]
        if j >= m or (i < n and aA_ext[i + 1] <= aB_ext[j + 1]):
            tris.append(("A", a_i, "B", b_j, "A", a_n))
            i += 1
        else:
            tris.append(("A", a_i, "B", b_j, "B", b_n))
            j += 1
    return tris


def _self_intersecting_pairs(mesh, candidate_faces):
    """Count intersections between candidate faces and all non-adjacent faces nearby (exact tri-tri test)."""
    from scipy.spatial import cKDTree
    V, F = numpy.asarray(mesh.vertices), numpy.asarray(mesh.faces)
    cen = V[F].mean(axis=1)
    rad = numpy.linalg.norm(V[F] - cen[:, None, :], axis=2).max(axis=1)
    tree = cKDTree(cen)
    rmax = rad.max()

    def seg_tri(p, q, t):
        # segment p-q against triangle t (Moller-Trumbore), excluding shared endpoints
        e1, e2 = t[1] - t[0], t[2] - t[0]
        d = q - p
        h = numpy.cross(d, e2); a = e1 @ h
        if abs(a) < 1e-14:
            return False
        f = 1.0 / a; s_ = p - t[0]; u = f * (s_ @ h)
        if u < 1e-9 or u > 1 - 1e-9:
            return False
        qv = numpy.cross(s_, e1); v = f * (d @ qv)
        if v < 1e-9 or u + v > 1 - 1e-9:
            return False
        tt = f * (e2 @ qv)
        return 1e-9 < tt < 1 - 1e-9

    N = numpy.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    N /= numpy.maximum(numpy.linalg.norm(N, axis=1, keepdims=True), 1e-30)
    hits = 0
    for fi in candidate_faces:
        for fj in tree.query_ball_point(cen[fi], rad[fi] + rmax):
            if fj == fi:
                continue
            shared = len(set(F[fi]) & set(F[fj]))
            if shared >= 2:                                   # neighbours across an edge: check for a fold
                if N[fi] @ N[fj] < -0.985:
                    hits += 1
                continue
            T1, T2 = V[F[fi]], V[F[fj]]
            if any(seg_tri(T1[k], T1[(k + 1) % 3], T2) for k in range(3)) or \
               any(seg_tri(T2[k], T2[(k + 1) % 3], T1) for k in range(3)):
                hits += 1
    return hits


def _refine_band(mesh, band_mask, protected_vertices, project, max_edge, max_iter=12):
    """
    Split band edges longer than max_edge at their midpoints (new nodes projected with `project`).
    Edges whose both ends are protected (e.g. the target patch) are never split. Faces next to a split
    edge outside the band are split too (keeps the mesh conforming); their new node also lies on the
    original surface, so the shape does not change.
    """
    V = [numpy.asarray(mesh.vertices, float)]
    F = numpy.asarray(mesh.faces).copy()
    band = numpy.asarray(band_mask, bool).copy()
    prot = numpy.zeros(len(V[0]), bool); prot[protected_vertices] = True
    for _ in range(max_iter):
        Vall = numpy.vstack(V)
        e = numpy.concatenate([F[band][:, [0, 1]], F[band][:, [1, 2]], F[band][:, [2, 0]]])
        e = numpy.sort(e, axis=1)
        e = numpy.unique(e, axis=0)
        L = numpy.linalg.norm(Vall[e[:, 0]] - Vall[e[:, 1]], axis=1)
        prot_all = numpy.zeros(len(Vall), bool); prot_all[:len(prot)] = prot
        ok = (L > max_edge) & ~(prot_all[e[:, 0]] & prot_all[e[:, 1]])
        if not ok.any():
            break
        order = numpy.argsort(-L[ok])
        cand = e[ok][order]
        # map edge -> faces
        face_edges = {}
        for fi, f in enumerate(F):
            for a, b in ((f[0], f[1]), (f[1], f[2]), (f[2], f[0])):
                face_edges.setdefault((min(a, b), max(a, b)), []).append(fi)
        touched = numpy.zeros(len(F), bool)
        new_faces, new_band, kill = [], [], []
        mids = []
        nv = len(Vall)
        for a, b in cand:
            fs = face_edges.get((a, b), [])
            if not fs or touched[fs].any():
                continue
            w = nv + len(mids)
            mids.append(0.5 * (Vall[a] + Vall[b]))
            for fi in fs:
                f = F[fi]
                k = [i for i in range(3) if f[i] not in (a, b)][0]
                x, u, v = f[k], f[(k + 1) % 3], f[(k + 2) % 3]           # keep orientation
                new_faces += [[x, u, w], [x, w, v]]
                new_band += [band[fi], band[fi]]
                kill.append(fi)
                touched[fi] = True
        if not mids:
            break
        mids = project(numpy.array(mids))
        V.append(mids)
        keep_f = numpy.ones(len(F), bool); keep_f[kill] = False
        F = numpy.vstack([F[keep_f], numpy.array(new_faces)])
        band = numpy.concatenate([band[keep_f], numpy.array(new_band, bool)])
    out = trimesh.Trimesh(numpy.vstack(V), F, process=False)
    return out, band


def conform_contact_to_surface(obj_surface, target_mesh, on_tol_mm=0.02, footprint_tol_mm=0.05,
                               smooth_outline_iterations=3, lift_rings=(1, 2, 3),
                               patch_modes=("centroid", "all", "any"), morph_band_mm=1.0,
                               morph_smooth_nodes=3, keep_wall=True, rim_gap_mm=None, rim_tol_mm=0.1,
                               ring_size_mm=0):
    """
    Replace the contact face of obj_surface (the part lying on target_mesh, e.g. the ON on the eyeball) by
    the target's own triangles, so the contact is node-for-node identical with the target.

    1. contact face of obj  = triangles whose vertices all lie within on_tol_mm of the target
    2. target triangles covering that face become the new contact face (see patch_modes)
    3. obj's contact face and lift_rings rings of triangles above it are removed
    4. the hole edge is stitched to the boundary of the target patch with a strip of triangles, which is
       checked for self-intersections
    smooth_outline_iterations: the selected patch has a jagged outline (whole triangles are in or out);
       triangles attached by only one edge are dropped and notches bordered on two edges are filled
    lift_rings: rings of triangles above the contact removed before stitching; tried in order
    patch_modes: which target triangles form the contact, tried in order until the stitching is valid:
       "centroid" = triangle centre on the contact face: matches the object's footprint (default first)
       "all"      = all three corners on it: slightly smaller, strictly inside the object's outline
       "any"      = at least one corner on the contact face: slightly larger than the footprint
    morph_band_mm: before stitching, the object's wall within this distance of the target is shifted
       sideways by the local mismatch between its own contact outline and the patch outline (each wall
       node takes the offset of its nearest outline point; full at the bottom, fading smoothly to zero at
       the top of the band). This absorbs the patch's triangle zigzag over the band instead of in the
       last element, so no dent forms at the contact. 0 = off.
    morph_smooth_nodes: smoothing of the outline offsets along the outline (number of nodes); keeps the
       zigzag of the eyeball triangles from being copied up the wall
    keep_wall: True (default) = the object's wall is kept unchanged down to the target (no lift ring, no
       morph, hence no dent at the last element). False = the older strip method (lift ring + morph), which
       uses the target's triangles for the whole contact but cuts across the last wall element. The target patch is then taken strictly inside the object's contact outline and
       the thin ring between the two outlines is filled with triangles lying on the target surface: the
       contact face = target triangles (identical nodes) + that ring (its outer nodes are the object's
       rim nodes, which lie on the target surface but are not target nodes).
    rim_gap_mm: with keep_wall, minimum width of that ring; patch triangles closer to the object's outline
       are removed so the ring has no slivers (default: half the target's mean edge length)
    rim_tol_mm: with keep_wall, wall triangles lying entirely within this distance of the target count as
       contact and are replaced by the ring. This removes nearly flat "toes" where the wall comes down onto
       the target almost tangentially (they would give flat tets); the wall changes by at most rim_tol_mm.
    ring_size_mm: with keep_wall, optionally split the band's triangles down to about this size (experimental;
       0 = off, default). Recommended instead: gmsh_mesh_from_surface(..., fixed_faces=info["contact_face_mask"]),
       which re-meshes everything except the contact face with the adaptive element size.
    Returns (new_surface, info) with info["target_vertex_ids"], info["target_face_ids"]: indices into
    target_mesh of the contact nodes / faces (for coupling in the simulation).
    """
    from scipy.sparse.csgraph import connected_components
    from scipy.sparse import coo_matrix
    obj = obj_surface.copy()
    tgt = target_mesh if isinstance(target_mesh, trimesh.Trimesh) else trimesh.load(target_mesh)

    # 1. contact face on the object
    _, d_obj = closest_points_on_mesh(tgt, obj.vertices)
    on_tgt = d_obj < on_tol_mm
    contact_faces = numpy.all(on_tgt[obj.faces], axis=1)
    if contact_faces.sum() < 3:
        raise ValueError("No contact face found: the object does not lie on the target surface "
                         "(increase on_tol_mm or check that the same eyeball mesh was used for the cut)")
    contact = obj.submesh([numpy.flatnonzero(contact_faces)], append=True)
    c_loops = _boundary_loops(contact)
    contact_loop = max(c_loops, key=len)                       # the object's own contact outline

    # which target corners / centres lie on the contact face
    _, d_c = closest_points_on_mesh(contact, tgt.triangles_center)
    near_f = d_c < 1.0 + 4 * footprint_tol_mm
    near_v = numpy.unique(tgt.faces[near_f])
    inside_v = numpy.zeros(len(tgt.vertices), bool)
    _, d_v = closest_points_on_mesh(contact, tgt.vertices[near_v])
    inside_v[near_v[d_v < footprint_tol_mm]] = True
    corners_in = inside_v[tgt.faces].sum(axis=1)
    centre_in = d_c < footprint_tol_mm
    allowed_by_mode = {"any": corners_in >= 1, "centroid": centre_in, "all": corners_in == 3}

    fa = tgt.face_adjacency

    def _is_disc(face_ids):
        pm = tgt.submesh([face_ids], append=True)
        e = pm.edges_sorted
        _, inv, cnt = numpy.unique(e, axis=0, return_inverse=True, return_counts=True)
        bnd = e[cnt[inv] == 1]
        if len(bnd) == 0:
            return False
        deg = numpy.bincount(bnd.ravel(), minlength=len(pm.vertices))
        if deg.max() > 2:
            return False
        chi = len(pm.vertices) - len(pm.edges_unique) + len(pm.faces)
        return chi == 1 and len(_boundary_loops(pm)) == 1

    def _select_patch(allowed):
        sel = numpy.flatnonzero(allowed)
        if len(sel) < 3:
            return None
        in_sel = numpy.zeros(len(tgt.faces), bool); in_sel[sel] = True
        pairs = fa[in_sel[fa[:, 0]] & in_sel[fa[:, 1]]]
        loc = -numpy.ones(len(tgt.faces), int); loc[sel] = numpy.arange(len(sel))
        g = coo_matrix((numpy.ones(len(pairs)), (loc[pairs[:, 0]], loc[pairs[:, 1]])), shape=(len(sel), len(sel)))
        _, lab = connected_components(g, directed=False)
        sel = sel[lab == numpy.argmax(numpy.bincount(lab))]
        for _ in range(smooth_outline_iterations):                    # fill notches / drop teeth, keep a disc
            changed = False
            for mode in ("fill", "trim"):
                member = numpy.zeros(len(tgt.faces), bool); member[sel] = True
                nb = numpy.zeros(len(tgt.faces), int)
                numpy.add.at(nb, fa[:, 0], member[fa[:, 1]])
                numpy.add.at(nb, fa[:, 1], member[fa[:, 0]])
                cand = member.copy()
                if mode == "fill":
                    cand[~member & (nb >= 2) & allowed] = True
                else:
                    cand[member & (nb <= 1)] = False
                if numpy.array_equal(cand, member):
                    continue
                ids = numpy.flatnonzero(cand)
                if _is_disc(ids):
                    sel, changed = ids, True
            if not changed:
                break
        return sel if _is_disc(sel) else None

    vf = obj.vertex_faces

    def _stitch(patch, rings):
        patch_loops = _boundary_loops(patch)
        nrm = patch.face_normals.mean(axis=0); nrm /= numpy.linalg.norm(nrm)
        e1 = numpy.cross(nrm, [1, 0, 0] if abs(nrm[0]) < 0.9 else [0, 1, 0]); e1 /= numpy.linalg.norm(e1)
        e2 = numpy.cross(nrm, e1)
        centre = patch.vertices.mean(axis=0)

        def signed_area(P):
            x, y = (P - centre) @ e1, (P - centre) @ e2
            return 0.5 * numpy.sum(x * numpy.roll(y, -1) - y * numpy.roll(x, -1))

        remove = numpy.any(on_tgt[obj.faces], axis=1)
        for _ in range(rings - 1):
            vs = numpy.unique(obj.faces[remove])
            nbf = vf[vs].ravel(); nbf = nbf[nbf >= 0]
            remove[nbf] = True
        kept = obj.submesh([numpy.flatnonzero(~remove)], append=True)
        kept_loops = _boundary_loops(kept)
        if len(kept_loops) != 1 or len(patch_loops) != 1:
            return None, None, -1
        A_idx, B_idx = kept_loops[0], patch_loops[0]
        if signed_area(kept.vertices[A_idx]) < 0:
            A_idx = A_idx[::-1]
        if signed_area(patch.vertices[B_idx]) < 0:
            B_idx = B_idx[::-1]
        band_faces = numpy.zeros(0, int)
        if morph_band_mm > 0:
            kept = kept.copy()
            _, d_k = closest_points_on_mesh(tgt, kept.vertices)
            d_A = d_k[A_idx].max()
            if morph_band_mm > d_A:
                # for every point of the object's own contact outline C: in-plane offset to the closest point
                # of the patch outline B, smoothed along C so the patch's triangle zigzag is not copied upwards
                C = contact.vertices[contact_loop]
                Bp = patch.vertices[B_idx]
                Bseg_a, Bseg_b = Bp, numpy.roll(Bp, -1, axis=0)
                ab = Bseg_b - Bseg_a
                off = numpy.zeros_like(C)
                for k, c in enumerate(C):
                    t_ = numpy.clip(numpy.einsum("ij,ij->i", c - Bseg_a, ab) / numpy.maximum((ab ** 2).sum(1), 1e-12), 0, 1)
                    proj = Bseg_a + t_[:, None] * ab
                    off[k] = proj[numpy.argmin(numpy.linalg.norm(proj - c, axis=1))] - c
                off -= (off @ nrm)[:, None] * nrm                       # sideways only (in the contact plane)
                win = max(int(morph_smooth_nodes), 1)
                if win > 1:
                    kern = numpy.ones(win) / win
                    off = numpy.stack([numpy.convolve(numpy.r_[off[-win:, j], off[:, j], off[:win, j]], kern, "same")[win:-win]
                                       for j in range(3)], axis=1)
                # every band node takes the (smoothed) offset of its nearest outline point
                from scipy.spatial import cKDTree
                in_band = numpy.flatnonzero(d_k < morph_band_mm)
                _, nearest = cKDTree(C).query(kept.vertices[in_band])
                t = numpy.clip((morph_band_mm - d_k[in_band]) / max(morph_band_mm - d_A, 1e-9), 0.0, 1.0)
                w = t * t * (3.0 - 2.0 * t)                       # smoothstep: 1 at the hole edge, 0 at the top
                V = kept.vertices.copy()
                V[in_band] += w[:, None] * off[nearest]
                kept.vertices = V
                band_faces = numpy.flatnonzero(numpy.any(numpy.isin(kept.faces, in_band), axis=1))

        strip = _zipper_by_angle(kept.vertices[A_idx], patch.vertices[B_idx], centre, e1, e2)
        nk = len(kept.vertices)

        def gid(tag, k):
            return int(A_idx[k]) if tag == "A" else nk + int(B_idx[k])
        sf = numpy.array([[gid(t[0], t[1]), gid(t[2], t[3]), gid(t[4], t[5])] for t in strip])
        verts = numpy.vstack([kept.vertices, patch.vertices])
        faces = numpy.vstack([kept.faces, patch.faces[:, ::-1] + nk, sf])
        m = trimesh.Trimesh(verts, faces, process=False)
        trimesh.repair.fix_normals(m)
        check = numpy.concatenate([band_faces, numpy.arange(len(kept.faces) + len(patch.faces), len(faces))])
        hits = _self_intersecting_pairs(m, check)
        return m, sf, hits if m.is_watertight else -1

    if keep_wall:
        # contact incl. nearly flat toes: wall triangles entirely within rim_tol_mm of the target
        contact_faces = numpy.all((d_obj < rim_tol_mm)[obj.faces], axis=1)
        contact = obj.submesh([numpy.flatnonzero(contact_faces)], append=True)
        contact_loop = max(_boundary_loops(contact), key=len)
        _, d_v2 = closest_points_on_mesh(contact, tgt.vertices[near_v])
        inside_v = numpy.zeros(len(tgt.vertices), bool)
        inside_v[near_v[d_v2 < rim_tol_mm + footprint_tol_mm]] = True
        # patch strictly inside the outline, at least rim_gap_mm away from it (no slivers in the ring)
        C = contact.vertices[contact_loop]
        gap_min = rim_gap_mm if rim_gap_mm is not None else 0.5 * float(tgt.edges_unique_length.mean())
        Cseg_a, Cseg_b = C, numpy.roll(C, -1, axis=0)
        cab = Cseg_b - Cseg_a

        def dist_to_outline(P):
            out = numpy.empty(len(P))
            for k, pnt in enumerate(P):
                tt = numpy.clip(numpy.einsum("ij,ij->i", pnt - Cseg_a, cab) / numpy.maximum((cab ** 2).sum(1), 1e-12), 0, 1)
                out[k] = numpy.min(numpy.linalg.norm(Cseg_a + tt[:, None] * cab - pnt, axis=1))
            return out

        cand_v = numpy.flatnonzero(inside_v)
        far_v = numpy.zeros(len(tgt.vertices), bool)
        far_v[cand_v[dist_to_outline(tgt.vertices[cand_v]) >= gap_min]] = True
        allowed = numpy.all(far_v[tgt.faces], axis=1)
        sel = _select_patch(allowed)
        if sel is None:
            raise RuntimeError("keep_wall: no clean patch inside the outline; reduce rim_gap_mm")
        patch = tgt.submesh([sel], append=True)
        B_idx = _boundary_loops(patch)[0]
        nrm = patch.face_normals.mean(axis=0); nrm /= numpy.linalg.norm(nrm)
        e1 = numpy.cross(nrm, [1, 0, 0] if abs(nrm[0]) < 0.9 else [0, 1, 0]); e1 /= numpy.linalg.norm(e1)
        e2 = numpy.cross(nrm, e1)
        centre = patch.vertices.mean(axis=0)

        def signed_area(P):
            x, y = (P - centre) @ e1, (P - centre) @ e2
            return 0.5 * numpy.sum(x * numpy.roll(y, -1) - y * numpy.roll(x, -1))

        kept = obj.submesh([numpy.flatnonzero(~contact_faces)], append=True)   # wall untouched
        kept_loops = _boundary_loops(kept)
        if len(kept_loops) != 1:
            raise RuntimeError("keep_wall: the object's contact outline is not a single loop")
        A_idx = kept_loops[0]
        if signed_area(kept.vertices[A_idx]) < 0:
            A_idx = A_idx[::-1]
        if signed_area(patch.vertices[B_idx]) < 0:
            B_idx = B_idx[::-1]
        nk = len(kept.vertices)
        h_ring = ring_size_mm if ring_size_mm is not None else float(tgt.edges_unique_length.mean())
        A_pts, B_pts = kept.vertices[A_idx], patch.vertices[B_idx]

        def _level_loop(level):
            """Closed contour of the original surface at a given distance to the target (largest loop)."""
            F = obj.faces
            dF = d_obj[F]
            segs = []
            for f, df in zip(F, dF):
                cut = []
                for i0, i1 in ((0, 1), (1, 2), (2, 0)):
                    a0, a1 = df[i0], df[i1]
                    if (a0 - level) * (a1 - level) < 0:
                        key = (min(f[i0], f[i1]), max(f[i0], f[i1]))
                        tt = (level - a0) / (a1 - a0)
                        cut.append((key, obj.vertices[f[i0]] + tt * (obj.vertices[f[i1]] - obj.vertices[f[i0]])))
                if len(cut) == 2:
                    segs.append(cut)
            if not segs:
                return None
            pos, adj = {}, {}
            for (k0, p0), (k1, p1) in segs:
                pos[k0], pos[k1] = p0, p1
                adj.setdefault(k0, []).append(k1)
                adj.setdefault(k1, []).append(k0)
            loops, seen = [], set()
            for start in adj:
                if start in seen:
                    continue
                loop, prev, cur = [start], None, start
                seen.add(start)
                while True:
                    nxt = [n for n in adj[cur] if n != prev]
                    if not nxt or nxt[0] == start or nxt[0] in seen:
                        break
                    prev, cur = cur, nxt[0]
                    loop.append(cur); seen.add(cur)
                loops.append(numpy.array([pos[k] for k in loop]))
            return max(loops, key=len)

        def _resample(P, m):
            """Resample a closed polyline to m points evenly spaced by arc length."""
            Q = numpy.vstack([P, P[:1]])
            arc = numpy.concatenate([[0], numpy.cumsum(numpy.linalg.norm(numpy.diff(Q, axis=0), axis=1))])
            tt = numpy.linspace(0, arc[-1], m, endpoint=False)
            return numpy.stack([numpy.interp(tt, arc, Q[:, k]) for k in range(3)], 1)

        def _perimeter(P):
            return float(numpy.sum(numpy.linalg.norm(numpy.diff(numpy.vstack([P, P[:1]]), axis=0), axis=1)))

        def build_band(n_mid):
            """
            Band from loop A (wall) to loop B (patch). n_mid = number of intermediate rings: height contours of
            the original wall (between rim_tol_mm and the target) plus rings on the target surface between the
            object's own contact outline and the patch outline.
            """
            rings = []
            if n_mid > 0:
                C0 = _level_loop(min(on_tol_mm, 0.5 * rim_tol_mm))      # the object's own contact outline
                # split the rings between the wall part and the part on the target, by their widths
                wall_w = numpy.median(numpy.linalg.norm(A_pts[:, None] - C0[None], axis=2).min(1)) if C0 is not None else 0
                flat_w = numpy.median(numpy.linalg.norm(B_pts[:, None] - C0[None], axis=2).min(1)) if C0 is not None else 0
                n_wall = int(round(n_mid * wall_w / max(wall_w + flat_w, 1e-9)))
                n_flat = n_mid - n_wall
                for k in range(1, n_wall + 1):                         # wall: top -> bottom
                    lvl = rim_tol_mm * (1 - k / (n_wall + 1))
                    if k == n_wall and n_flat >= 0:
                        lvl = max(lvl, 0.5 * on_tol_mm)
                    L = _level_loop(lvl)
                    if L is not None:
                        rings.append(L)
                if C0 is not None and n_flat > 0:                       # on the target: outline -> patch
                    rings.append(C0)
                    phis = None
                    for k in range(1, n_flat):
                        t = k / n_flat
                        m_k = max(int(round(((1 - t) * _perimeter(C0) + t * _perimeter(B_pts)) / h_ring)), 8)
                        Cs = _resample(C0, m_k)
                        # matching points on B by angle around the centre
                        def ang(P):
                            q = P - centre
                            return numpy.arctan2(q @ e2, q @ e1)
                        aB = ang(B_pts); o = numpy.argsort(aB)
                        Bs = numpy.stack([numpy.interp(ang(Cs), aB[o], B_pts[o][:, j], period=2 * numpy.pi)
                                          for j in range(3)], 1)
                        P, _ = closest_points_on_mesh(tgt, (1 - t) * Cs + t * Bs)
                        rings.append(P)
            # resample intermediate rings to ~h_ring spacing, all running the same way round as A and B
            rings = [_resample(R, max(int(round(_perimeter(R) / h_ring)), 8)) for R in rings]
            rings = [R if signed_area(R) > 0 else R[::-1] for R in rings]

            verts = [kept.vertices, patch.vertices]
            ids = [numpy.asarray(A_idx)]
            off = nk + len(patch.vertices)
            for R in rings:
                ids.append(off + numpy.arange(len(R)))
                verts.append(R)
                off += len(R)
            ids.append(nk + numpy.asarray(B_idx))
            all_v = numpy.vstack(verts)
            loops = [A_pts] + rings + [B_pts]
            band = []
            for Pa, Pb, ia, ib in zip(loops[:-1], loops[1:], ids[:-1], ids[1:]):
                for tri in _zipper_by_angle(Pa, Pb, centre, e1, e2):
                    band.append([ia[tri[1]] if tri[0] == "A" else ib[tri[1]],
                                 ia[tri[3]] if tri[2] == "A" else ib[tri[3]],
                                 ia[tri[5]] if tri[4] == "A" else ib[tri[5]]])
            return all_v, numpy.array(band)

        # number of intermediate rings from the widest part of the band
        gap = numpy.percentile(numpy.linalg.norm(
            A_pts[:, None, :] - B_pts[None, :, :], axis=2).min(axis=1), 90)
        n0 = max(int(numpy.ceil(gap / h_ring)) - 1, 0) if h_ring > 0 else 0
        n_mid_try = list(range(n0, -1, -1))                          # fewer rings as fallback, down to one layer
        verts, strip_faces = build_band(0)                          # single layer: valid by construction
        faces = numpy.vstack([kept.faces, patch.faces[:, ::-1] + nk, strip_faces])
        base_mesh = trimesh.Trimesh(verts, faces, process=False)
        trimesh.repair.fix_normals(base_mesh)
        n_fixed = len(kept.faces) + len(patch.faces)
        hits = _self_intersecting_pairs(base_mesh, numpy.arange(n_fixed, len(faces)))
        if not base_mesh.is_watertight or hits:
            raise RuntimeError(f"keep_wall: band between the outlines is invalid ({hits} intersections); "
                               f"try a larger rim_gap_mm or keep_wall=False")
        new = base_mesh
        if h_ring > 0:
            band_mask = numpy.zeros(len(faces), bool); band_mask[n_fixed:] = True
            # band faces as the refinement region; also the wall faces along A that were left large
            protected = nk + numpy.arange(len(patch.vertices))       # eyeball patch: never split its edges
            # new nodes go onto the part of the original surface that the band replaces: the removed faces,
            # minus everything the eyeball patch already covers (otherwise nodes land on top of the patch)
            removed = numpy.flatnonzero(contact_faces)
            _, d_rp = closest_points_on_mesh(patch, obj.triangles_center[removed])
            proj_mesh = obj.submesh([removed[d_rp > 0.5 * footprint_tol_mm]], append=True)
            project = lambda P: closest_points_on_mesh(proj_mesh, P)[0]
            ref, ref_band = _refine_band(base_mesh, band_mask, protected, project, max_edge=1.3 * h_ring)
            trimesh.repair.fix_normals(ref)
            hits_r = _self_intersecting_pairs(ref, numpy.flatnonzero(ref_band))
            if ref.is_watertight and hits_r == 0:
                new = ref
                print(f"   band refined to ~{h_ring:.2f} mm: {ref_band.sum()} triangles (was {len(strip_faces)})")
            else:
                print(f"   band refinement invalid ({hits_r} intersections), keeping the single layer")
        used_rings, used_mode = 0, "inside, wall kept"
    else:
        new = None

    # 2.-4. largest patch first; smaller ones only if the stitching is not valid
    for pmode in (patch_modes if not keep_wall else ()):
        sel_try = _select_patch(allowed_by_mode[pmode])
        if sel_try is None:
            print(f"   patch '{pmode}': not a clean disc, skipped")
            continue
        patch_try = tgt.submesh([sel_try], append=True)
        for rings in lift_rings:
            m, sf, hits = _stitch(patch_try, rings)
            if m is not None and hits == 0:
                new, strip_faces, used_rings, used_mode, sel, patch = m, sf, rings, pmode, sel_try, patch_try
                break
            print(f"   patch '{pmode}', lift rings {rings}: stitching invalid "
                  f"({'not closed' if hits < 0 else str(hits) + ' intersections'})")
        if new is not None:
            break
    if new is None:
        raise RuntimeError("Could not stitch the contact without self-intersections; check the contact region "
                           "(knife edges: use contact_fill_mm before the cut) or increase lift_rings")
    new.remove_unreferenced_vertices()
    if new.volume < 0:
        new.invert()

    # target ids of the contact nodes (patch vertices are exact copies of target vertices)
    from scipy.spatial import cKDTree
    _, tv = cKDTree(tgt.vertices).query(patch.vertices)
    info = dict(target_face_ids=sel, target_vertex_ids=numpy.unique(tv),
                contact_area_mm2=float(patch.area), strip_triangles=len(strip_faces),
                watertight=bool(new.is_watertight), lift_rings=used_rings, patch_mode=used_mode,
                strip_faces=strip_faces)
    print(f"   conforming contact: {len(sel)} eyeball faces / {len(info['target_vertex_ids'])} nodes "
          f"({patch.area:.2f} mm^2), patch '{used_mode}', {len(strip_faces)} stitching triangles "
          f"(lift rings {used_rings}), "
          f"watertight {new.is_watertight}")
    # faces of the result that are exactly target triangles (the conforming contact face)
    new = trimesh.Trimesh(numpy.asarray(new.vertices), numpy.asarray(new.faces), process=False)
    dvn, _ = cKDTree(numpy.asarray(tgt.vertices)).query(numpy.asarray(new.vertices))
    info["contact_face_mask"] = numpy.all((dvn < 1e-9)[numpy.asarray(new.faces)], axis=1)
    return new, info
