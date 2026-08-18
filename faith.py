# ============================================================
#  STOCHASTIC PROGRESSIVE PHOTON MAPPING (SPPM)
#  Mitsuba 3 + Dr.Jit — assembled from Phases 0–4
# ============================================================
from ntpath import realpath
import drjit as dr
import mitsuba as mi

mi.set_variant("cuda_ad_rgb")  # or 'llvm_ad_rgb'
dr.set_flag(dr.JitFlag.KernelHistory, True)

# ============================================================
#  SCENE (loaded FIRST so we can read its resolution)
# ============================================================
scene = mi.load_dict(mi.cornell_box())


sensor = scene.sensors()[0]
sampler = sensor.sampler()

# --- Pull image resolution from the sensor's film ------------
film = sensor.film()
crop = film.crop_size()  # ScalarVector2u (width, height)
width, height = int(crop.x), int(crop.y)  # Python ints
N = width * height

# --- Sampler must know the wavefront size (= one ray per pixel)
sampler.seed(0, wavefront_size=N)
sampler.set_samples_per_wavefront(1)

# ============================================================
#  CONFIG
# ============================================================
photons_per_pass = 5_000_000
max_bounces = 8
num_iterations = 32  # 256
initial_radius = 0.005
alpha = 2.0 / 3.0
cell_size = initial_radius
num_buckets = 1 << 22
P1, P2, P3 = 73856093, 19349663, 83492791

# ============================================================
#  SCENE + SAMPLER
# ============================================================

ph_sampler = mi.load_dict({"type": "independent"})

# environment emitter (None if the scene has none)
env_emitter = scene.environment()


def eval_emission(si, ray, observed):
    """Radiance emitted toward the camera along -ray.d for newly observed
    ray results: self-emission Le at surface hits, environment radiance
    on misses. `observed` masks lanes whose ray result is new this bounce.
    """
    hit = observed & si.is_valid()
    le = dr.select(hit, si.emitter(scene).eval(si, active=hit), mi.Color3f(0.0))
    if env_emitter is not None:
        miss = observed & ~si.is_valid()
        # env emitters evaluate a SurfaceInteraction with wi = -ray.d
        si_env = dr.zeros(mi.SurfaceInteraction3f, dr.width(ray))
        si_env.wi = -ray.d
        le += dr.select(miss, env_emitter.eval(si_env, active=miss), mi.Color3f(0.0))
    return le


# ============================================================
#  PERSISTENT PER-PIXEL STATE  (lives across ALL iterations)
# ============================================================

# vp - visible point
# pixel id (stable index), literally N = width * height.
# literally 0,1,2,3,...,N-1
# to remember order of pixels after hash grid sort
vp_pix = dr.arange(mi.UInt32, N)
# visible point position (where they land after path tracing)
vp_pos = dr.zeros(mi.Point3f, N)
# visible point normal
vp_nrm = dr.zeros(mi.Normal3f, N)
# accumulated photon energy Φ (getting this is why we have vp in the first place)
vp_flux = dr.zeros(mi.Color3f, N)
# accumulated photon count n at a vp (damped, so Float)
vp_cnt = dr.zeros(mi.Float, N)
# current search radius R (to know how close vp to choose)
vp_rad = dr.full(mi.Float, initial_radius, N)
# does this pixel have a visible point? (???, probably for points that did not arrive at diffuse)
vp_valid = dr.zeros(mi.Bool, N)
# photons deposited this pass (per pixel, reset each iteration)
vp_pass = dr.zeros(mi.UInt32, N)
# diffuse reflectance (albedo) at the visible point
vp_albedo = dr.zeros(mi.Color3f, N)
# specular throughput along the camera path up to the visible point;
# weights this iteration's photon deposits so each pass is weighted by
# its own camera path (not the final one)
vp_throughput = dr.zeros(mi.Color3f, N)
# emitted/environment radiance seen along the camera path, accumulated
# across iterations (PBRT's Ld / Mitsuba's gp.emission term); averaged
# by num_iterations in the final estimate
vp_emission = dr.zeros(mi.Color3f, N)


# ============================================================
#  PHASE 1: CAMERA PASS — trace one ray per pixel to its first
#           diffuse hit; that hit IS the pixel's visible point.
# ============================================================
@dr.syntax
def camera_pass():
    idx = dr.arange(mi.UInt32, N)  # isn't this equal to vp_pix?
    x = idx % width
    y = idx // width
    # sample_ray expects fractional pixel coords in [0, 1] relative to crop
    pos2 = mi.Point2f(
        mi.Float(x) / width, mi.Float(y) / height
    ) + sampler.next_2d() / mi.Point2f(width, height)

    # camera does it's shenanigans to determine where rays are cast
    # this is where randomness comes from in rendering
    ray, _ = sensor.sample_ray(
        mi.Float(0), sampler.next_1d(), pos2, sampler.next_2d(), active=mi.Bool(True)
    )

    # si = scene_intersection
    si = scene.ray_intersect(ray)
    # what does this validate?
    valid = si.is_valid()
    # emitted/environment radiance along the camera path (throughput is 1
    # for the primary ray); accumulated per pass, weighted by throughput
    emission = eval_emission(si, ray, mi.Bool(True))
    # ???
    si_vp = si
    hit_diffuse = mi.Bool(False)
    # bounce count
    i = mi.UInt32(0)
    # diffuse reflectance (albedo) at the vp, and the specular throughput
    # accumulated along the camera path up to that point
    albedo_vp = mi.Color3f(0.0)
    throughput_vp = mi.Color3f(0.0)
    throughput = mi.Color3f(1.0)

    while valid & (i < max_bounces):
        bsdf = si.bsdf()
        is_diffuse = mi.has_flag(bsdf.flags(), mi.BSDFFlags.Diffuse)

        rec = valid & is_diffuse
        hit_diffuse = hit_diffuse | rec
        si_vp = dr.select(rec, si, si_vp)
        albedo_vp = dr.select(
            rec, bsdf.eval_diffuse_reflectance(si, active=rec), albedo_vp
        )
        throughput_vp = dr.select(rec, throughput, throughput_vp)

        active = valid & ~is_diffuse & (i < max_bounces - 1)
        bs, val = bsdf.sample(
            mi.BSDFContext(), si, sampler.next_1d(), sampler.next_2d(), active
        )
        throughput = dr.select(active, throughput * val, throughput)
        ray = si.spawn_ray(si.to_world(bs.wo))
        si = scene.ray_intersect(ray, active)
        # emission at newly observed hits/escapes, weighted by the
        # throughput of the path up to the previous surface
        emission += throughput * eval_emission(si, ray, active)
        valid = active & si.is_valid()
        i += 1

    return (
        dr.select(hit_diffuse, si_vp.p, mi.Point3f(dr.inf)),
        dr.select(hit_diffuse, si_vp.n, mi.Normal3f(0)),
        hit_diffuse,
        dr.select(hit_diffuse, albedo_vp, mi.Color3f(0.0)),
        dr.select(hit_diffuse, throughput_vp, mi.Color3f(0.0)),
        # NOT masked by hit_diffuse: pixels without a visible point still
        # keep the emission seen along their camera path (PBRT does the same)
        emission,
    )


print("phase2")


# ============================================================
#  PHASE 2: BUILD SPATIAL HASH GRID (sort visible points by bucket)
# ============================================================
def build_grid(vp_pos, vp_nrm, vp_pix, vp_flux, vp_cnt, vp_rad, vp_valid):
    valid_idx = dr.compress(vp_valid)  # M active points
    pos = dr.gather(mi.Point3f, vp_pos, valid_idx)
    nrm = dr.gather(mi.Normal3f, vp_nrm, valid_idx)
    pix = dr.gather(mi.UInt32, vp_pix, valid_idx)
    flux = dr.gather(mi.Color3f, vp_flux, valid_idx)
    cnt = dr.gather(mi.Float, vp_cnt, valid_idx)
    rad = dr.gather(mi.Float, vp_rad, valid_idx)

    cx = mi.UInt32(dr.floor(pos.x / cell_size))
    cy = mi.UInt32(dr.floor(pos.y / cell_size))
    cz = mi.UInt32(dr.floor(pos.z / cell_size))
    bucket = ((cx * P1) ^ (cy * P2) ^ (cz * P3)) & (num_buckets - 1)

    order = dr.argsort(bucket)
    b = dr.gather(mi.UInt32, bucket, order)
    s_pos = dr.gather(mi.Point3f, pos, order)
    s_nrm = dr.gather(mi.Normal3f, nrm, order)
    s_pix = dr.gather(mi.UInt32, pix, order)
    s_rad = dr.gather(mi.Float, rad, order)

    # flux/cnt are scatter targets: we keep the persistent vp_* as targets,
    # so only the lookup arrays are returned here.
    return b, s_pos, s_nrm, s_pix, s_rad


print("phase3")


# ============================================================
#  PHASE 3: PHOTON PASS — emit photons, bounce with Russian
#           roulette, scatter flux at every diffuse hit.
# ============================================================
def hash_cell(x, y, z):
    return ((x * P1) ^ (y * P2) ^ (z * P3)) & (num_buckets - 1)


def photon_pass(iteration, b, s_pos, s_nrm, s_pix, s_rad, M):
    ph_sampler.seed(iteration, wavefront_size=photons_per_pass)

    ray, weight, _ = scene.sample_emitter_ray(
        mi.Float(0),
        ph_sampler.next_1d(),
        ph_sampler.next_2d(),
        ph_sampler.next_2d(),
        mi.Bool(True),
    )
    ph_flux = weight

    # Trace photons, recording a hit at EVERY diffuse surface interaction.
    # After a diffuse hit the photon continues via Russian roulette
    # (survival probability = mean diffuse reflectance), so indirect
    # illumination is captured without exploding the path count.
    rec_pos = []
    rec_nrm = []
    rec_flux = []
    rec_hit = []

    si = scene.ray_intersect(ray)
    alive = si.is_valid()

    for bounce in range(max_bounces):
        bsdf = si.bsdf()
        is_diffuse = mi.has_flag(bsdf.flags(), mi.BSDFFlags.Diffuse)

        # Record the photon *before* scattering this surface (arriving flux).
        rec = alive & is_diffuse
        rec_pos.append(dr.select(rec, si.p, mi.Point3f(0.0)))
        rec_nrm.append(dr.select(rec, si.n, mi.Normal3f(0.0)))
        rec_flux.append(dr.select(rec, ph_flux, mi.Color3f(0.0)))
        rec_hit.append(rec)

        # Sample a scattering direction for the next bounce.
        can = alive & (bounce < max_bounces - 1)
        bs, val = bsdf.sample(
            mi.BSDFContext(), si, ph_sampler.next_1d(), ph_sampler.next_2d(), can
        )

        # Russian roulette on diffuse surfaces (reflect w.p. = mean albedo).
        rr = ph_sampler.next_1d()
        refl = bsdf.eval_diffuse_reflectance(si, active=can & is_diffuse)
        q = dr.clip(dr.mean(refl), 0.0, 1.0)
        rr_continue = rr < q
        will_continue = can & dr.select(is_diffuse, rr_continue, mi.Bool(True))

        # Specular: weight is just the BSDF sample weight.
        # Diffuse: divide by q to compensate for the roulette absorption.
        mult = dr.select(is_diffuse, val / dr.maximum(q, 1e-8), val)
        ph_flux = dr.select(will_continue, ph_flux * mult, ph_flux)

        ray = si.spawn_ray(si.to_world(bs.wo))
        si = scene.ray_intersect(ray, will_continue)
        alive = will_continue & si.is_valid()

    # Flatten the per-bounce records, then compact the valid ones.
    total = photons_per_pass * max_bounces
    flat_pos = dr.zeros(mi.Point3f, total)
    flat_nrm = dr.zeros(mi.Normal3f, total)
    flat_flux = dr.zeros(mi.Color3f, total)
    flat_hit = dr.zeros(mi.Bool, total)

    for bounce in range(max_bounces):
        idx = mi.UInt32(bounce) * photons_per_pass + dr.arange(
            mi.UInt32, photons_per_pass
        )
        dr.scatter(flat_pos, rec_pos[bounce], idx)
        dr.scatter(flat_nrm, rec_nrm[bounce], idx)
        dr.scatter(flat_flux, rec_flux[bounce], idx)
        dr.scatter(flat_hit, rec_hit[bounce], idx)

    hit_idx = dr.compress(flat_hit)
    ph_pos = dr.gather(mi.Point3f, flat_pos, hit_idx)
    ph_nrm = dr.gather(mi.Normal3f, flat_nrm, hit_idx)
    ph_flux = dr.gather(mi.Color3f, flat_flux, hit_idx)
    K = dr.width(ph_pos)

    # --- hash photons, probe 27 neighbor cells ------------------
    pcx = mi.UInt32(dr.floor(ph_pos.x / cell_size))
    pcy = mi.UInt32(dr.floor(ph_pos.y / cell_size))
    pcz = mi.UInt32(dr.floor(ph_pos.z / cell_size))

    offsets = [
        (dx, dy, dz) for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)
    ]
    for dx, dy, dz in offsets:
        nb = hash_cell(
            mi.UInt32(mi.Int32(pcx) + dx),
            mi.UInt32(mi.Int32(pcy) + dy),
            mi.UInt32(mi.Int32(pcz) + dz),
        )

        lo = dr.binary_search(
            0, M, lambda idx: dr.gather(mi.UInt32, b, idx, active=idx < M) < nb
        )
        hi = dr.binary_search(
            0, M, lambda idx: dr.gather(mi.UInt32, b, idx, active=idx < M) <= nb
        )

        def body(idx):
            active = idx < hi
            c_pos = dr.gather(mi.Point3f, s_pos, idx, active)
            c_nrm = dr.gather(mi.Normal3f, s_nrm, idx, active)
            c_rad = dr.gather(mi.Float, s_rad, idx, active)
            c_pix = dr.gather(mi.UInt32, s_pix, idx, active)
            # weight this photon's flux by the visible point's specular
            # throughput, so each pass is weighted by its own camera path
            c_tp = dr.gather(mi.Color3f, vp_throughput, c_pix, active)
            # diffuse reflectance rho_d at the visible point: f_r = rho_d / pi.
            # We deposit rho_d here and the 1/pi is folded into the pi^2 of
            # the final radiance estimate (one pi for the disk area, one for
            # the Lambertian BRDF).
            c_alb = dr.gather(mi.Color3f, vp_albedo, c_pix, active)

            d2 = dr.squared_norm(c_pos - ph_pos)
            # normal-based rejection: only accept photons on the same side of
            # the surface as the visible point, which stops light from leaking
            # through thin geometry (opposite faces have opposite normals)
            same_side = dr.dot(c_nrm, ph_nrm) > 0
            hit = active & (d2 <= c_rad * c_rad) & same_side

            dr.scatter_add(
                vp_flux, c_tp * c_alb * ph_flux / photons_per_pass, c_pix, hit
            )
            dr.scatter_add(vp_pass, mi.UInt32(1), c_pix, hit)
            return (idx + 1,)

        dr.while_loop(state=(lo,), cond=lambda idx: idx < hi, body=body)

    return K


print("phase4")


# ============================================================
#  PHASE 4: RADIUS SHRINK  (SPPM's convergence step)
# ============================================================
def shrink_radii(K):
    global vp_rad, vp_cnt, vp_pass, vp_flux
    # SPPM convergence step (Hachisuka & Jensen):
    #   gamma = (N + a*M) / (N + M)
    #   R   <- R * sqrt(gamma)
    #   tau <- tau * gamma
    #   N   <- N + a*M
    gamma = (vp_cnt + alpha * vp_pass) / (vp_cnt + vp_pass)
    update = vp_pass > 0
    vp_rad *= dr.select(update, dr.sqrt(gamma), mi.Float(1.0))
    vp_flux *= dr.select(update, gamma, mi.Float(1.0))

    vp_cnt += alpha * vp_pass  # damped accumulator, not full M
    vp_pass = dr.zeros(mi.UInt32, N)  # reset for next iteration


def to_img_rgb(v):  # mi.Color3f, width N  -> (H, W, 3) tensor
    return dr.reshape(mi.TensorXf(dr.ravel(v, order="F")), (height, width, 3))


def to_img_1(v):  # mi.Float/mi.UInt, width N -> (H, W) tensor
    return dr.reshape(mi.TensorXf(v), (height, width))


# Print the scene size to compare with initial_radius
sph = scene.bbox().bounding_sphere()
dr.print("scene radius: {}", sph.radius)

print("mainloop")

# ============================================================
#  MAIN LOOP
# ============================================================
for iteration in range(num_iterations):
    print("phase1")
    # Phase 1 — refresh visible points
    vp_pos, vp_nrm, vp_valid, vp_albedo, vp_throughput, pass_emission = camera_pass()
    vp_emission += pass_emission
    dr.eval(vp_pos, vp_nrm, vp_valid, vp_throughput, vp_emission)

    dr.print("phase2")
    # Phase 2 — rebuild the grid (visible points moved)
    b, s_pos, s_nrm, s_pix, s_rad = build_grid(
        vp_pos, vp_nrm, vp_pix, vp_flux, vp_cnt, vp_rad, vp_valid
    )
    M = dr.width(b)

    dr.print("phase3")
    # Phase 3 — scatter this iteration's photons into the pixels
    K = photon_pass(iteration, b, s_pos, s_nrm, s_pix, s_rad, M)
    dr.eval(vp_flux, vp_cnt)

    # scale = dr.max(vp_cnt)
    # mi.Bitmap(to_img_rgb(mi.Color3f(mi.Float(vp_cnt))) * 4 / scale).write("dbg_cnt.exr")
    # dr.print("photons deposited total: {}", dr.sum(vp_cnt))
    # dr.print("photons recorded this pass K: {}", K)

    dr.print("phase4")

    # Phase 4 — shrink radii using the accumulated count
    shrink_radii(K)
    # dr.print(
    #     "dr.sum(vp_flux)={} dr.sum(vp_cnt)={} ratio={}",
    #     dr.sum(dr.sum(vp_flux)),
    #     dr.sum(vp_cnt),
    #     dr.sum(dr.sum(vp_flux)) / dr.sum(vp_cnt),
    # )

    if iteration == 0:
        # run ONE iteration first to JIT-compile everything (warmup), then:
        dr.kernel_history_clear()

# ============================================================
#  FINAL IMAGE  (radiance estimate: flux / disk area)
# ============================================================
# PBRT-style final estimate: L = Ld / i + tau / (N_total * pi * R^2).
# vp_emission is the sum of per-pass camera-path emission (Ld); vp_flux is
# SPPM's tau with the per-pass 1/photons_per_pass already folded in, and
# rho_d = pi * f_r deposited per pass, hence the pi^2 (disk area * BRDF).
radiance = (vp_emission + vp_flux / (dr.pi * dr.pi * vp_rad * vp_rad)) / num_iterations
# radiance = vp_flux / (dr.pi * vp_rad * vp_rad * photons_per_pass)
# radiance = dr.linear_to_srgb(radiance / (1.0 + radiance))
# radiance /= dr.max(dr.max(radiance))

# radiance = dr.select(vp_valid, radiance, mi.Color3f(0.0))


display = radiance  # dr.linear_to_srgb(radiance / (1.0 + radiance))

print("denoiser")

img = mi.Bitmap(to_img_rgb(display))

if False:
    denoiser = mi.OptixDenoiser(
        input_size=(width, height), albedo=False, normals=False, temporal=False
    )
    img = denoiser(img)


img.convert(mi.Bitmap.PixelFormat.RGB, mi.Struct.Type.UInt8, True).write("sppm.png")
dr.eval(radiance)

# img = mi.render(scene, spp=64)
# mi.util.write_bitmap("ref.exr", img)


# ref = mi.render(scene, spp=16)  # built-in path tracer, correct units
# print("ref mean:", dr.mean(ref), "ref max:", dr.max(ref))
# print(
#     "my  mean:",
#     dr.mean(mi.TensorXf(radiance)),
#     "my  max:",
#     dr.max(mi.TensorXf(radiance)),
# )
# print(
#     "tonemaped mean:",
#     dr.mean(mi.TensorXf(display)),
#     "tonemapped max:",
#     dr.max(mi.TensorXf(display)),
# )


hist = dr.kernel_history()

total = sum(k["execution_time"] for k in hist)
print(f"{total:.3f} ms across {len(hist)} operations(s)")
