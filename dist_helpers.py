# ### Generate Purturbations to Particles

# In[6]:

import matplotlib.pyplot as plt
import numpy as np
import h5py as h5

def uniform_initial(n):
    x = np.random.uniform(-1, 1, n)
    y = np.random.uniform(-1, 1, n)
    z = np.random.uniform(-1, 1, n)
    return x, y, z

def normal_initial(n):
    x = np.random.normal(size=n)
    y = np.random.normal(size=n)
    z = np.random.normal(size=n)
    return x, y, z

def rectangular_initial(n):
    x = np.random.uniform(-0.1, 0.1, n)
    y = np.random.uniform(-0.1, 0.1, n)
    z = np.random.uniform(-1, 1, n)
    return x, y, z

def pancake_initial(n):
    x = np.random.uniform(-1, 1, n)
    y = np.random.uniform(-1, 1, n)
    z = np.random.uniform(-0.1, 0.1, n)
    return x, y, z
# In[2]:


def plot_3d(x,y,z):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(
        x,y,z,
        s=5,              # marker size
        alpha=0.6,
        # cmap='viridis',
        linewidths=0,     # no edge lines, cleaner look
    )

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Particle Distribution')

    max_range = np.ptp(np.column_stack((x,y,z)), axis=0).max() / 2  # ptp = peak to peak (max - min)
    mid = np.mean(np.column_stack((x,y,z)), axis=0)
    ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
    ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
    ax.set_zlim(mid[2] - max_range, mid[2] + max_range)

    plt.tight_layout()
    plt.show()

def perturb(sz,scale):
    x = np.random.uniform(low=-scale, high=scale, size=sz)
    y = np.random.uniform(low=-scale, high=scale, size=sz)
    z = np.random.uniform(low=-scale, high=scale, size=sz)
    return x, y, z


def _format_particle_count(n):
    n = int(n)
    if n >= 1_000_000:
        val = n / 1_000_000
        if val == int(val):
            return f"{int(val)}m"
        return f"{val:.1f}m".replace('.', 'p')
    val = n / 1_000
    if val >= 1 and val == int(val):
        return f"{int(val)}k"
    if val >= 1:
        return f"{val:.1f}k".replace('.', 'p')
    return str(n)


def _format_scale(scale):
    exp = -np.log10(scale)
    if abs(exp - round(exp)) < 1e-9:
        return f"em{int(round(exp))}"
    return f"em{exp:.2f}".replace('.', 'p').replace('-', 'n')


def _rotation_matrix(rx, ry, rz):
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def _dist_name(func):
    name = func.__name__
    if name.endswith('_initial'):
        name = name[:-len('_initial')]
    return name


def _format_rotation(rot):
    degs = [np.degrees(r) for r in rot]
    parts = []
    for axis, d in zip(('rx', 'ry', 'rz'), degs):
        if d != 0:
            s = f"{d:g}".replace('.', 'p').replace('-', 'n')
            parts.append(f"{axis}{s}")
    return '_'.join(parts)


def generate_dists(generators, n_particles, scale_factors, rotations=None, out_of_bounds='truncate'):
    """Generate all permutations of distributions from the given parameter arrays.

    Parameters
    ----------
    generators : list of callables
        Distribution generator functions, each with signature f(n) -> (x, y, z).
    n_particles : list of int/float
        Number of particles for each configuration (e.g. 1e4, 1e5).
    scale_factors : list of float
        Perturbation scale factors (e.g. 0.001, 0.01, 0.1).
    rotations : list of (float, float, float), optional
        Per-axis rotation angles in radians. Each entry is (rx, ry, rz).
        Defaults to [(0, 0, 0)] (no rotation).
    out_of_bounds : str, optional
        How to handle particles outside [-1, 1] after rotation.
        'truncate' clips coordinates, 'remove' discards particles entirely.
    """
    if rotations is None:
        rotations = [(0, 0, 0)]

    n_gen = len(generators)
    n_np = len(n_particles)
    n_sf = len(scale_factors)
    n_rot = len(rotations)
    total = n_gen * n_np * n_sf * n_rot
    print(f"Generating {total} distributions "
          f"({n_gen} generators x {n_np} particle counts x {n_sf} scales x {n_rot} rotations)")

    has_nontrivial_rot = any(any(r != 0 for r in rot) for rot in rotations)
    dist = []
    count = 0

    for gen in generators:
        dname = _dist_name(gen)
        for scale in scale_factors:
            scale_str = _format_scale(scale)
            for rot in rotations:
                for n in n_particles:
                    n_int = int(n)
                    n_str = _format_particle_count(n)
                    rx, ry, rz = rot

                    count += 1
                    rot_str = ''
                    if has_nontrivial_rot and any(r != 0 for r in rot):
                        rot_str = f"_{_format_rotation(rot)}"
                    name = f"{dname}{rot_str}_{scale_str}_n{n_str}"
                    print(f"  [{count}/{total}] {name} ...", end=' ', flush=True)

                    x, y, z = gen(n_int)

                    is_rotated = any(r != 0 for r in rot)
                    if is_rotated:
                        R = _rotation_matrix(rx, ry, rz)
                        coords = R @ np.stack([x, y, z])
                        x, y, z = coords[0], coords[1], coords[2]

                        if out_of_bounds == 'truncate':
                            x = np.clip(x, -1, 1)
                            y = np.clip(y, -1, 1)
                            z = np.clip(z, -1, 1)
                        elif out_of_bounds == 'remove':
                            mask = (np.abs(x) <= 1) & (np.abs(y) <= 1) & (np.abs(z) <= 1)
                            x, y, z = x[mask], y[mask], z[mask]

                    pt = perturb(len(x), scale)
                    dist.append((name, (x, y, z), pt))
                    print("done", flush=True)

    return dist


# In[7]:


def plot_perturbations(ix,iy,iz,px,py,pz):
    fig = plt.figure(figsize=(12, 6))

    ax1 = fig.add_subplot(121, projection='3d')
    ax1.scatter(ix, iy, iz, s=5, alpha=0.6)
    ax1.set_title('Initial Distribution')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')

    ax2 = fig.add_subplot(122, projection='3d')
    ax2.scatter(px, py, pz, s=5, alpha=0.6)
    ax2.set_title('Perturbed Distribution')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')

    plt.tight_layout()
    plt.show()

