import config
import numpy as np
from collections import deque
from PIL import Image
from scipy.ndimage import distance_transform_edt


def generate_province_map(main_layout):
    main_layout.progress.setVisible(True)
    main_layout.progress.setValue(10)

    boundary_image = main_layout.boundary_image_display.get_image()
    land_image = main_layout.land_image_display.get_image()

    if boundary_image is None and land_image is None:
        raise ValueError("Need at least boundary OR land image.")

    terrain_display = getattr(main_layout, "terrain_image_display", None)
    terrain_image = terrain_display.get_image() if terrain_display else None

    terrain_mask = (
        generate_terrain_mask(terrain_image)
        if terrain_image is not None
        else None
    )

    # ---------- BOUNDARY ----------
    if boundary_image is not None:
        b_arr = np.array(boundary_image, copy=False)
        r, g, b = config.BOUNDARY_COLOR
        boundary_mask = (
            (b_arr[..., 0] == r) &
            (b_arr[..., 1] == g) &
            (b_arr[..., 2] == b)
        )
        map_h, map_w = boundary_mask.shape
    else:
        boundary_mask = None

    # ---------- LAND / SEA ----------
    if land_image is not None:
        o_arr = np.array(land_image, copy=False)
        sea_mask = is_sea_color(o_arr)
        land_mask = ~sea_mask
        if boundary_mask is None:
            map_h, map_w = sea_mask.shape
    else:
        sea_mask = np.zeros((map_h, map_w), bool)
        land_mask = np.ones((map_h, map_w), bool)

    # Проверка terrain
    if terrain_mask is not None:
        if terrain_mask.shape != land_mask.shape:
            raise ValueError("Terrain map size must match land map size")

    # ---------- MASK LOGIC ----------
    if boundary_mask is None:
        land_fill, land_border = land_mask, sea_mask
        sea_fill, sea_border = sea_mask, land_mask
    else:
        land_fill = land_mask & ~boundary_mask
        land_border = boundary_mask | sea_mask
        sea_fill = sea_mask & ~boundary_mask
        sea_border = boundary_mask | land_mask

    # ---------- GENERATION ----------
    land_points = main_layout.land_slider.value()
    sea_points = main_layout.ocean_slider.value()

    land_map, land_meta, next_id = create_province_map(
        land_fill, land_border, land_points, 0, "land", terrain_mask
    )

    main_layout.progress.setValue(50)

    if sea_points > 0:
        sea_map, sea_meta, _ = create_province_map(
            sea_fill, sea_border, sea_points, next_id, "ocean", None
        )
    else:
        sea_map = np.full((map_h, map_w), -1, np.int32)
        sea_meta = []

    metadata = land_meta + sea_meta

    province_image = combine_maps(
        land_map, sea_map, metadata, land_mask, sea_mask
    )

    main_layout.province_image_display.set_image(province_image)
    main_layout.province_data = metadata

    main_layout.progress.setValue(100)
    main_layout.button_exp_prov_img.setEnabled(True)
    main_layout.button_exp_prov_csv.setEnabled(True)

    return province_image, metadata


# ---------- UTILITIES ----------

def is_sea_color(arr):
    r, g, b = config.OCEAN_COLOR
    return (arr[..., 0] == r) & (arr[..., 1] == g) & (arr[..., 2] == b)


def _color_from_id(pid):
    rng = np.random.default_rng(pid + 1)
    c = rng.integers(40, 256, 3)
    return int(c[0]), int(c[1]), int(c[2])


def generate_jitter_seeds(mask, num_points):
    if num_points <= 0:
        return []

    h, w = mask.shape
    grid = max(1, int(np.sqrt(num_points)))
    rng = np.random.default_rng()
    seeds = []

    for gy in range(grid):
        for gx in range(grid):
            y0, y1 = int(gy * h / grid), int((gy + 1) * h / grid)
            x0, x1 = int(gx * w / grid), int((gx + 1) * w / grid)
            cell = mask[y0:y1, x0:x1]
            ys, xs = np.where(cell)
            if xs.size:
                i = rng.integers(xs.size)
                seeds.append((x0 + xs[i], y0 + ys[i]))

    return seeds


def create_province_map(fill_mask, border_mask, num_points, start_id, ptype, terrain_mask):
    if num_points <= 0 or not fill_mask.any():
        return np.full(fill_mask.shape, -1, np.int32), [], start_id

    seeds = generate_jitter_seeds(fill_mask, num_points)
    pmap, meta = flood_fill(fill_mask, seeds, start_id, ptype, terrain_mask)
    assign_borders(pmap, border_mask)
    finalize_metadata(meta)

    next_id = max(meta.keys()) + 1 if meta else start_id
    return pmap, list(meta.values()), next_id


def flood_fill(fill_mask, seeds, start_id, ptype, terrain_mask):
    h, w = fill_mask.shape
    pmap = np.full((h, w), -1, np.int32)
    meta = {}
    q = deque()

    for i, (sx, sy) in enumerate(seeds):
        pid = start_id + i
        pmap[sy, sx] = pid
        r, g, b = _color_from_id(pid)

        terrain_counts = {}

        if terrain_mask is not None:
            t = terrain_mask[sy, sx]
            terrain_counts[t] = 1

        meta[pid] = {
            "province_id": pid,
            "province_type": ptype,
            "terrain_counts": terrain_counts,
            "R": r, "G": g, "B": b,
            "sum_x": sx, "sum_y": sy, "count": 1
        }

        q.append((sx, sy, pid))

    while q:
        x, y, pid = q.popleft()
        d = meta[pid]

        for dx, dy in [(1,0),(-1,0),(0,1),(0,-1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < w and 0 <= ny < h:
                if pmap[ny, nx] == -1 and fill_mask[ny, nx]:
                    pmap[ny, nx] = pid
                    d["sum_x"] += nx
                    d["sum_y"] += ny
                    d["count"] += 1

                    if terrain_mask is not None:
                        t = terrain_mask[ny, nx]
                        d["terrain_counts"][t] = d["terrain_counts"].get(t, 0) + 1

                    q.append((nx, ny, pid))

    return pmap, meta



def assign_borders(pmap, border_mask):
    if not border_mask.any():
        return
    valid = pmap >= 0
    _, (ny, nx) = distance_transform_edt(~valid, return_indices=True)
    pmap[border_mask] = pmap[ny[border_mask], nx[border_mask]]


def finalize_metadata(meta):
    for d in meta.values():
        d["x"] = d["sum_x"] / d["count"]
        d["y"] = d["sum_y"] / d["count"]

        if d["province_type"] == "ocean":
            d["terrain"] = "Ocean"
        else:
            terrain_counts = d.get("terrain_counts")
            d["terrain"] = max(terrain_counts, key=terrain_counts.get) if terrain_counts else "Unknown"

        del d["sum_x"], d["sum_y"], d["count"]
        del d["terrain_counts"]




def combine_maps(land_map, sea_map, metadata, land_mask, sea_mask):
    h, w = land_map.shape
    combined = np.full((h, w), -1, np.int32)

    combined[(land_map >= 0) & land_mask] = land_map[(land_map >= 0) & land_mask]
    combined[(sea_map >= 0) & sea_mask] = sea_map[(sea_map >= 0) & sea_mask]

    _, (ny, nx) = distance_transform_edt(combined < 0, return_indices=True)
    combined[combined < 0] = combined[ny[combined < 0], nx[combined < 0]]

    out = np.zeros((h, w, 3), np.uint8)
    lut = {d["province_id"]: (d["R"], d["G"], d["B"]) for d in metadata}
    for pid, color in lut.items():
        out[combined == pid] = color

    return Image.fromarray(out)


def generate_terrain_mask(image):
    arr = np.array(image, copy=False)
    mask = np.full(arr.shape[:2], "Unknown", dtype=object)
    for rgb, terrain in config.TERRAIN_COLORS.items():
        match = (arr[..., :3] == rgb).all(axis=2)
        mask[match] = terrain
    return mask
