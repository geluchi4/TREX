#!/usr/bin/env python3
"""
Convierte un fichero LAZ/LAS a OBJ con textura PNG real (proyeccion cenital).
Genera 3 ficheros: .obj + .mtl + .png
Uso: python laz_to_obj.py fichero.laz [salida.obj]
Requiere: pip install laspy[lazrs] numpy scipy pillow
"""

import sys
import math
import numpy as np
import laspy
from scipy.spatial import Delaunay
from PIL import Image

TEX_SIZE = 2048  # Resolución de la textura (2048x2048)

def laz_to_obj(input_path, output_path=None):
    if output_path is None:
        output_path = input_path.rsplit(".", 1)[0] + ".obj"
    base = output_path.rsplit(".", 1)[0]
    mtl_path = base + ".mtl"
    tex_path = base + ".png"
    tex_name = tex_path.split("\\")[-1].split("/")[-1]
    mtl_name = base.split("\\")[-1].split("/")[-1] + ".mtl"

    print(f"[1/5] Leyendo {input_path}...")
    las = laspy.read(input_path)

    x = np.array(las.x)
    y = np.array(las.y)
    z = np.array(las.z)

    has_color = False
    try:
        r = np.array(las.red,   dtype=np.float32)
        g = np.array(las.green, dtype=np.float32)
        b = np.array(las.blue,  dtype=np.float32)
        if r.max() > 1.0:
            r /= 65535.0
            g /= 65535.0
            b /= 65535.0
        has_color = True
        print(f"      Color RGB real encontrado")
    except Exception:
        print(f"      Sin RGB, usando color por altura")

    print(f"      {len(x):,} puntos cargados")

    # Guardar rango original para UV
    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()

    # Centrar en origen
    cx, cy, cz = x.mean(), y.mean(), z.min()
    x -= cx
    y -= cy
    z -= cz

    MAX_POINTS = 500_000
    if len(x) > MAX_POINTS:
        print(f"[2/5] Submuestreando a {MAX_POINTS:,} puntos...")
        idx = np.random.choice(len(x), MAX_POINTS, replace=False)
        x, y, z = x[idx], y[idx], z[idx]
        if has_color:
            r, g, b = r[idx], g[idx], b[idx]
        x_orig = np.array(las.x)[idx]
        y_orig = np.array(las.y)[idx]
    else:
        print(f"[2/5] Preparando {len(x):,} puntos...")
        x_orig = np.array(las.x)
        y_orig = np.array(las.y)

    if not has_color:
        z_norm = (z - z.min()) / (z.max() - z.min() + 1e-6)
        r = 0.3 + 0.5 * z_norm
        g = 0.25 + 0.3 * z_norm
        b = 0.1 * np.ones_like(z_norm)

    print("[3/5] Triangulando...")
    tri = Delaunay(np.stack([x, y], axis=-1))
    pts = np.stack([x, y, z], axis=-1)
    simplices = tri.simplices
    v0 = pts[simplices[:, 0]]
    v1 = pts[simplices[:, 1]]
    v2 = pts[simplices[:, 2]]
    edge_max = np.maximum(
        np.linalg.norm(v1 - v0, axis=1),
        np.maximum(np.linalg.norm(v2 - v1, axis=1), np.linalg.norm(v0 - v2, axis=1))
    )
    threshold = np.percentile(edge_max, 95)
    simplices = simplices[edge_max < threshold]

    print("[4/5] Generando textura cenital PNG...")
    # UV basado en posición real X/Y -> píxel en textura
    x_range = x_max - x_min + 1e-6
    y_range = y_max - y_min + 1e-6
    u = (x_orig - x_min) / x_range          # 0..1
    v_uv = 1.0 - (y_orig - y_min) / y_range  # 0..1 (invertir Y)

    # Crear imagen
    pixels = np.zeros((TEX_SIZE, TEX_SIZE, 3), dtype=np.uint8)
    px_col = (u * (TEX_SIZE - 1)).astype(np.int32)
    px_row = (v_uv * (TEX_SIZE - 1)).astype(np.int32)
    ri = (np.clip(r, 0, 1) * 255).astype(np.uint8)
    gi = (np.clip(g, 0, 1) * 255).astype(np.uint8)
    bi = (np.clip(b, 0, 1) * 255).astype(np.uint8)
    pixels[px_row, px_col, 0] = ri
    pixels[px_row, px_col, 1] = gi
    pixels[px_row, px_col, 2] = bi
    img = Image.fromarray(pixels, "RGB")
    img.save(tex_path)

    print(f"[5/5] Guardando OBJ + MTL...")

    with open(mtl_path, "w") as f:
        f.write("newmtl lidar_tex\n")
        f.write("Ka 1.0 1.0 1.0\n")
        f.write("Kd 1.0 1.0 1.0\n")
        f.write(f"map_Kd {tex_name}\n")

    n = len(x)
    with open(output_path, "w") as f:
        f.write(f"mtllib {mtl_name}\n")
        for i in range(n):
            f.write(f"v {x[i]:.3f} {y[i]:.3f} {z[i]:.3f}\n")
        for i in range(n):
            f.write(f"vt {u[i]:.6f} {v_uv[i]:.6f}\n")
        f.write("usemtl lidar_tex\n")
        for tri_idx in simplices:
            a, b_, c = tri_idx[0]+1, tri_idx[1]+1, tri_idx[2]+1
            f.write(f"f {a}/{a} {b_}/{b_} {c}/{c}\n")

    print(f"      Vertices:   {n:,}")
    print(f"      Triangulos: {len(simplices):,}")
    print(f"      Textura:    {TEX_SIZE}x{TEX_SIZE} px")
    print(f"      Color:      {'RGB real' if has_color else 'por altura'}")
    print(f"\n✓ Ficheros generados:")
    print(f"  - {output_path}")
    print(f"  - {mtl_path}")
    print(f"  - {tex_path}")
    print(f"\n  IMPORTANTE: los 3 ficheros deben estar en la misma carpeta al importar en WinTAK")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python laz_to_obj.py fichero.laz [salida.obj]")
        sys.exit(1)
    entrada = sys.argv[1]
    salida = sys.argv[2] if len(sys.argv) > 2 else None
    laz_to_obj(entrada, salida)
