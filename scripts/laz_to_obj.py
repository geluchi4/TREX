#!/usr/bin/env python3
"""
Convierte un fichero LAZ/LAS a OBJ con color real RGB.
Uso: python laz_to_obj.py fichero.laz [salida.obj]
Requiere: pip install laspy[lazrs] numpy scipy
"""

import sys
import numpy as np
import laspy
from scipy.spatial import Delaunay

def laz_to_obj(input_path, output_path=None):
    if output_path is None:
        output_path = input_path.rsplit(".", 1)[0] + ".obj"
    mtl_path = output_path.rsplit(".", 1)[0] + ".mtl"

    print(f"[1/4] Leyendo {input_path}...")
    las = laspy.read(input_path)

    x = np.array(las.x)
    y = np.array(las.y)
    z = np.array(las.z)

    # Intentar extraer colores RGB
    has_color = False
    try:
        r = np.array(las.red,   dtype=np.float32)
        g = np.array(las.green, dtype=np.float32)
        b = np.array(las.blue,  dtype=np.float32)
        # Los valores RGB en LAS van de 0-65535, normalizar a 0-1
        if r.max() > 1.0:
            r /= 65535.0
            g /= 65535.0
            b /= 65535.0
        has_color = True
        print(f"      Color RGB encontrado")
    except Exception:
        print(f"      Sin color RGB, se usará color por altura")

    print(f"      {len(x):,} puntos cargados")
    print(f"      Z: {z.min():.1f} - {z.max():.1f} m")

    # Centrar en origen
    cx, cy, cz = x.mean(), y.mean(), z.min()
    x -= cx
    y -= cy
    z -= cz

    # Submuestrear si hay demasiados puntos
    MAX_POINTS = 500_000
    if len(x) > MAX_POINTS:
        print(f"[2/4] Submuestreando a {MAX_POINTS:,} puntos...")
        idx = np.random.choice(len(x), MAX_POINTS, replace=False)
        x, y, z = x[idx], y[idx], z[idx]
        if has_color:
            r, g, b = r[idx], g[idx], b[idx]
    else:
        print(f"[2/4] Preparando {len(x):,} puntos...")

    # Color por altura si no hay RGB
    if not has_color:
        z_norm = (z - z.min()) / (z.max() - z.min() + 1e-6)
        r = 0.3 + 0.5 * z_norm
        g = 0.25 + 0.3 * z_norm
        b = 0.1 * np.ones_like(z_norm)

    print("[3/4] Triangulando...")
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

    print(f"[4/4] Guardando {output_path}...")

    # Escribir MTL con colores por vértice via texturas procedurales no es posible en OBJ estándar,
    # pero WinTAK soporta vertex colors como "v x y z r g b"
    with open(output_path, "w") as f:
        f.write("# OBJ con color RGB real desde LiDAR\n")
        if has_color:
            f.write(f"# Color: RGB real del vuelo fotogrametrico\n")
        else:
            f.write(f"# Color: generado por altura\n")
        for i in range(len(x)):
            f.write(f"v {x[i]:.3f} {y[i]:.3f} {z[i]:.3f} {r[i]:.4f} {g[i]:.4f} {b[i]:.4f}\n")
        for tri_idx in simplices:
            f.write(f"f {tri_idx[0]+1} {tri_idx[1]+1} {tri_idx[2]+1}\n")

    print(f"      Vertices: {len(x):,}")
    print(f"      Triangulos: {len(simplices):,}")
    print(f"      Color: {'RGB real' if has_color else 'por altura'}")
    print(f"\n✓ Fichero guardado: {output_path}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python laz_to_obj.py fichero.laz [salida.obj]")
        sys.exit(1)

    entrada = sys.argv[1]
    salida = sys.argv[2] if len(sys.argv) > 2 else None
    laz_to_obj(entrada, salida)
