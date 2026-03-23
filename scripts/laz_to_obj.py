#!/usr/bin/env python3
"""
Convierte un fichero LAZ/LAS a OBJ con malla 3D.
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

    print(f"[1/4] Leyendo {input_path}...")
    las = laspy.read(input_path)

    x = np.array(las.x)
    y = np.array(las.y)
    z = np.array(las.z)

    print(f"      {len(x):,} puntos cargados")
    print(f"      X: {x.min():.1f} - {x.max():.1f}")
    print(f"      Y: {y.min():.1f} - {y.max():.1f}")
    print(f"      Z: {z.min():.1f} - {z.max():.1f}")

    # Centrar en origen para evitar problemas de precisión
    cx, cy, cz = x.mean(), y.mean(), z.min()
    x -= cx
    y -= cy
    z -= cz

    # Submuestrear si hay demasiados puntos (Delaunay es lento con >500k)
    MAX_POINTS = 500_000
    if len(x) > MAX_POINTS:
        print(f"[2/4] Submuestreando a {MAX_POINTS:,} puntos...")
        idx = np.random.choice(len(x), MAX_POINTS, replace=False)
        x, y, z = x[idx], y[idx], z[idx]
    else:
        print(f"[2/4] Preparando {len(x):,} puntos...")

    print("[3/4] Triangulando (puede tardar unos minutos)...")
    # Delaunay 2D sobre XY, usando Z como altura
    tri = Delaunay(np.stack([x, y], axis=-1))

    # Filtrar triángulos muy grandes (ruido en bordes)
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
    with open(output_path, "w") as f:
        f.write("# OBJ generado desde LiDAR LAZ\n")
        for px, py, pz in zip(x, y, z):
            f.write(f"v {px:.3f} {py:.3f} {pz:.3f}\n")
        for tri_idx in simplices:
            # OBJ usa índices base 1
            f.write(f"f {tri_idx[0]+1} {tri_idx[1]+1} {tri_idx[2]+1}\n")

    print(f"      Vertices: {len(x):,}")
    print(f"      Triangulos: {len(simplices):,}")
    print(f"\n✓ Fichero guardado: {output_path}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python laz_to_obj.py fichero.laz [salida.obj]")
        sys.exit(1)

    entrada = sys.argv[1]
    salida = sys.argv[2] if len(sys.argv) > 2 else None
    laz_to_obj(entrada, salida)
