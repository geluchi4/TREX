#!/usr/bin/env python3
"""
Convierte un fichero LAZ/LAS a OBJ con malla 3D.
Uso: python3 laz_to_obj.py fichero.laz [salida.obj]
"""

import sys
import numpy as np
import laspy
import open3d as o3d

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

    print("[2/4] Creando nube de puntos...")
    points = np.stack([x, y, z], axis=-1).astype(np.float64)

    # Submuestrear si hay demasiados puntos (>5M puede ser lento)
    if len(points) > 5_000_000:
        print(f"      Demasiados puntos, submuestreando a 5M...")
        idx = np.random.choice(len(points), 5_000_000, replace=False)
        points = points[idx]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    print("[3/4] Calculando normales y generando malla (esto puede tardar unos minutos)...")
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=2.0, max_nn=30)
    )
    pcd.orient_normals_consistent_tangent_plane(30)

    # Reconstrucción de superficie con Poisson
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=10, width=0, scale=1.1, linear_fit=False
    )

    # Eliminar triángulos con baja densidad (bordes ruidosos)
    densities = np.asarray(densities)
    threshold = np.percentile(densities, 10)
    vertices_to_remove = densities < threshold
    mesh.remove_vertices_by_mask(vertices_to_remove)
    mesh.compute_vertex_normals()

    print(f"[4/4] Guardando {output_path}...")
    o3d.io.write_triangle_mesh(output_path, mesh)

    verts = np.asarray(mesh.vertices)
    tris = np.asarray(mesh.triangles)
    print(f"      Vertices: {len(verts):,}")
    print(f"      Triangulos: {len(tris):,}")
    print(f"\n✓ Fichero guardado: {output_path}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python3 laz_to_obj.py fichero.laz [salida.obj]")
        sys.exit(1)

    entrada = sys.argv[1]
    salida = sys.argv[2] if len(sys.argv) > 2 else None
    laz_to_obj(entrada, salida)
