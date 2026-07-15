"""img_to_3d.py — image -> 3D MESH, in picture-ai (RTX 5080). Uses Hunyuan3D-2's SHAPE pipeline
(Apache-2.0, pure PyTorch — the texture stage with its custom CUDA rasterizer is NOT used; textures are
baked in Blender from the source image). Reuses picture-ai's torch 2.11+cu128 (Blackwell) env.

  picture-ai/venv/Scripts/python.exe scripts/img_to_3d.py --image <png> --out <glb> [--steps 30] [--octree 256]

The input should be a single object (a garment) on a plain background — a picture-ai ghost-mannequin shot
is ideal. Background is removed to a clean alpha before reconstruction.
"""
import sys, argparse, time
from pathlib import Path
from PIL import Image

HY = r"C:\Users\arhal_iz5093n\Desktop\projects\Hunyuan3D-2"
sys.path.insert(0, HY)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="tencent/Hunyuan3D-2")
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--octree", type=int, default=256)
    a = ap.parse_args()

    from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline
    from hy3dgen.rembg import BackgroundRemover

    t0 = time.time()
    print(f"loading shape model {a.model} ...", flush=True)
    pipe = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(a.model)
    print(f"model loaded in {time.time()-t0:.0f}s", flush=True)

    img = Image.open(a.image).convert("RGBA")
    if img.getextrema()[3][0] == 255:            # fully opaque -> no alpha -> strip background
        img = BackgroundRemover()(img)

    t1 = time.time()
    mesh = pipe(image=img, num_inference_steps=a.steps, octree_resolution=a.octree,
                generator=None)[0]
    print(f"reconstructed in {time.time()-t1:.0f}s -> {len(mesh.vertices)} verts, {len(mesh.faces)} faces",
          flush=True)

    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(str(out))
    print("SAVED", out, flush=True)


if __name__ == "__main__":
    main()
