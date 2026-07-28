# One-off asset generation for HalfaxForge's tavern scene. EXTERNAL to the engine
# (this is the picture-ai repo) — the engine only ever loads the baked PNGs this
# writes. Run with the picture-ai venv:
#   ./venv/Scripts/python.exe scripts/gen_tavern.py
#
# Textures are text2img (RealVisXL SDXL). Character skins are img2img at strength
# 0.5 over the rigged model's own UV texture, so the UV islands stay put (face
# stays on the face) — the standard "repaint the model's UV" pipeline.
import torch, time, os
from diffusers import StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline, DPMSolverMultistepScheduler
from PIL import Image

FORGE = r"C:\Users\arhal_iz5093n\Desktop\projects\halfaxforge"
OUT = os.path.join(FORGE, "assets", "tavern")
SOLDIER_UV = os.path.join(FORGE, "assets", "demo", "skin_explorer_orig.png")
os.makedirs(OUT, exist_ok=True)

NEG_TEX  = "text, watermark, signature, people, hands, blurry, low quality, jpeg artifacts, strong perspective, harsh shadows, border, frame"
NEG_SKIN = "text, watermark, extra limbs, deformed, distorted, blurry, low quality, seams, background"

t0 = time.time()
pipe = StableDiffusionXLPipeline.from_pretrained(
    "SG161222/RealVisXL_V5.0", torch_dtype=torch.float16, use_safetensors=True).to("cuda")
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config, use_karras_sigmas=True)
pipe.set_progress_bar_config(disable=True)
img2img = StableDiffusionXLImg2ImgPipeline(**pipe.components)
img2img.set_progress_bar_config(disable=True)
print(f"loaded {time.time()-t0:.1f}s", flush=True)

def t2i(name, prompt, seed, w=1024, h=1024):
    im = pipe(prompt=prompt, negative_prompt=NEG_TEX, width=w, height=h,
              num_inference_steps=30, guidance_scale=6.0,
              generator=torch.Generator("cuda").manual_seed(seed)).images[0]
    im.save(os.path.join(OUT, name + ".png")); print("wrote", name, flush=True)

def skin(name, prompt, seed):
    base = Image.open(SOLDIER_UV).convert("RGB").resize((1024, 1024))
    im = img2img(prompt=prompt, negative_prompt=NEG_SKIN, image=base, strength=0.5,
                 num_inference_steps=34, guidance_scale=6.5,
                 generator=torch.Generator("cuda").manual_seed(seed)).images[0]
    im.save(os.path.join(OUT, name + ".png")); print("wrote", name, flush=True)

t2i("floor",   "seamless top-down texture of dark weathered oak wood planks, medieval tavern floor, warm brown, high detail, pbr albedo, even flat lighting", 42)
t2i("wall",    "seamless texture of a medieval tavern wall, cream plaster between dark timber framing beams, wattle and daub, high detail, even flat lighting", 7)
t2i("wood",    "seamless texture of polished oak wood, medieval furniture grain, warm honey brown, high detail, pbr albedo, even flat lighting", 11)
t2i("steel",   "seamless texture of polished forged steel blade, faint scratches, cool grey metal, high detail, pbr albedo, even flat lighting", 23)
t2i("tankard", "seamless texture of dented pewter tankard metal, medieval, grey silver sheen, high detail, pbr albedo, even flat lighting", 5)
t2i("barrel",  "seamless texture of a wooden barrel, oak staves with dark iron bands, medieval, high detail, pbr albedo, even flat lighting", 17)
t2i("rug",     "seamless top-down texture of a medieval woven wool rug, deep red and gold geometric pattern, high detail, even flat lighting", 31)
t2i("brass",   "seamless texture of aged brass metal, warm golden, faint patina, high detail, pbr albedo, even flat lighting", 3)
skin("char_keep",   "full body game character uv texture of a burly medieval tavern innkeeper, cream linen shirt, brown leather apron, weathered friendly face, short brown beard, realistic skin", 101)
skin("char_patron", "full body game character uv texture of a rugged medieval warrior patron, worn brown leather armor over a dark green tunic, leather boots, stubble, realistic skin", 202)
print("ALL DONE", flush=True)
