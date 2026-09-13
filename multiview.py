# 切り抜き 1 枚から、45° 刻み 8 方向の画像を作る（MV-Adapter, SDXL）。
#
# ワーカー（worker_entrypoint.py）が JOB_KIND=views のときに別プロセスで呼ぶ。
# 工程の印（=== ... ===）は標準出力に出し、ワーカーがそれを拾って status に書く。
# 3D 生成（minimal_demo_mmgp.py）と同じ作り。
#
# MV-Adapter は 6 方向（0/45/90/180/270/315）で学習されているが、方位角は
# カメラ条件として渡す作りなので 8 方向も指定できる。135° と 225° は学習外。
#
# 重みは HF_HOME/mvadapter に置いてある前提（ワーカーが起動時に落とす。いまは配置していない）。
#
# 出力は灰色の背景付きなので、切り抜きサービスと同じ BiRefNet（CPU）で背景を抜き、
# **8 枚に共通の範囲**で切り詰める。1 枚ずつ切り詰めると向きごとに大きさが変わり、
# 板に貼ったときに回すたびに家具が伸び縮みする

import argparse
import os
import sys
import time

import torch
from diffusers import AutoencoderKL
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "cutout"))
from birefnet import content_box, crop_with_margin, load_session, matte  # noqa: E402
from mvadapter.pipelines.pipeline_mvadapter_i2mv_sdxl import MVAdapterI2MVSDXLPipeline  # noqa: E402
from mvadapter.schedulers.scheduling_shift_snr import ShiftSNRScheduler  # noqa: E402
from mvadapter.utils import get_orthogonal_camera, get_plucker_embeds_from_cameras_ortho  # noqa: E402

AZIMUTHS = [0, 45, 90, 135, 180, 225, 270, 315]
NEGATIVE_PROMPT = "watermark, ugly, deformed, noisy, blurry, low contrast"
# 背景を抜くモデル。Dockerfile がイメージに焼く（cutout/Dockerfile と同じ配布物）
BIREFNET_PATH = "/opt/birefnet/birefnet-general-lite.onnx"


def load_pipeline(weights_dir: str, num_views: int) -> MVAdapterI2MVSDXLPipeline:
    vae = AutoencoderKL.from_pretrained(os.path.join(weights_dir, "vae"), torch_dtype=torch.float16)
    pipe = MVAdapterI2MVSDXLPipeline.from_pretrained(
        os.path.join(weights_dir, "sdxl"), vae=vae,
        torch_dtype=torch.float16, variant="fp16", use_safetensors=True,
    )
    pipe.scheduler = ShiftSNRScheduler.from_scheduler(
        pipe.scheduler, shift_mode="interpolated", shift_scale=8.0, scheduler_class=None
    )
    pipe.init_custom_adapter(num_views=num_views)
    pipe.load_custom_adapter(os.path.join(weights_dir, "adapter"), weight_name="mvadapter_i2mv_sdxl.safetensors")
    pipe.to(device="cuda", dtype=torch.float16)
    pipe.cond_encoder.to(device="cuda", dtype=torch.float16)
    pipe.enable_vae_slicing()
    return pipe


def preprocess(image: Image.Image, size: int) -> Image.Image:
    """切り抜きを、中身が枠の 9 割に収まるよう中央に置き、透明部分を灰色で埋める（学習時と同じ）"""
    alpha = image.getchannel("A")
    box = alpha.getbbox() or (0, 0, image.width, image.height)
    content = image.crop(box)
    scale = size * 0.9 / max(content.width, content.height)
    content = content.resize((max(1, round(content.width * scale)), max(1, round(content.height * scale))), Image.LANCZOS)
    canvas = Image.new("RGBA", (size, size), (128, 128, 128, 255))
    canvas.alpha_composite(content, ((size - content.width) // 2, (size - content.height) // 2))
    return canvas.convert("RGB")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-image", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--weights-dir", default=os.path.join(os.environ.get("HF_HOME", "/tmp/models"), "mvadapter"))
    parser.add_argument("--size", type=int, default=768)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--birefnet", default=BIREFNET_PATH)
    args = parser.parse_args()

    num_views = len(AZIMUTHS)
    print("=== Loading multiview model ===", flush=True)
    pipe = load_pipeline(args.weights_dir, num_views)

    cameras = get_orthogonal_camera(
        elevation_deg=[0] * num_views, distance=[1.8] * num_views,
        left=-0.55, right=0.55, bottom=-0.55, top=0.55,
        azimuth_deg=[a - 90 for a in AZIMUTHS], device="cuda",
    )
    plucker = get_plucker_embeds_from_cameras_ortho(cameras.c2w, [1.1] * num_views, args.size)
    control_images = ((plucker + 1.0) / 2.0).clamp(0, 1)

    reference = preprocess(Image.open(args.input_image).convert("RGBA"), args.size)

    print("=== Generating views ===", flush=True)
    started = time.perf_counter()
    images = pipe(
        "high quality",
        height=args.size, width=args.size,
        num_inference_steps=args.steps, guidance_scale=3.0,
        num_images_per_prompt=num_views,
        control_image=control_images, control_conditioning_scale=1.0,
        reference_image=reference, reference_conditioning_scale=1.0,
        negative_prompt=NEGATIVE_PROMPT,
        cross_attention_kwargs={"scale": 1.0},
        generator=torch.Generator(device="cuda").manual_seed(args.seed),
    ).images
    print(f"generated {num_views} views in {time.perf_counter() - started:.0f}s", flush=True)

    # 拡散モデルは GPU から下ろしてから背景を抜く。CPU の推論と GPU は競合しないが、
    # 8 枚の RGBA を持つぶんメモリが要るので、先に空けておく
    del pipe
    torch.cuda.empty_cache()

    print("=== Cutting out views ===", flush=True)
    started = time.perf_counter()
    session = load_session(args.birefnet, os.cpu_count() or 1)
    matted = [matte(session, image) for image in images]
    # 共通の範囲 = 8 枚の中身の範囲の和。どの向きも同じ枠に収まり、足元の高さもそろう
    boxes = [box for box in (content_box(rgba) for rgba in matted) if box]
    if boxes:
        union = (
            min(b[0] for b in boxes), min(b[1] for b in boxes),
            max(b[2] for b in boxes), max(b[3] for b in boxes),
        )
        matted = [crop_with_margin(rgba, union) for rgba in matted]
    print(f"cut out {num_views} views in {time.perf_counter() - started:.0f}s", flush=True)

    os.makedirs(args.output_dir, exist_ok=True)
    for azimuth, rgba in zip(AZIMUTHS, matted):
        rgba.save(os.path.join(args.output_dir, f"{azimuth:03d}.png"))
    print("=== Views generated ===", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
