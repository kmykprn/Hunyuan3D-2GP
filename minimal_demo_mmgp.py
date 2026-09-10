# Hunyuan 3D is licensed under the TENCENT HUNYUAN NON-COMMERCIAL LICENSE AGREEMENT
# except for the third-party components listed below.
# Hunyuan 3D does not impose any additional limitations beyond what is outlined
# in the repsective licenses of these third-party components.
# Users must comply with all terms and conditions of original licenses of these third-party
# components and must ensure that the usage of the third party components adheres to
# all relevant laws and regulations.

# For avoidance of doubts, Hunyuan 3D means the large language models and
# their software and algorithms, including trained model weights, parameters (including
# optimizer states), machine-learning model code, inference-enabling code, training-enabling code,
# fine-tuning enabling code and other elements of the foregoing made publicly available
# by Tencent in accordance with TENCENT HUNYUAN COMMUNITY LICENSE AGREEMENT.

import os
import time
import gc
import atexit
import signal
import sys

from hunyuan3d_mmpg.cli_args import parse_arguments

import torch
from mmgp import offload
from hunyuan3d_mmpg.utils import (
    replace_property_getter,
    get_example_mv_list,
    get_example_img_list,
    get_example_txt_list,
    randomize_seed_fn,
    gen_save_folder,
    export_mesh,
)
from hy3dgen.shapegen.utils import logger

# メモリ監視用のヘルパー関数
def print_gpu_memory(stage=""):
    """GPU メモリ使用状況を表示"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        reserved = torch.cuda.memory_reserved() / 1024**3    # GB
        max_memory = torch.cuda.max_memory_allocated() / 1024**3  # GB
        print(f"\n[{stage}] GPU Memory:")
        print(f"  Allocated: {allocated:.2f} GB")
        print(f"  Reserved:  {reserved:.2f} GB")
        print(f"  Peak:      {max_memory:.2f} GB")
        
        # nvidia-smiからの情報も取得
        try:
            import subprocess
            result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used,memory.total', '--format=csv,noheader,nounits'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                used, total = result.stdout.strip().split(', ')
                print(f"  nvidia-smi: {int(used)/1024:.2f} GB / {int(total)/1024:.2f} GB")
        except:
            pass
    else:
        print(f"[{stage}] CUDA not available")

def cleanup_models():
    """モデルのクリーンアップ"""
    print("\n=== Starting cleanup ===")
    print_gpu_memory("Before cleanup")
    
    # グローバル変数を削除
    global_vars = ['i23d_worker', 'texgen_worker', 't2i_worker', 'rmbg_worker', 
                   'face_reduce_worker', 'floater_remove_worker', 'degenerate_face_remove_worker', 'pipe']
    
    for var_name in global_vars:
        if var_name in globals():
            print(f"Deleting {var_name}...")
            del globals()[var_name]
    
    # ガベージコレクション
    print("Running garbage collection...")
    gc.collect()
    
    # GPUメモリを解放
    if torch.cuda.is_available():
        print("Clearing CUDA cache...")
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    
    print_gpu_memory("After cleanup")
    print("=== Cleanup completed ===\n")

# プロセス終了時の処理を登録
atexit.register(cleanup_models)

def signal_handler(sig, frame):
    """シグナルハンドラー（Ctrl+C等）"""
    print("\nReceived interrupt signal")
    cleanup_models()
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

def _gen_shape(
    caption=None,
    image=None,
    mv_image_front=None,
    mv_image_back=None,
    mv_image_left=None,
    mv_image_right=None,
    steps=50,
    guidance_scale=7.5,
    seed=1234,
    octree_resolution=256,
    check_box_rembg=False,
    num_chunks=200000,
    randomize_seed: bool = False,
):
    # グローバル変数を参照
    global t2i_worker, rmbg_worker, i23d_worker, HAS_T2I
    
    # MV_MODEの時は、front, back, left, rightから画像を読み込む
    if not MV_MODE and image is None and caption is None:
        raise ValueError("Please provide either a caption or an image.")
    if MV_MODE:
        if mv_image_front is None and mv_image_back is None and mv_image_left is None and mv_image_right is None:
            raise ValueError("Please provide at least one view image.")
        image = {}
        if mv_image_front:
            image['front'] = mv_image_front
        if mv_image_back:
            image['back'] = mv_image_back
        if mv_image_left:
            image['left'] = mv_image_left
        if mv_image_right:
            image['right'] = mv_image_right

    # シード値を決定
    seed = int(randomize_seed_fn(seed, randomize_seed))

    # オクトツリーの解像度を決定（値が大きいほど、細かく空間が分割される）
    octree_resolution = int(octree_resolution)
    
    # promptを表示
    if caption:
        print('prompt is', caption)
    
    # 3Dモデルの出力先ディレクトリを作成
    save_folder = gen_save_folder()
    
    stats = {
        'model': {
            'shapegen': f'{args.model_path}/{args.subfolder}',
            'texgen': f'{args.texgen_model_path}',
        },
        'params': {
            'caption': caption,
            'steps': steps,
            'guidance_scale': guidance_scale,
            'seed': seed,
            'octree_resolution': octree_resolution,
            'check_box_rembg': check_box_rembg,
            'num_chunks': num_chunks,
        }
    }
    time_meta = {}

    if image is None:
        start_time = time.time()
        try:
            image = t2i_worker(caption)
        except Exception as e:
            print(f"Text to 3D is disable. Please enable it by `python gradio_app.py --enable_t23d`.:{e}")
            raise Exception
        time_meta['text2image'] = time.time() - start_time

    # remove disk io to make responding faster, uncomment at your will.
    # image.save(os.path.join(save_folder, 'input.png'))
    if MV_MODE:
        start_time = time.time()
        for k, v in image.items():
            if check_box_rembg or v.mode == "RGB":
                img = rmbg_worker(v.convert('RGB'))
                image[k] = img
        time_meta['remove background'] = time.time() - start_time
    else:
        if check_box_rembg or image.mode == "RGB":
            start_time = time.time()
            image = rmbg_worker(image.convert('RGB'))
            time_meta['remove background'] = time.time() - start_time

    # remove disk io to make responding faster, uncomment at your will.
    # image.save(os.path.join(save_folder, 'rembg.png'))

    # image to white model
    start_time = time.time()

    generator = torch.Generator()
    generator = generator.manual_seed(int(seed))
    outputs = i23d_worker(
        image=image,
        num_inference_steps=steps,
        guidance_scale=guidance_scale,
        generator=generator,
        octree_resolution=octree_resolution,
        num_chunks=num_chunks,
        output_type='mesh'
    )
    time_meta['shape generation'] = time.time() - start_time
    logger.info("---Shape generation takes %s seconds ---" % (time.time() - start_time))

    tmp_start = time.time()
    mesh = export_to_trimesh(outputs)[0]
    time_meta['export to trimesh'] = time.time() - tmp_start

    stats['number_of_faces'] = mesh.faces.shape[0]
    stats['number_of_vertices'] = mesh.vertices.shape[0]

    stats['time'] = time_meta
    main_image = image if not MV_MODE else image['front']
    return mesh, main_image, save_folder, stats, seed


def generate_3d_cli(
    input_image_path: str = None,
    text_prompt: str = None,
    output_dir: str = None,
    steps: int = 50,
    guidance_scale: float = 7.5,
    seed: int = 1234,
    generate_texture: bool = False,
    remove_background: bool = True,
    octree_resolution: int = 256,
    num_chunks: int = 200000,
):
    """
    CLIから3Dモデルを生成するメイン関数
    
    Args:
        input_image_path: 入力画像のパス
        text_prompt: テキストプロンプト（画像がない場合）
        output_dir: 出力ディレクトリ（指定しない場合は自動生成）
        steps: 生成ステップ数
        guidance_scale: ガイダンススケール
        seed: シード値
        generate_texture: テクスチャ生成するか
        remove_background: 背景除去するか
        octree_resolution: オクトツリー解像度
        num_chunks: チャンク数
    
    Returns:
        dict: 生成結果の情報（パス、統計情報など）
    """
    # グローバル変数を参照
    global HAS_TEXTUREGEN, face_reduce_worker, texgen_worker
    
    # 画像を読み込み
    image = None
    if input_image_path:
        from PIL import Image
        image = Image.open(input_image_path)
    
    start_time_0 = time.time()
    
    # 3D形状生成
    mesh, processed_image, save_folder, stats, used_seed = _gen_shape(
        caption=text_prompt,
        image=image,
        steps=steps,
        guidance_scale=guidance_scale,
        seed=seed,
        octree_resolution=octree_resolution,
        check_box_rembg=remove_background,
        num_chunks=num_chunks,
        randomize_seed=False,
    )
    
    # 出力ディレクトリを指定された場合は変更
    if output_dir:
        import os
        os.makedirs(output_dir, exist_ok=True)
        save_folder = output_dir
    
    # メッシュをエクスポート
    shape_path = export_mesh(mesh, save_folder, textured=False)
    
    result = {
        'shape_path': shape_path,
        'save_folder': save_folder,
        'stats': stats,
        'seed': used_seed,
    }
    
    # テクスチャ生成（オプション）
    if generate_texture and HAS_TEXTUREGEN:
        print("Generating texture...")
        tmp_time = time.time()
        mesh = face_reduce_worker(mesh)
        textured_mesh = texgen_worker(mesh, processed_image)
        stats['time']['texture_generation'] = time.time() - tmp_time
        texture_path = export_mesh(textured_mesh, save_folder, textured=True)
        result['texture_path'] = texture_path
        print(f"Texture generation took {stats['time']['texture_generation']:.2f} seconds")
    
    stats['time']['total'] = time.time() - start_time_0
    
    if args.low_vram_mode:
        torch.cuda.empty_cache()
    
    return result


def generation_all(
    caption=None,
    image=None,
    mv_image_front=None,
    mv_image_back=None,
    mv_image_left=None,
    mv_image_right=None,
    steps=50,
    guidance_scale=7.5,
    seed=1234,
    octree_resolution=256,
    check_box_rembg=False,
    num_chunks=200000,
    randomize_seed: bool = False,
):
    # グローバル変数を参照
    global face_reduce_worker, texgen_worker, HAS_TEXTUREGEN, SAVE_DIR
    
    start_time_0 = time.time()
    mesh, image, save_folder, stats, seed = _gen_shape(
        caption,
        image,
        mv_image_front=mv_image_front,
        mv_image_back=mv_image_back,
        mv_image_left=mv_image_left,
        mv_image_right=mv_image_right,
        steps=steps,
        guidance_scale=guidance_scale,
        seed=seed,
        octree_resolution=octree_resolution,
        check_box_rembg=check_box_rembg,
        num_chunks=num_chunks,
        randomize_seed=randomize_seed,
    )

    # テクスチャ生成が有効な場合のみ実行
    if HAS_TEXTUREGEN:
        tmp_time = time.time()
        mesh = face_reduce_worker(mesh)
        logger.info("---Face Reduction takes %s seconds ---" % (time.time() - tmp_time))
        stats['time']['face reduction'] = time.time() - tmp_time

        tmp_time = time.time()
        textured_mesh = texgen_worker(mesh, image)
        logger.info("---Texture Generation takes %s seconds ---" % (time.time() - tmp_time))
        stats['time']['texture generation'] = time.time() - tmp_time
        stats['time']['total'] = time.time() - start_time_0

        textured_mesh.metadata['extras'] = stats
        path_textured = export_mesh(textured_mesh, save_folder, textured=True)
    else:
        # テクスチャ生成が無効な場合は形状のみエクスポート
        print("Texture generation is disabled. Exporting shape only.")
        stats['time']['total'] = time.time() - start_time_0
        mesh.metadata['extras'] = stats
        path_textured = export_mesh(mesh, save_folder, textured=False)
    if args.low_vram_mode:
        torch.cuda.empty_cache()

    return path_textured


if __name__ == '__main__':
    
    # 引数を取得
    args = parse_arguments()

    SAVE_DIR = args.cache_path
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    MV_MODE = 'mv' in args.model_path
    
    MAX_SEED = int(1e7)

    # イメージとテキストのサンプル（パス）を取得
    example_is = get_example_img_list()
    example_ts = get_example_txt_list()
    example_mvs = get_example_mv_list()

    SUPPORTED_FORMATS = ['glb', 'obj', 'ply', 'stl']
    
    print_gpu_memory("Initial state")

    # テクスチャ付GLBファイルを作成するモデルの読み込み
    HAS_TEXTUREGEN = False
    if not args.disable_tex:
        try:
            print("\n=== Loading texture generation model ===")
            print_gpu_memory("Before texgen import")
            from hy3dgen.texgen import Hunyuan3DPaintPipeline
            print_gpu_memory("After texgen import")
            
            texgen_worker = Hunyuan3DPaintPipeline.from_pretrained(args.texgen_model_path)
            print_gpu_memory("After texgen model loaded")
            HAS_TEXTUREGEN = True
        except Exception as e:
            print(e)
            print("Failed to load texture generator.")
            print('Please try to install requirements by following README.md')
            HAS_TEXTUREGEN = False

    # テキストから3Dモデルを作成するモデルの読み込み
    HAS_T2I = False
    if args.enable_t23d:
        from hy3dgen.text2image import HunyuanDiTPipeline

        t2i_worker = HunyuanDiTPipeline('Tencent-Hunyuan/HunyuanDiT-v1.1-Diffusers-Distilled')
        HAS_T2I = True

    from hy3dgen.shapegen import (
        FaceReducer,
        FloaterRemover,
        DegenerateFaceRemover,
        Hunyuan3DDiTFlowMatchingPipeline
    )
    from hy3dgen.shapegen.pipelines import export_to_trimesh
    from hy3dgen.rembg import BackgroundRemover

    # 背景削除ワーカーの初期化
    rmbg_worker = BackgroundRemover()
    
    # 画像を3Dに変換するワーカーの初期化
    print("\n=== Loading i23d model ===")
    print_gpu_memory("Before i23d model")
    i23d_worker = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
        args.model_path,
        subfolder=args.subfolder,
        use_safetensors=True,
        device=args.device,
    )
    print_gpu_memory("After i23d model loaded")
    
    # Turboモデルを使用するときの設定
    if args.enable_flashvdm:
        mc_algo = 'mc' if args.device in ['cpu', 'mps'] else args.mc_algo
        i23d_worker.enable_flashvdm(mc_algo=mc_algo)
    if args.compile:
        i23d_worker.compile()

    # 各種ワーカーの設定
    floater_remove_worker = FloaterRemover()
    degenerate_face_remove_worker = DegenerateFaceRemover()
    face_reduce_worker = FaceReducer()
  
    # profile(VRAMを使うパラメータ。1~5）を設定 - gradio_app.pyと同じ順序
    profile = int(args.profile)
    kwargs = {}

    # モデルがCUDAにあることを固定（_execution_deviceがcpuを返した場合のエラーに対応）
    replace_property_getter(i23d_worker, "_execution_device", lambda self : "cuda")
    
    # pipeにshapegenモデルを追加
    pipe = offload.extract_models("i23d_worker", i23d_worker)
    
    # texgenモデルを追加
    if HAS_TEXTUREGEN:
        pipe.update(offload.extract_models( "texgen_worker", texgen_worker))
        texgen_worker.models["multiview_model"].pipeline.vae.use_slicing = True
    
    # T2Iモデルを追加
    if HAS_T2I:
        pipe.update(offload.extract_models( "t2i_worker", t2i_worker))
    
    # mmgpのメモリ設定
    if profile < 5:
        kwargs["pinnedMemory"] = "i23d_worker/model"
    if profile !=1 and profile !=3:
        kwargs["budgets"] = { "*" : 2200 }
    verboseLevel = int(args.verbose)  # gradio_app.pyと同じ変数名
    offload.default_verboseLevel = verboseLevel
    offload.profile(pipe, profile_no = profile, verboseLevel = verboseLevel, **kwargs)

    if args.low_vram_mode:
        torch.cuda.empty_cache()
    
    # CLI実行モード
    # python minimal_demo_mmgp.py --input-image assets/example_images/052.png --output ./output --texture
    if args.input_image or args.prompt:
        print("\n=== CLI Mode ===")
        print(f"Input image: {args.input_image}")
        print(f"Text prompt: {args.prompt}")
        print(f"Output directory: {args.output}")
        print(f"Generate texture: {args.texture}")
        print(f"Steps: {args.steps}")
        print(f"Guidance scale: {args.guidance_scale}")
        print(f"Seed: {args.seed}")
        print("================\n")
        
        # 画像を読み込み（必要な場合）
        image = None
        if args.input_image:
            from PIL import Image
            image = Image.open(args.input_image)
            print(f"Loaded image: {args.input_image}")
        
        # 3D生成を実行
        if args.texture:
            # テクスチャ付き生成（generation_all使用）
            print("Generating 3D model with texture...")
            path_textured = generation_all(
                caption=args.prompt,
                image=image,
                steps=args.steps,
                guidance_scale=args.guidance_scale,
                seed=args.seed,
                check_box_rembg=True,  # 背景除去は常に有効
            )
            
            print(f"\n✅ 3D model with texture generated successfully!")
            print(f"Textured model: {path_textured}")
        else:
            # テクスチャなし生成（generate_3d_cli使用）
            print("Generating 3D model without texture...")
            result = generate_3d_cli(
                input_image_path=args.input_image,
                text_prompt=args.prompt,
                output_dir=args.output,
                steps=args.steps,
                guidance_scale=args.guidance_scale,
                seed=args.seed,
                generate_texture=False,
                remove_background=True,
            )
            
            print(f"\n✅ 3D model generated successfully!")
            print(f"Shape: {result['shape_path']}")
            print(f"Total time: {result['stats']['time']['total']:.2f} seconds")
    else:
        # 引数がない場合はヘルプを表示
        print("\n=== Hunyuan3D CLI Tool ===")
        print("Usage examples:")
        print("  # Generate from image:")
        print("  python minimal_demo_mmgp.py --input-image path/to/image.png --output ./output --texture")
        print("")
        print("  # Generate from text prompt:")
        print("  python minimal_demo_mmgp.py --prompt 'a cute cat' --output ./output")
        print("")
        print("  # With custom parameters:")
        print("  python minimal_demo_mmgp.py --input-image image.png --steps 100 --seed 42 --texture")
        print("========================\n")
    
    

