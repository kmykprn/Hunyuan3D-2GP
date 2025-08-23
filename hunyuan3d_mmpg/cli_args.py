import argparse

def parse_arguments():
    """コマンドライン引数のパース"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default='tencent/Hunyuan3D-2mini')
    parser.add_argument("--subfolder", type=str, default='hunyuan3d-dit-v2-mini')
    parser.add_argument("--texgen_model_path", type=str, default='tencent/Hunyuan3D-2')
    parser.add_argument('--port', type=int, default=8080)
    parser.add_argument('--host', type=str, default='0.0.0.0')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--mc_algo', type=str, default='dmc')
    parser.add_argument('--cache-path', type=str, default='gradio_cache')
    parser.add_argument('--enable_t23d', action='store_true')
    parser.add_argument('--profile', type=str, default="3")
    parser.add_argument('--verbose', type=str, default="1")

    parser.add_argument('--disable_tex', action='store_true')
    parser.add_argument('--enable_flashvdm', action='store_true')
    parser.add_argument('--low-vram-mode', action='store_true')
    parser.add_argument('--compile', action='store_true')
    parser.add_argument('--mini', action='store_true')
    parser.add_argument('--turbo', action='store_true')
    parser.add_argument('--mv', action='store_true')
    parser.add_argument('--h2', action='store_true')
    
    # CLI用の引数を追加
    parser.add_argument('--input-image', type=str, help='Input image path for 3D generation')
    parser.add_argument('--prompt', type=str, help='Text prompt for 3D generation')
    parser.add_argument('--output', type=str, default='./output', help='Output directory')
    parser.add_argument('--texture', action='store_true', help='Generate texture')
    parser.add_argument('--steps', type=int, default=50, help='Number of generation steps')
    parser.add_argument('--guidance-scale', type=float, default=7.5, help='Guidance scale')
    parser.add_argument('--seed', type=int, default=1234, help='Random seed')

    args = parser.parse_args()

    # プリセット設定の適用
    if args.mini:
        args.model_path = "tencent/Hunyuan3D-2mini"
        args.subfolder = "hunyuan3d-dit-v2-mini"
        args.texgen_model_path = "tencent/Hunyuan3D-2"

    if args.mv:
        args.model_path = "tencent/Hunyuan3D-2mv"
        args.subfolder = "hunyuan3d-dit-v2-mv"
        args.texgen_model_path = "tencent/Hunyuan3D-2"

    if args.h2:
        args.model_path = "tencent/Hunyuan3D-2"
        args.subfolder = "hunyuan3d-dit-v2-0"
        args.texgen_model_path = "tencent/Hunyuan3D-2"

    if args.turbo:
        args.subfolder = args.subfolder + "-turbo"
        args.enable_flashvdm = True

    return args