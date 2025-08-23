import os
import random
import shutil
import uuid
from pathlib import Path
from glob import glob

def get_example_img_list() -> list[str]:
    """
    assets/example_imagesからすべてのpngファイルのパスを取得する
    
    Args:
        None
    Returns:
        list[str]: .pngファイルのパス
    """
    print('Loading example img list ...')
    return sorted(glob('./assets/example_images/**/*.png', recursive=True))


def get_example_txt_list() -> list[str]:
    """
    assets/example_prompts.txtからテキストプロンプトのリストを取得する
    
    Args:
        None
    Returns:
        list[str]: テキストプロンプトのリスト
    """
    print('Loading example txt list ...')
    txt_list = list()
    for line in open('./assets/example_prompts.txt', encoding='utf-8'):
        txt_list.append(line.strip())
    return txt_list


def get_example_mv_list():
    """
    assets/example_mv_imagesからマルチビュー画像のリストを取得する
    各サブディレクトリから前後左右の4視点の画像パスを収集する
    
    Args:
        None
    Returns:
        list[list[str|None]]: 各要素が[front, back, left, right]の画像パスリスト
                              画像が存在しない場合はNone
    """
    print('Loading example mv list ...')
    mv_list = list()
    root = './assets/example_mv_images'
    for mv_dir in os.listdir(root):
        view_list = []
        for view in ['front', 'back', 'left', 'right']:
            path = os.path.join(root, mv_dir, f'{view}.png')
            if os.path.exists(path):
                view_list.append(path)
            else:
                view_list.append(None)
        mv_list.append(view_list)
    return mv_list


# デフォルト値を定義（minimal_demo_mmgp.pyと同じ値）
SAVE_DIR = "gradio_cache"  # デフォルトのキャッシュディレクトリ
MAX_SEED = int(1e7)  # 最大シード値

def gen_save_folder(max_size=200) -> str:
    """
    保存用のUUIDフォルダを生成する
    フォルダ数が上限を超えた場合は最古のフォルダを削除する
    
    Args:
        max_size (int): 保存フォルダの最大数（デフォルト: 200）
    Returns:
        str: 新しく作成したフォルダのパス
    """
    os.makedirs(SAVE_DIR, exist_ok=True)

    dirs = [f for f in Path(SAVE_DIR).iterdir() if f.is_dir()]

    # ディレクトリ数が最大値を超えたら削除
    if len(dirs) >= max_size:
        oldest_dir = min(dirs, key=lambda x: x.stat().st_ctime)
        shutil.rmtree(oldest_dir)
        print(f"Removed the oldest folder: {oldest_dir}")

    # uuidでディレクトリ名を作成
    new_folder = os.path.join(SAVE_DIR, str(uuid.uuid4()))
    os.makedirs(new_folder, exist_ok=True)
    print(f"Created new folder: {new_folder}")

    return new_folder


def randomize_seed_fn(seed: int, randomize_seed: bool) -> int:
    """
    シード値をランダム化する
    
    Args:
        seed (int): 元のシード値
        randomize_seed (bool): ランダム化するかどうか
    Returns:
        int: ランダム化されたシード値（randomize_seed=Falseの場合は元の値）
    """
    if randomize_seed:
        seed = random.randint(0, MAX_SEED)
    return seed


def export_mesh(mesh, save_folder, textured=False, type='glb'):
    """
    メッシュをファイルにエクスポートする
    
    Args:
        mesh: エクスポートするメッシュオブジェクト
        save_folder (str): 保存先フォルダのパス
        textured (bool): テクスチャ付きかどうか（デフォルト: False）
        type (str): ファイル形式（'glb', 'obj', 'ply', 'stl'）（デフォルト: 'glb'）
    Returns:
        str: 保存したファイルのパス
    """
    if textured:
        path = os.path.join(save_folder, f'textured_mesh.{type}')
    else:
        path = os.path.join(save_folder, f'white_mesh.{type}')
    if type not in ['glb', 'obj']:
        mesh.export(path)
    else:
        mesh.export(path, include_normals=textured)
    return path

def replace_property_getter(instance, property_name, new_getter):
    """
    インスタンスのプロパティのgetterを動的に置き換える
    mmgpオフロード用にデバイス管理を制御するために使用
    
    Args:
        instance: 対象のインスタンス
        property_name (str): 置き換えるプロパティ名
        new_getter: 新しいgetter関数
    Returns:
        instance: 変更されたインスタンス
    """
    # Get the original class and property
    original_class = type(instance)
    original_property = getattr(original_class, property_name)
    
    # Create a custom subclass for this instance
    custom_class = type(f'Custom{original_class.__name__}', (original_class,), {})
    
    # Create a new property with the new getter but same setter
    new_property = property(new_getter, original_property.fset)
    setattr(custom_class, property_name, new_property)
    
    # Change the instance's class
    instance.__class__ = custom_class
    
    return instance