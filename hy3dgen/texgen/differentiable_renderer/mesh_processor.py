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

import numpy as np

def meshVerticeInpaint_smooth(texture, mask, vtx_pos, vtx_uv, pos_idx, uv_idx):
    texture_height, texture_width, texture_channel = texture.shape
    vtx_num = vtx_pos.shape[0]

    vtx_mask = np.zeros(vtx_num, dtype=np.float32)
    vtx_color = [np.zeros(texture_channel, dtype=np.float32) for _ in range(vtx_num)]
    uncolored_vtxs = []
    G = [[] for _ in range(vtx_num)]

    for i in range(uv_idx.shape[0]):
        for k in range(3):
            vtx_uv_idx = uv_idx[i, k]
            vtx_idx = pos_idx[i, k]
            uv_v = int(round(vtx_uv[vtx_uv_idx, 0] * (texture_width - 1)))
            uv_u = int(round((1.0 - vtx_uv[vtx_uv_idx, 1]) * (texture_height - 1)))
            if mask[uv_u, uv_v] > 0:
                vtx_mask[vtx_idx] = 1.0
                vtx_color[vtx_idx] = texture[uv_u, uv_v]
            else:
                uncolored_vtxs.append(vtx_idx)
            G[pos_idx[i, k]].append(pos_idx[i, (k + 1) % 3])

    smooth_count = 2
    last_uncolored_vtx_count = 0
    while smooth_count > 0:
        uncolored_vtx_count = 0
        for vtx_idx in uncolored_vtxs:
            sum_color = np.zeros(texture_channel, dtype=np.float32)
            total_weight = 0.0
            vtx_0 = vtx_pos[vtx_idx]
            for connected_idx in G[vtx_idx]:
                if vtx_mask[connected_idx] > 0:
                    vtx1 = vtx_pos[connected_idx]
                    dist = np.sqrt(np.sum((vtx_0 - vtx1) ** 2))
                    dist_weight = 1.0 / max(dist, 1e-4)
                    dist_weight *= dist_weight
                    sum_color += vtx_color[connected_idx] * dist_weight
                    total_weight += dist_weight
            if total_weight > 0:
                vtx_color[vtx_idx] = sum_color / total_weight
                vtx_mask[vtx_idx] = 1.0
            else:
                uncolored_vtx_count += 1

        if last_uncolored_vtx_count == uncolored_vtx_count:
            smooth_count -= 1
        else:
            smooth_count += 1
        last_uncolored_vtx_count = uncolored_vtx_count

    new_texture = texture.copy()
    new_mask = mask.copy()
    for face_idx in range(uv_idx.shape[0]):
        for k in range(3):
            vtx_uv_idx = uv_idx[face_idx, k]
            vtx_idx = pos_idx[face_idx, k]
            if vtx_mask[vtx_idx] == 1.0:
                uv_v = int(round(vtx_uv[vtx_uv_idx, 0] * (texture_width - 1)))
                uv_u = int(round((1.0 - vtx_uv[vtx_uv_idx, 1]) * (texture_height - 1)))
                new_texture[uv_u, uv_v] = vtx_color[vtx_idx]
                new_mask[uv_u, uv_v] = 255
    return new_texture, new_mask

def meshVerticeInpaint_smooth_fast(texture, mask, vtx_pos, vtx_uv, pos_idx, uv_idx):
    """meshVerticeInpaint_smooth と同じ処理を numpy / scipy でまとめて行う。

    元の実装は 4 万面 × 3 頂点を Python のループで回し、2048² のテクスチャで 50 秒かかっていた
    （1 件の 1/4）。やっていることは「色の付いていない頂点を、色の付いた隣の頂点の
    距離の逆二乗の重み付き平均で埋め、進まなくなるまで繰り返す」なので、隣接を疎行列にして
    行列ベクトル積で回せば数秒で済む。

    違いは 1 点だけ: 元は 1 周の中で前の頂点の更新を後の頂点が見る（逐次）が、こちらは
    1 周ぶんをまとめて計算する（同時）。埋める色が僅かに変わるが、埋めるのは見えていない面の
    穴なので目に見える差にはならない。
    """
    from scipy import sparse

    texture_height, texture_width, texture_channel = texture.shape
    vtx_num = vtx_pos.shape[0]

    # --- 面の角ごとに、頂点と UV 上の画素を対応づける（元のループの 1 つ目）---
    vidx = pos_idx.reshape(-1).astype(np.int64)
    uvidx = uv_idx.reshape(-1).astype(np.int64)
    uv_v = np.rint(vtx_uv[uvidx, 0] * (texture_width - 1)).astype(np.int64)
    uv_u = np.rint((1.0 - vtx_uv[uvidx, 1]) * (texture_height - 1)).astype(np.int64)
    colored = mask[uv_u, uv_v] > 0

    vtx_mask = np.zeros(vtx_num, dtype=np.float32)
    vtx_color = np.zeros((vtx_num, texture_channel), dtype=np.float32)
    vtx_mask[vidx[colored]] = 1.0
    vtx_color[vidx[colored]] = texture[uv_u[colored], uv_v[colored]]
    uncolored = np.unique(vidx[~colored])

    # --- 隣接（各角から次の角へ）と距離の逆二乗の重み ---
    src = pos_idx.reshape(-1).astype(np.int64)
    dst = np.roll(pos_idx, -1, axis=1).reshape(-1).astype(np.int64)
    dist = np.sqrt(np.sum((vtx_pos[src] - vtx_pos[dst]) ** 2, axis=1))
    weight = (1.0 / np.maximum(dist, 1e-4)) ** 2
    W = sparse.csr_matrix((weight.astype(np.float64), (src, dst)), shape=(vtx_num, vtx_num))
    W_unc = W[uncolored]

    # --- 進まなくなるまで埋める（元の smooth_count と同じ止め方）---
    smooth_count = 2
    last_uncolored_vtx_count = 0
    while smooth_count > 0:
        weights_sum = W_unc @ vtx_mask.astype(np.float64)
        color_sum = W_unc @ (vtx_color * vtx_mask[:, None]).astype(np.float64)
        filled = weights_sum > 0
        vtx_color[uncolored[filled]] = (color_sum[filled] / weights_sum[filled, None]).astype(np.float32)
        vtx_mask[uncolored[filled]] = 1.0
        uncolored_vtx_count = int((~filled).sum())
        if last_uncolored_vtx_count == uncolored_vtx_count:
            smooth_count -= 1
        else:
            smooth_count += 1
        last_uncolored_vtx_count = uncolored_vtx_count

    # --- 色の付いた頂点の画素をテクスチャに書き戻す（元のループの 2 つ目）---
    new_texture = texture.copy()
    new_mask = mask.copy()
    painted = vtx_mask[vidx] == 1.0
    new_texture[uv_u[painted], uv_v[painted]] = vtx_color[vidx[painted]]
    new_mask[uv_u[painted], uv_v[painted]] = 255
    return new_texture, new_mask


def meshVerticeInpaint(texture, mask, vtx_pos, vtx_uv, pos_idx, uv_idx, method="smooth"):
    if method == "smooth":
        return meshVerticeInpaint_smooth_fast(texture, mask, vtx_pos, vtx_uv, pos_idx, uv_idx)
    elif method == "smooth_loop":
        return meshVerticeInpaint_smooth(texture, mask, vtx_pos, vtx_uv, pos_idx, uv_idx)
    else:
        raise ValueError("Invalid method. Use 'smooth' or 'smooth_loop'.")