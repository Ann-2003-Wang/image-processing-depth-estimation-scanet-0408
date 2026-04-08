import os
import os.path as osp
import argparse
import cv2
import numpy as np

def gaussian_filter(img, kernel_size, sigma):
    """Returns the image after Gaussian filter."""
    if kernel_size % 2 == 0 or kernel_size <= 1:
        raise ValueError("kernel_size must be an odd number > 1")
    orig_dtype = img.dtype

    # 彩色图：逐通道处理
    if img.ndim == 3:
        result = np.zeros_like(img, dtype=np.float32)
        for c in range(img.shape[2]):
            result[:, :, c] = gaussian_filter(img[:, :, c], kernel_size, sigma)
        return np.clip(result, 0, 255).astype(orig_dtype)

    # 灰度图处理
    H, W = img.shape
    center = kernel_size // 2
    pad = center

    # 生成高斯核
    kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
    if sigma == 0:
        kernel[center, center] = 1.0
    else:
        for i in range(kernel_size):
            for j in range(kernel_size):
                x = i - center
                y = j - center
                kernel[i, j] = np.exp(-(x*x + y*y) / (2 * sigma * sigma))
        kernel /= kernel.sum()

    # 零填充并卷积
    padded = np.pad(img, pad, mode='constant', constant_values=0).astype(np.float32)
    res_img = np.zeros_like(img, dtype=np.float32)
    for i in range(H):
        for j in range(W):
            region = padded[i:i+kernel_size, j:j+kernel_size]
            res_img[i, j] = np.sum(region * kernel)

    res_img = np.clip(res_img, 0, 255).astype(orig_dtype)
    return res_img


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="批量高斯滤波，支持多参数对比图")
    parser.add_argument("--compare", action="store_true", help="保存多参数对比图（原图+三种滤波结果）")
    parser.add_argument("--num", type=int, default=None, 
                        help="指定保存对比图的图像数量（默认全部，仅在 --compare 启用时有效）")
    args = parser.parse_args()

    # 高斯滤波参数组合（kernel_size, sigma）
    param_list = [(5, 1), (5, 3), (7, 3)]

    root_dir = osp.dirname(osp.abspath(__file__))
    output_dir = osp.join(root_dir, "output")
    os.makedirs(output_dir, exist_ok=True)

    if args.compare:
        compare_dir = osp.join(output_dir, "compare")
        os.makedirs(compare_dir, exist_ok=True)

    processed_count = 0

    for file in os.listdir(root_dir):
        if file.lower().endswith(".jpg") and "_gaussian" not in file.lower():
            img_path = osp.join(root_dir, file)
            print(f"正在处理: {img_path}")

            img = cv2.imread(img_path)
            if img is None:
                print(f"警告: 无法读取图像 {img_path}，跳过。")
                continue

            base_name, ext = osp.splitext(file)

            # 对每个参数组合生成滤波结果
            filtered_images = []
            for k, s in param_list:
                res = gaussian_filter(img, k, s)
                out_filename = f"{base_name}_gaussian_k{k}_s{s}{ext}"
                out_path = osp.join(output_dir, out_filename)
                cv2.imwrite(out_path, res)
                print(f"已保存: {out_path}")
                filtered_images.append(res)

            # 生成对比图（原图 + 三个滤波结果）
            if args.compare and (args.num is None or processed_count < args.num):
                # 水平拼接：原图 + 三个结果
                compare_img = np.hstack([img] + filtered_images)
                compare_filename = f"{base_name}_compare{ext}"
                compare_path = osp.join(compare_dir, compare_filename)
                cv2.imwrite(compare_path, compare_img)
                print(f"对比图已保存: {compare_path}")

            processed_count += 1
