import os
import os.path as osp
import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt

######## Task 2 新增函数：三列图像与各自直方图垂直拼接，生成综合对比图 ########
def save_combined_comparison(orig, global_eq, local_eq, 
                             hist_orig_path, hist_global_path, hist_local_path,
                             save_path):
    """
    将三列图像与各自直方图垂直拼接，生成综合对比图
    Args:
        orig, global_eq, local_eq: 三张图像 (H, W, 3) BGR格式（已转三通道）
        hist_*_path: 已保存的直方图图片路径（PNG）
        save_path: 综合图的保存路径
    """
    # 加载三张直方图（已经是RGB格式，matplotlib保存的）
    hist_orig = cv2.imread(hist_orig_path)
    hist_global = cv2.imread(hist_global_path)
    hist_local = cv2.imread(hist_local_path)

    # 如果直方图尺寸不一致，统一调整宽度（保持比例）
    # 这里假设三张图像宽度相同，我们让直方图宽度与对应图像一致
    h_img, w_img = orig.shape[:2]
    h_hist, w_hist = hist_orig.shape[:2]
    # 调整直方图宽度以匹配图像宽度
    if w_hist != w_img:
        hist_orig = cv2.resize(hist_orig, (w_img, int(h_hist * w_img / w_hist)))
        hist_global = cv2.resize(hist_global, (w_img, int(h_hist * w_img / w_hist)))
        hist_local = cv2.resize(hist_local, (w_img, int(h_hist * w_img / w_hist)))

    # 垂直拼接图像和直方图（上下）
    col1 = np.vstack((orig, hist_orig))
    col2 = np.vstack((global_eq, hist_global))
    col3 = np.vstack((local_eq, hist_local))

    # 水平拼接三列
    combined = np.hstack((col1, col2, col3))
    cv2.imwrite(save_path, combined)
    print(f"综合对比图已保存: {save_path}")

######## Task 2 新增函数：绘制灰度图像直方图并保存 ########
def save_histogram(image, title, save_path):
    """
    绘制灰度图像直方图并保存
    Args:
        image: 灰度图 (H, W), dtype=uint8
        title: 图表标题
        save_path: 保存路径（如 .png）
    """
    plt.figure(figsize=(6,4))
    plt.hist(image.ravel(), bins=256, range=(0,256), density=True, color='gray', alpha=0.7)
    plt.title(title)
    plt.xlabel('Pixel value')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
######## Task 2 新增函数：全局直方图均衡化 ########
def histogram_equalization(img):
    """
    对灰度图像进行全局直方图均衡化
    Args:
        img: 灰度图像 (H, W)，dtype=uint8
    Returns:
        均衡化后的图像，dtype=uint8
    """
    
    # 避免除以零
    # 如果图像只有一个灰度值，直接返回原图
    if np.all(img == img.flat[0]):
        return img
    # 计算直方图
    hist, _ = np.histogram(img.flatten(), bins=256, range=[0, 256])
    # 计算累积分布函数 (CDF)
    cdf = hist.cumsum()
    # 归一化到 [0, 255] 并转为 uint8
    cdf_normalized = ((cdf - cdf.min()) * 255 / (cdf.max() - cdf.min())).astype(np.uint8)
    # 用 CDF 作为查找表进行映射
    return cdf_normalized[img]

######## Task 2 新增函数：局部直方图均衡化 ########
def local_histogram_equalization(img, window_size):
    """
    对灰度图像进行局部直方图均衡化（自适应直方图均衡化）
    Args:
        img: 灰度图像 (H, W)，dtype=uint8
        window_size: 正方形邻域的边长（奇数）
    Returns:
        均衡化后的图像，dtype=uint8
    """
    if window_size % 2 == 0:
        raise ValueError("window_size 必须为奇数")
    H, W = img.shape
    pad = window_size // 2
    # 零填充
    padded = np.pad(img, pad, mode='constant', constant_values=0)
    result = np.zeros_like(img, dtype=np.uint8)

    # 遍历每个像素
    for i in range(H):
        for j in range(W):
            # 提取邻域窗口
            window = padded[i:i+window_size, j:j+window_size]
            # 计算窗口内直方图和 CDF
            hist, _ = np.histogram(window.flatten(), bins=256, range=[0, 256])
            cdf = hist.cumsum()
            # 归一化 CDF 到 [0,255]
            if np.all(window == window.flat[0]):
                new_val = img[i, j]
            else:
                cdf_norm = (cdf - cdf.min()) * 255 / (cdf.max() - cdf.min())
                cdf_norm = cdf_norm.astype(np.uint8)
                new_val = cdf_norm[img[i, j]]
            result[i, j] = new_val
    return result

if __name__ == "__main__":
    # ---------- 命令行参数解析 ----------
    parser = argparse.ArgumentParser(description="批量图像直方图均衡化（全局与局部）")
    parser.add_argument("--window", type=int, default=51, 
                        help="局部窗口大小（奇数），默认 51")
    parser.add_argument("--compare", action="store_true", 
                        help="保存对比图（原图、全局均衡、局部均衡三列并排）")
    parser.add_argument("--num", type=int, default=None,
                        help="指定处理的图像数量（默认处理所有）")
    parser.add_argument("--plot_hist", action="store_true",
                    help="同时保存原图、全局均衡化、局部均衡化的直方图")
    parser.add_argument("--combined", action="store_true",
                    help="生成综合对比图（图像+直方图，3x2布局）")
    args = parser.parse_args()

    # 设置输入输出目录
    root_dir = osp.dirname(osp.abspath(__file__))
    output_dir = osp.join(root_dir, "output")
    os.makedirs(output_dir, exist_ok=True)

    # 如果生成对比图，创建 compare 子目录
    if args.compare:
        compare_dir = osp.join(output_dir, "compare")
        os.makedirs(compare_dir, exist_ok=True)

    processed_count = 0
    # 遍历目录下所有图片
    for file in os.listdir(root_dir):
        if file.lower().endswith(('.jpg', '.png', '.jpeg')) and "_gaussian" not in file.lower():
            img_path = osp.join(root_dir, file)
            print(f"正在处理: {img_path}")

            img_bgr = cv2.imread(img_path)
            if img_bgr is None:
                print(f"警告: 无法读取图像 {img_path}，跳过。")
                continue

            # 转为灰度图进行处理（直方图均衡化通常用于灰度图）
            img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

            # 全局直方图均衡化
            global_eq = histogram_equalization(img_gray)

            # 局部直方图均衡化
            local_eq = local_histogram_equalization(img_gray, args.window)

            # 保存结果（灰度图，如需彩色可转为三通道）
            base_name, ext = osp.splitext(file)

            # 保存全局结果
            global_path = osp.join(output_dir, f"{base_name}_global{ext}")
            cv2.imwrite(global_path, global_eq)

            # 保存局部结果
            local_path = osp.join(output_dir, f"{base_name}_local{ext}")
            cv2.imwrite(local_path, local_eq)

            print(f"已保存: {global_path}")
            print(f"已保存: {local_path}")

            # 如果启用对比图，生成三列对比图像
            if args.compare and (args.num is None or processed_count < args.num):
                # 将灰度图转为三通道以便与彩色原图拼接（原图为彩色，我们将其转为灰度显示保持一致）
                # 也可以直接用灰度图拼接，但为了视觉效果，我们将原图也转为灰度显示
                orig_gray_3ch = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
                global_3ch = cv2.cvtColor(global_eq, cv2.COLOR_GRAY2BGR)
                local_3ch = cv2.cvtColor(local_eq, cv2.COLOR_GRAY2BGR)
                compare_img = np.hstack((orig_gray_3ch, global_3ch, local_3ch))
                compare_filename = f"{base_name}_compare{ext}"
                compare_path = osp.join(compare_dir, compare_filename)
                cv2.imwrite(compare_path, compare_img)
                print(f"对比图已保存: {compare_path}")
                
            if args.plot_hist:
                hist_dir = osp.join(output_dir, "histograms")
                os.makedirs(hist_dir, exist_ok=True)

                save_histogram(img_gray, f"Original Histogram - {base_name}",
                               osp.join(hist_dir, f"{base_name}_orig_hist.png"))
                save_histogram(global_eq, f"Global Equalized Histogram - {base_name}",
                               osp.join(hist_dir, f"{base_name}_global_hist.png"))
                save_histogram(local_eq, f"Local Equalized Histogram - {base_name}",
                               osp.join(hist_dir, f"{base_name}_local_hist.png"))
                
            if args.combined:
                # 确保对比图已经生成（如果用户没加 --compare，则仍需生成对比图用于组合）
                if not args.compare:
                    # 临时生成对比图（不保存，仅用于组合）
                    orig_3ch = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
                    global_3ch = cv2.cvtColor(global_eq, cv2.COLOR_GRAY2BGR)
                    local_3ch = cv2.cvtColor(local_eq, cv2.COLOR_GRAY2BGR)
                    # 此处不需要保存对比图，只需得到三列图像
                else:
                    # 如果已经生成对比图，可以直接用已经转好的三列图像（但我们需要的是单个图像，不是拼接后的）
                    orig_3ch = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
                    global_3ch = cv2.cvtColor(global_eq, cv2.COLOR_GRAY2BGR)
                    local_3ch = cv2.cvtColor(local_eq, cv2.COLOR_GRAY2BGR)

                # 直方图路径（如果用户没加 --plot_hist，需先保存直方图）
                if not args.plot_hist:
                    # 临时保存直方图
                    temp_hist_dir = osp.join(output_dir, "temp_hist")
                    os.makedirs(temp_hist_dir, exist_ok=True)
                    hist_orig_path = osp.join(temp_hist_dir, f"{base_name}_orig_hist.png")
                    hist_global_path = osp.join(temp_hist_dir, f"{base_name}_global_hist.png")
                    hist_local_path = osp.join(temp_hist_dir, f"{base_name}_local_hist.png")
                    save_histogram(img_gray, "Original Histogram", hist_orig_path)
                    save_histogram(global_eq, "Global Equalized Histogram", hist_global_path)
                    save_histogram(local_eq, "Local Equalized Histogram", hist_local_path)
                else:
                    # 直方图已保存，获取路径
                    hist_dir = osp.join(output_dir, "histograms")
                    hist_orig_path = osp.join(hist_dir, f"{base_name}_orig_hist.png")
                    hist_global_path = osp.join(hist_dir, f"{base_name}_global_hist.png")
                    hist_local_path = osp.join(hist_dir, f"{base_name}_local_hist.png")

                combined_dir = osp.join(output_dir, "combined")
                os.makedirs(combined_dir, exist_ok=True)
                combined_path = osp.join(combined_dir, f"{base_name}_combined.png")
                save_combined_comparison(orig_3ch, global_3ch, local_3ch,
                                         hist_orig_path, hist_global_path, hist_local_path,
                                         combined_path)

            processed_count += 1
            if args.num is not None and processed_count >= args.num:
                break
