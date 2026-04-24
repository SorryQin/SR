# sorryqin
import cv2
import numpy as np
import os
import shutil
import argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import logging


logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def parse_args():
    parser = argparse.ArgumentParser(description="Image Dataset Filtering")
    parser.add_argument("--input_dir", type=str, default="./unsplash_images/cyberpunk", help="源文件夹路径")
    parser.add_argument("--output_dir", type=str, default="./filtered_dataset/cyberpunk", help="筛选后的存放路径")
    parser.add_argument("--min_res", type=int, default=1080, help="最小分辨率（宽或高）")
    parser.add_argument("--min_laplacian", type=float, default=200.0, help="最小拉普拉斯方差（清晰度阈值）")
    parser.add_argument("--min_entropy", type=float, default=5.0, help="最小信息熵（细节丰富度阈值）")
    parser.add_argument("--min_bpp", type=float, default=0.5, help="最小BPP（压缩程度阈值）")
    parser.add_argument("--max_niqe", type=float, default=5.0, help="最大NIQE值（值越小画质越好）")
    parser.add_argument("--disable_niqe", action="store_true", help="是否禁用NIQE计算（为了提速）")
    return vars(parser.parse_args())

def calculate_entropy(img_gray):
    """计算图像一维信息熵"""
    hist = cv2.calcHist([img_gray], [0], None, [256], [0, 256])
    hist = hist.ravel() / hist.sum()
    # 加上 1e-7 防止 log2(0) 报错
    entropy = -np.sum(hist * np.log2(hist + 1e-7))
    return entropy

def process_single_image(img_path, config):
    """
    处理单张图片的函数。
    注意：这里的设计是“层层递进”的漏斗模式，计算量从上到下依次增加。
    """
    try:
        file_size_bits = os.path.getsize(img_path) * 8
        img = cv2.imread(str(img_path))
        if img is None:
            return (img_path, "Fail: Read Error")
        
        h, w = img.shape[:2]
        
        # 分辨率过滤
        if w < config["min_res"] or h < config["min_res"]:
            return (img_path, "Fail: Resolution")

        bpp = file_size_bits / (w * h)
        if bpp < config["min_bpp"]:
            return (img_path, "Fail: BPP")

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 拉普拉斯方差
        lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        if lap_var < config["min_laplacian"]:
            return (img_path, "Fail: Laplacian")

        # 信息熵
        entropy = calculate_entropy(gray)
        if entropy < config["min_entropy"]:
            return (img_path, "Fail: Entropy")

        # NIQE
        if not config["disable_niqe"]:
            import pyiqa
            import torch

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            niqe_metric = pyiqa.create_metric('niqe', device=device)
            
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).unsqueeze(0).float() / 255.0
            img_tensor = img_tensor.to(device)
            
            niqe_score = niqe_metric(img_tensor).item()
            
            if niqe_score > config["max_niqe"]:
                return (img_path, f"Fail: NIQE ({niqe_score:.2f})")

        return (img_path, "Pass")

    except Exception as e:
        return (img_path, f"Error: {str(e)}")

def main():
    config = parse_args()
    
    out_path = Path(config["output_dir"])
    out_path.mkdir(parents=True, exist_ok=True)
    
    input_path = Path(config["input_dir"])
    if not input_path.exists():
        logging.error(f"输入路径不存在: {input_path}")
        return

    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
    all_imgs = [input_path / f for f in os.listdir(input_path) 
                if f.lower().endswith(image_extensions)]
    
    logging.info(f"找到 {len(all_imgs)} 张待处理图片，开始筛选...")
    if not config["disable_niqe"]:
        logging.info("NIQE 检测已开启")

    # 使用 functools.partial 将 config 绑定到函数中，方便多进程调用
    process_func = partial(process_single_image, config=config)

    passed_count = 0
    # ProcessPoolExecutor 默认使用系统的 CPU 核心数
    with ProcessPoolExecutor() as executor:
        results = list(executor.map(process_func, all_imgs))

    for result in results:
        img_path, status = result
        if status == "Pass":
            try:
                shutil.copy(img_path, out_path / img_path.name)
                passed_count += 1
            except Exception as e:
                logging.error(f"复制文件失败 {img_path.name}: {e}")
        # 淘汰的详细原因
        # else:
        #     logging.info(f"淘汰 {img_path.name}: {status}")

    # 打印最终报告
    print("\n" + "=" * 40)
    print("处理完成！数据统计如下：")
    print("=" * 40)
    print(f"原始图片数: {len(all_imgs)}")
    print(f"通过筛选数: {passed_count}")  
    if len(all_imgs) > 0:
        print(f"淘汰比例:   {((len(all_imgs) - passed_count) / len(all_imgs)) * 100:.2f}%")
    print("=" * 40)

if __name__ == "__main__":
    main()
