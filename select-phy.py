import cv2
import numpy as np
import os
import shutil
import argparse
from pathlib import Path
import logging
from datetime import datetime
import torch

_local_niqe_metric = None
_local_device = None

def get_niqe_model():
    global _local_niqe_metric, _local_device
    if _local_niqe_metric is None:
        import pyiqa
        import torch
        _local_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _local_niqe_metric = pyiqa.create_metric('niqe', device=_local_device)
    return _local_niqe_metric, _local_device

def parse_args():
    parser = argparse.ArgumentParser(description="Image Dataset Filtering (Single Thread)")
    parser.add_argument("--input_dir", type=str, default="./unsplash_images", help="源文件夹路径")
    parser.add_argument("--output_dir", type=str, default="./filtered_dataset/phy-niqe", help="筛选后的存放路径")
    parser.add_argument("--min_res", type=int, default=1080, help="最小分辨率（宽或高）")
    parser.add_argument("--min_laplacian", type=float, default=200.0, help="最小拉普拉斯方差（清晰度阈值）")
    parser.add_argument("--min_entropy", type=float, default=5.0, help="最小信息熵（细节丰富度阈值）")
    parser.add_argument("--min_bpp", type=float, default=0.5, help="最小BPP（压缩程度阈值）")
    parser.add_argument("--max_niqe", type=float, default=5.0, help="最大NIQE值（值越小画质越好）")
    parser.add_argument("--disable_niqe", action="store_true", help="是否禁用NIQE计算（为了提速）")
    return vars(parser.parse_args())

def setup_logger(output_dir):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    log_filename = f"filter_job_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    log_filepath = Path(output_dir) / log_filename

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filepath, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    return log_filepath

def calculate_entropy(img_gray):
    """计算图像一维信息熵"""
    hist = cv2.calcHist([img_gray], [0], None, [256], [0, 256])
    hist = hist.ravel() / hist.sum()
    entropy = -np.sum(hist * np.log2(hist + 1e-7))
    return entropy

def process_single_image(img_path, config):
    try:
        file_size_bits = os.path.getsize(img_path) * 8
        img = cv2.imread(str(img_path))
        if img is None:
            return (img_path, "Fail: Read Error")
        
        h, w = img.shape[:2]
        
        # 分辨率过滤
        if w < config["min_res"] or h < config["min_res"]:
            return (img_path, "Fail: Resolution")

        # BPP 过滤
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
            niqe_metric, device = get_niqe_model()
            
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
    
    log_file = setup_logger(config["output_dir"])
    out_path = Path(config["output_dir"])
    out_path.mkdir(parents=True, exist_ok=True)
    
    input_path = Path(config["input_dir"])
    if not input_path.exists():
        logging.error(f"输入路径不存在: {input_path}")
        return

    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
    all_imgs = [
        p for p in input_path.rglob('*') 
        if p.is_file() and p.suffix.lower() in image_extensions
    ]
    
    logging.info(f"找到 {len(all_imgs)} 张待处理图片，开始筛选...")
    if not config["disable_niqe"]:
        logging.info("NIQE 检测已开启")

    passed_count = 0
    rejected_count = 0

    # --- 核心改动：用单线程的 for 循环替代多进程 ---
    for img_path in all_imgs:
        # 直接调用处理函数
        result = process_single_image(img_path, config)
        
        # 解析返回结果
        img_path, status = result
        relative_path = img_path.relative_to(input_path)
        
        # 实时处理通过或淘汰的逻辑
        if status == "Pass":
            try:
                dest_path = out_path / relative_path
                dest_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(img_path, dest_path)
                passed_count += 1
                logging.info(f"[PASS] 保留: {relative_path}")
            except Exception as e:
                logging.error(f"复制文件失败 {img_path.name}: {e}")
        else:
            rejected_count += 1
            logging.info(f"[REJECT] 淘汰: {relative_path} | 原因: {status}")

    # 打印最终报告
    print("\n" + "=" * 40)
    print("处理完成！数据统计如下：")
    print("=" * 40)
    print(f"原始图片数: {len(all_imgs)}")
    print(f"通过筛选数: {passed_count}")
    print(f"淘汰图片数: {rejected_count}")
    if len(all_imgs) > 0:
        print(f"淘汰比例:   {((len(all_imgs) - passed_count) / len(all_imgs)) * 100:.2f}%")
    print("=" * 40)

if __name__ == "__main__":
    main()
