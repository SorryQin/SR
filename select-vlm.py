import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import os
import json
import shutil
from pathlib import Path
from tqdm import tqdm
import re
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="VLM 多显卡并行审计工具")
    parser.add_argument("--input", type=str, required=True, help="输入文件夹路径")
    parser.add_argument("--output", type=str, required=True, help="输出结果根路径")
    parser.add_argument("--model", type=str, default="../Qwen/qwen/Qwen2.5-VL-7B-Instruct", help="模型存放路径")
    parser.add_argument("--threshold", type=int, default=8, help="通过分数阈值 (0-10)")
    return parser.parse_args()

# 使用打分制，控制力更强
def vlm_inference(model, processor, img_path):
    prompt = (
        "你是一个苛刻的图像质量审计专家。你的任务是筛选出用于‘顶级超分训练’的底图。\n"
        "【打分准则】（从 10 分开始扣除）：\n"
        "1. 模糊/重影：只要有轻微模糊，综合得分不得超过 6 分。\n"
        "2. 伪影/噪点：发现 JPEG 压缩色块或人工锐化白边，直接降至 5 分以下。\n"
        "3. AI 痕迹：若纹理不自然（扭曲、崩坏），直接判定为 2 分。\n"
        "4. 只有各方面都无可挑剔、纹理极其锐利的图才能给 9-10 分。\n\n"
        "请按以下步骤思考：\n"
        "步骤 1：仔细寻找图中的缺陷（如噪点、模糊、伪影）。\n"
        "步骤 2：根据缺陷严重程度进行扣分。\n\n"
        "请严格按 JSON 返回，final_score 必须为整数：\n"
        '{"defects": "发现的缺陷描述", "clarity_score": 0, "purity_score": 0, "final_score": 0, "reason": "综合评价"}'
    )
    
    messages = [{"role": "user", "content": [
        {"type": "image", "image": str(img_path)},
        {"type": "text", "text": prompt}
    ]}]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, _ = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, padding=True, return_tensors="pt").to(model.device)

    generated_ids = model.generate(
        **inputs, 
        max_new_tokens=200, 
        do_sample=False,  # 关键：关闭随机采样
        temperature=0.0   # 关键：强制确定性输出
    )
    output_text = processor.batch_decode(generated_ids[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0]
    
    try:
        match = re.search(r"\{.*\}", output_text, re.DOTALL)
        if match:
            return json.loads(match.group())
        else:
            return {"final_score": 0, "reason": f"解析失败: {output_text[:50]}"}
    except Exception as e:
        return {"final_score": 0, "reason": f"异常: {str(e)}"}

def main():
    args = parse_args()
    
    input_root = Path(args.input)
    output_base = Path(args.output)
    output_base.mkdir(parents=True, exist_ok=True)
    result_json = output_base / "audit_log.json"

    print(f"正在显卡 {torch.cuda.current_device()} 上加载模型...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model, torch_dtype="auto", device_map="auto", trust_remote_code=True
    )
    processor = AutoProcessor.from_pretrained(args.model)

    # 递归扫描
    image_extensions = ('*.jpg', '*.jpeg', '*.png', '*.webp')
    image_files = []
    for ext in image_extensions:
        image_files.extend(list(input_root.rglob(ext)))
        image_files.extend(list(input_root.rglob(ext.upper())))
    image_files = list(set(image_files))

    results_data = []
    print(f"开始审计: {args.input} | 目标数量: {len(image_files)}")
    
    for img_path in tqdm(image_files):
        audit_result = vlm_inference(model, processor, img_path)
        score = audit_result.get("final_score", 0)
        rel_path = img_path.relative_to(input_root)
        
        results_data.append({
            "image_path": str(rel_path),
            "score": score,
            "details": audit_result
        })

        if score >= args.threshold:
            dest_path = output_base / rel_path
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(img_path, dest_path)
        
        with open(result_json, 'w', encoding='utf-8') as f:
            json.dump(results_data, f, ensure_ascii=False, indent=4)

    print(f"任务完成！结果保存在: {args.output}")

if __name__ == "__main__":
    main()
