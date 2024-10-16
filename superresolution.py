import os
import torch
from super_image import EdsrModel, ImageLoader,DrlnModel
from PIL import Image
from tqdm import tqdm
from super_image import MsrnModel, ImageLoader

def super_resolve_images(input_dir, output_dir, scale=2):
    # 确保输出目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 获取所有图像文件名
    image_files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f)) and f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    # 使用tqdm显示进度条
    for image_name in tqdm(image_files, desc="Processing Images", unit="image"):
        image_path = os.path.join(input_dir, image_name)
        
        # 加载图像
        image = Image.open(image_path)
        inputs = ImageLoader.load_image(image)
        
        # 加载模型并处理图像
        # model = EdsrModel.from_pretrained('eugenesiow/edsr-base', scale=scale)
        model = MsrnModel.from_pretrained('eugenesiow/msrn', scale=scale)
        # model = DrlnModel.from_pretrained('eugenesiow/drln', scale=scale)
        preds = model(inputs)
        
        # 使用原图名称保存处理后的图像
        output_image_path = os.path.join(output_dir, image_name)
        ImageLoader.save_image(preds, output_image_path)

        # 清理缓存，避免显存占用过多
        del preds
        torch.cuda.empty_cache()

if __name__ == '__main__':
    import argparse

    # 设置命令行参数
    parser = argparse.ArgumentParser(description='Super-Resolution Image Processing')
    parser.add_argument('--input_dir', type=str, required=True, help='Directory of input images')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save super-resolved images')
    parser.add_argument('--scale', type=int, default=2, help='Scale factor for super-resolution')

    args = parser.parse_args()

    # 调用超分函数
    super_resolve_images(args.input_dir, args.output_dir, args.scale)





    # python superresolution.py --input_dir outputs/bungeenerf/rome/baseline/2024-08-20_22:00:21/test/ours_30000/renders --output_dir outputs/bungeenerf/rome/baseline/2024-08-20_22:00:21/test/ours_30000/super --scale 2






    # python superresolution.py --input_dir outputs/bungeenerf/rome/baseline/2024-08-20_22:00:21/test/ours_30000/renders --output_dir outputs/bungeenerf/rome/baseline/2024-08-20_22:00:21/test/ours_30000/super --scale 2

