from pathlib import Path
import os
from PIL import Image
import torch
import torchvision.transforms.functional as tf
from utils.loss_utils import ssim
from lpipsPyTorch import lpips
from utils.image_utils import psnr
from argparse import ArgumentParser
import json
from tqdm import tqdm

from pathlib import Path
import os
from PIL import Image
import torch
import torchvision.transforms.functional as tf
from utils.loss_utils import ssim
from lpipsPyTorch import lpips
from utils.image_utils import psnr
from argparse import ArgumentParser
import json
from tqdm import tqdm

def read_images(gt_dir, renders_dir):
    renders = []
    gts = []
    image_names = []
    for fname in os.listdir(renders_dir):
        if fname.endswith(('.png', '.jpg', '.jpeg')):
            render_path = Path(renders_dir) / fname
            gt_path = Path(gt_dir) / fname

            render = Image.open(render_path)
            gt = Image.open(gt_path)

            # Uniformly resize to a certain dimension using the same interpolation method
            common_size = (512, 512)
            render = render.resize(common_size, Image.BICUBIC)
            gt = gt.resize(common_size, Image.BICUBIC)

            renders.append(tf.to_tensor(render).unsqueeze(0)[:, :3, :, :].cuda())
            gts.append(tf.to_tensor(gt).unsqueeze(0)[:, :3, :, :].cuda())
            image_names.append(fname)
    return renders, gts, image_names

def calculate_metrics(renders, gts, image_names):
    ssims = []
    psnrs = []
    lpipss = []

    for idx in tqdm(range(len(renders)), desc="Metric evaluation progress"):
        # Adjust the SSIM window size or use other parameters
        ssims.append(ssim(renders[idx], gts[idx], window_size=11)) # Example of adjusting window size

        psnrs.append(psnr(renders[idx], gts[idx]))
        
        # Calculate LPIPS using different network types
        lpipss.append(lpips(renders[idx], gts[idx], net_type='alex'))  # Use different network types

    avg_ssim = torch.tensor(ssims).mean().item()
    avg_psnr = torch.tensor(psnrs).mean().item()
    avg_lpips = torch.tensor(lpipss).mean().item()

    per_view_metrics = {
        "SSIM": {name: ssim for ssim, name in zip(ssims, image_names)},
        "PSNR": {name: psnr for psnr, name in zip(psnrs, image_names)},
        "LPIPS": {name: lp for lp, name in zip(lpipss, image_names)}
    }

    return avg_ssim, avg_psnr, avg_lpips, per_view_metrics

def evaluate(gt_dir, renders_dir):
    renders, gts, image_names = read_images(gt_dir, renders_dir)
    
    avg_ssim, avg_psnr, avg_lpips, per_view_metrics = calculate_metrics(renders, gts, image_names)

    # print metric scores
    print(f"SSIM : {avg_ssim:.7f}")
    print(f"PSNR : {avg_psnr:.7f}")
    print(f"LPIPS: {avg_lpips:.7f}")
    
    output_dir = Path(gt_dir).parents[2]
    
    results = {
        "SSIM": avg_ssim,
        "PSNR": avg_psnr,
        "LPIPS": avg_lpips
    }

    # Save results as JSON
    os.makedirs(output_dir, exist_ok=True)
    with open(output_dir / "results_super.json", 'w') as f:
        json.dump(results, f, indent=4)

    print(f"Results saved to {output_dir / 'results_super.json'}")

if __name__ == "__main__":
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    parser = ArgumentParser(description="Script for calculating SSIM, PSNR, and LPIPS metrics")
    parser.add_argument('--gt_dir', required=True, type=str, help="Directory containing ground truth images")
    parser.add_argument('--renders_dir', required=True, type=str, help="Directory containing rendered images")
    args = parser.parse_args()

    evaluate(args.gt_dir, args.renders_dir)


    # python metrics_super.py --gt_dir outputs/360_v2/bicycle/test/ours_30000/gt_-1 --renders_dir outputs/360_v2/bicycle/test/ours_30000/super
