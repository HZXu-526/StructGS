import os
import torch
from super_image import EdsrModel, ImageLoader,DrlnModel
from PIL import Image
from tqdm import tqdm
from super_image import MsrnModel, ImageLoader

def super_resolve_images(input_dir, output_dir, scale=2):
    # Ensure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Get all image filenames
    image_files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f)) and f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    # Display progress bar using tqdm
    for image_name in tqdm(image_files, desc="Processing Images", unit="image"):
        image_path = os.path.join(input_dir, image_name)
        
        # load the image
        image = Image.open(image_path)
        inputs = ImageLoader.load_image(image)
        
        # Load the model and process the image
        # model = EdsrModel.from_pretrained('eugenesiow/edsr-base', scale=scale)
        model = MsrnModel.from_pretrained('eugenesiow/msrn', scale=scale)
        # model = DrlnModel.from_pretrained('eugenesiow/drln', scale=scale)
        preds = model(inputs)
        
        # Save the processed image using the original image name
        output_image_path = os.path.join(output_dir, image_name)
        ImageLoader.save_image(preds, output_image_path)

        # Clear cache to avoid excessive GPU memory usage
        del preds
        torch.cuda.empty_cache()

if __name__ == '__main__':
    import argparse

    # Set command line arguments
    parser = argparse.ArgumentParser(description='Super-Resolution Image Processing')
    parser.add_argument('--input_dir', type=str, required=True, help='Directory of input images')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save super-resolved images')
    parser.add_argument('--scale', type=int, default=2, help='Scale factor for super-resolution')

    args = parser.parse_args()

    super_resolve_images(args.input_dir, args.output_dir, args.scale)





    # python superresolution.py --input_dir outputs/360_v2/bicycle/test/ours_30000/test_preds_-1 --output_dir outputs/360_v2/bicycle/test/ours_30000/super --scale 2
