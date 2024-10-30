import os
import subprocess

# scenes
scenes = ["drjohnson", "playroom"]

# The base paths for the dataset and output directory.
data_base_path = "training_data/3DGS/db"
output_base_path = "outputs/db"

# Iterate through each scene and execute the training, rendering, and evaluation commands in sequence.
for scene in scenes:
    scene_data_path = os.path.join(data_base_path, scene)
    output_path = os.path.join(output_base_path, scene)

    # Training
    train_cmd = f"python train.py -s {scene_data_path} -m {output_path} --eval"
    subprocess.run(train_cmd, shell=True)

    # Rendering
    render_cmd = f"python render.py -m {output_path} --data_device cpu --skip_train"
    subprocess.run(render_cmd, shell=True)

    # Evaluation
    metrics_cmd = f"python metrics.py -m {output_path}"
    subprocess.run(metrics_cmd, shell=True)

    # Super-resolution
    sr_input_dir = os.path.join(output_path, "test/ours_30000/test_preds_-1")
    sr_output_dir = os.path.join(output_path, "test/ours_30000/super")
    super_res_cmd = f"python superresolution.py --input_dir {sr_input_dir} --output_dir {sr_output_dir} --scale 2"
    subprocess.run(super_res_cmd, shell=True)

    # Super-resolution metrics
    sr_gt_dir = os.path.join(output_path, "test/ours_30000/gt_-1")
    sr_renders_dir = sr_output_dir
    metrics_super_cmd = f"python metrics_super.py --gt_dir {sr_gt_dir} --renders_dir {sr_renders_dir}"
    subprocess.run(metrics_super_cmd, shell=True)

