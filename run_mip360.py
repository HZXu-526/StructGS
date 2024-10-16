import os
import subprocess

# scenes
scenes = ["bicycle", "bonsai", "counter", "flowers", "garden", "kitchen", "room", "stump", "treehill"]

# The base paths for the dataset and output directory.
data_base_path = r"E:\PhD\training_data\3DGS\360_v2"
output_base_path = "outputs/360_v2"

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