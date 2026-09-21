import torch

checkpoint = torch.load(
    "experiments/lafan1_detail_model_release/checkpoint_lafan1_detail_model_release.pth",
    map_location="cpu")

print(f"checkpoint epoch: {checkpoint["epoch"]}")
print(f"checkpoint iteration: {checkpoint["iteration"]}")
