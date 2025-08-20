from pathlib import Path
import torch

from lerobot.configs.types import FeatureType
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.utils import dataset_to_policy_features
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

def main():
    # Create output directory
    output_directory = Path("outputs/train/abb_pick_place_smolvla")
    output_directory.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda:0")

    training_steps = 5000
    log_freq = 1

    # Dataset metadata
    dataset_metadata = LeRobotDatasetMetadata("AIR-AUDI/abb_pick_place_all")
    features = dataset_to_policy_features(dataset_metadata.features)
    output_features = {k: ft for k, ft in features.items() if ft.type is FeatureType.ACTION}
    input_features = {k: ft for k, ft in features.items() if k not in output_features}

    # Delta timestamps aligned to 30 Hz recording frequency
    # Each timestep = 1/30 = 0.0333... seconds
    # Recording frequency: 30 Hz (from sync node timer: 1.0 / 30.0)
    # Conservative - 0.5 second prediction horizon (recommended for pick-and-place)
    delta_timestamps = {
        "observation.images.cam0": [-0.0667, -0.0333, 0.0000],  # 2 past + current (3 timesteps)
        "observation.images.cam2": [-0.0667, -0.0333, 0.0000],  # 2 past + current (3 timesteps)
        "observation.images.cam4": [-0.0667, -0.0333, 0.0000],  # 2 past + current (3 timesteps)
        "observation.state": [-0.0667, -0.0333, 0.0000],        # 2 past + current (3 timesteps)
        "action": [
            -0.0333, 0.0000,  # 1 past + current
            0.0333, 0.0667, 0.1000, 0.1333, 0.1667, 0.2000,     # 0.033s - 0.2s   (6 steps)
            0.2333, 0.2667, 0.3000, 0.3333, 0.3667, 0.4000,     # 0.233s - 0.4s   (6 steps)
            0.4333, 0.4667, 0.5000                               # 0.433s - 0.5s   (3 steps)
        ],  # Total: 17 timesteps (1 past + current + 15 future = 0.5s prediction)
    }
    
    # Uncomment this for longer prediction (complex movements)
    """
    delta_timestamps = {
        "observation.images.cam0": [-0.1000, -0.0667, -0.0333, 0.0000],  # 3 past + current
        "observation.images.cam2": [-0.1000, -0.0667, -0.0333, 0.0000],
        "observation.images.cam4": [-0.1000, -0.0667, -0.0333, 0.0000],
        "observation.state": [-0.1000, -0.0667, -0.0333, 0.0000],
        "action": [
            -0.0667, -0.0333, 0.0000,  # 2 past + current
            0.0333, 0.0667, 0.1000, 0.1333, 0.1667, 0.2000, 0.2333, 0.2667,
            0.3000, 0.3333, 0.3667, 0.4000, 0.4333, 0.4667, 0.5000, 0.5333,
            0.5667, 0.6000, 0.6333, 0.6667, 0.7000, 0.7333, 0.7667, 0.8000
        ],  # Total: 27 timesteps (2 past + current + 24 future = 0.8s prediction)
    }
    """
    
    # Calculate action sequence length from delta timestamps
    action_sequence_length = len(delta_timestamps["action"])
    print(f"Action sequence length: {action_sequence_length}")
    print(f"Prediction horizon: {max(delta_timestamps['action']):.3f} seconds")
    print(f"Future timesteps: {action_sequence_length - 2}")  # -2 for past and current
    
    cfg = SmolVLAConfig(
        input_features=input_features, 
        output_features=output_features,
        resize_imgs_with_padding=(512, 512),
        chunk_size=action_sequence_length,     
        n_action_steps=action_sequence_length, 
    )
    
    print(f"SmolVLA chunk_size: {cfg.chunk_size}")
    print(f"SmolVLA n_action_steps: {cfg.n_action_steps}")
    
    policy = SmolVLAPolicy(cfg, dataset_stats=dataset_metadata.stats)
    policy.train()
    policy.to(device)

    # Dataset with corrected delta timestamps
    dataset = LeRobotDataset(
        "AIR-AUDI/abb_pick_place_all", 
        delta_timestamps=delta_timestamps, 
        tolerance_s=1,
    )

    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-4)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=4,
        batch_size=64,
        shuffle=True,
        pin_memory=device.type != "cpu",
        drop_last=True,
    )

    policy.train()
    policy.to(device)

    print(f"Starting training with {len(dataset)} samples...")
    print(f"Batch size: 64, Training steps: {training_steps}")

    for step, batch in enumerate(dataloader):
        batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
        
        # Debug: Print batch shapes on first iteration
        if step == 0:
            print("\nBatch shapes:")
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    print(f"  {key}: {value.shape}")
                else:
                    print(f"  {key}: {type(value)} - {value}")
        
        loss, loss_dict = policy.forward(batch)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if step % log_freq == 0:
            print(f"Step: {step}, Loss: {loss.item():.3f}")
        if step >= training_steps:
            break

    # Save a policy checkpoint
    policy.save_pretrained(output_directory)
    print(f"Model saved to {output_directory}")


if __name__ == "__main__":
    main()