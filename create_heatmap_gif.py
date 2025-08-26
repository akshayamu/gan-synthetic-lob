import numpy as np
import matplotlib.pyplot as plt
import imageio
import os

def reconstruct_lob_from_features(features):
    """
    Reconstruct LOB snapshot from normalized features
    Features: [bid_prices_norm, ask_prices_norm, bid_volumes_log, ask_volumes_log]
    """
    # Split features
    bid_prices_norm = features[:10]    # Normalized bid prices
    ask_prices_norm = features[10:20]  # Normalized ask prices
    bid_volumes_log = features[20:30]  # Log bid volumes
    ask_volumes_log = features[30:40]  # Log ask volumes
    
    # For visualization, we'll create relative price levels
    bid_levels = np.arange(10)
    ask_levels = np.arange(10)
    
    # Convert log volumes back to actual volumes
    bid_volumes = np.exp(bid_volumes_log) - 1
    ask_volumes = np.exp(ask_volumes_log) - 1
    
    return bid_levels, ask_levels, bid_volumes, ask_volumes

def create_lob_heatmap(bid_levels, ask_levels, bid_volumes, ask_volumes, title="", save_path=None):
    """Create a heatmap-style visualization of the order book"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot bids (left side, blue)
    ax.barh(bid_levels, bid_volumes, color='blue', alpha=0.7, label='Bids')
    
    # Plot asks (right side, red) - negative for left alignment
    ax.barh(ask_levels, ask_volumes, color='red', alpha=0.7, label='Asks')
    
    ax.set_xlabel('Volume')
    ax.set_ylabel('Price Level (Relative)')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

# Load synthetic data
synthetic_data = np.load('data/improved_generated_lob_samples.npy')
print(f"Creating heatmap GIF from {len(synthetic_data)} synthetic samples...")

# Create frames directory
os.makedirs('data/frames', exist_ok=True)

# Generate frames
images = []
for i in range(min(20, len(synthetic_data))):  # First 20 samples
    features = synthetic_data[i]
    
    bid_levels, ask_levels, bid_volumes, ask_volumes = reconstruct_lob_from_features(features)
    
    # Create frame
    save_path = f'data/frames/frame_{i:03d}.png'
    create_lob_heatmap(
        bid_levels, ask_levels, bid_volumes, ask_volumes,
        title=f'Synthetic Order Book Snapshot {i+1}',
        save_path=save_path
    )
    
    # Add to images list
    images.append(imageio.imread(save_path))
    
    if i % 5 == 0:
        print(f"Generated frame {i}/{min(20, len(synthetic_data))}")

# Create GIF
gif_path = 'data/liquidity_heatmap.gif'
imageio.mimsave(gif_path, images, duration=0.5, loop=0)
print(f"GIF saved to {gif_path}")

# Clean up frames (optional)
import shutil
shutil.rmtree('data/frames')
print("Temporary frames cleaned up")

print("✅ Success! You now have:")
print("1. KS-test evaluation results")
print("2. Liquidity heatmap GIF for LinkedIn")
print("3. All results saved in data/ directory")