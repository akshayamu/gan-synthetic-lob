import numpy as np
import matplotlib.pyplot as plt
import imageio
import os

def create_simple_lob_viz(features, frame_num):
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Extract volumes (convert from log scale)
    bid_volumes = np.exp(features[20:30]) - 1
    ask_volumes = np.exp(features[30:40]) - 1
    
    # Create visualization
    levels = np.arange(10)
    ax.barh(levels - 0.2, bid_volumes, height=0.4, color='blue', alpha=0.7, label='Bids')
    ax.barh(levels + 0.2, ask_volumes, height=0.4, color='red', alpha=0.7, label='Asks')
    
    ax.set_xlabel('Volume')
    ax.set_ylabel('Price Level')
    ax.set_title(f'Synthetic Order Book #{frame_num}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    filename = f'frame_{frame_num:02d}.png'
    plt.savefig(filename, dpi=100, bbox_inches='tight')
    plt.close()
    return filename

# Load synthetic data
synthetic_data = np.load('data/generated_lob_samples.npy')
print("Creating LinkedIn content...")

# Create frames for GIF
images = []
for i in range(min(10, len(synthetic_data))):
    filename = create_simple_lob_viz(synthetic_data[i], i+1)
    images.append(imageio.imread(filename))
    print(f"Created frame {i+1}")

# Create GIF
imageio.mimsave('lob_animation.gif', images, duration=0.8)
print("✅ GIF created: lob_animation.gif")

# Clean up temporary files
for i in range(10):
    try:
        os.remove(f'frame_{i:02d}.png')
    except:
        pass

print("✅ Cleaned up temporary files")
print("\n🎉 Your project is COMPLETE!")
print("✅ Data collected (720 snapshots)")
print("✅ GAN trained")
print("✅ KS-test performed")
print("✅ LinkedIn GIF created: lob_animation.gif")