import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import random

def robust_stretch(band, p_lower=2, p_upper=98):
    """
    Applies a percentile stretch to a satellite band for visualization.
    Raw DNs often look pitch black without this.
    """
    if not band.any(): 
        return np.zeros_like(band)
    b_min, b_max = np.percentile(band[band > 0], (p_lower, p_upper))
    return np.clip((band - b_min) / (b_max - b_min), 0, 1)

def save_publication_tiles(data_dir="data/tiles", out_dir="output/tiles_visualization", num_samples=3):
    """
    Randomly selects .npz tiles, generates publication-grade plots, 
    and saves them to a specified directory.
    """
    tile_dir = Path(data_dir)
    save_dir = Path(out_dir)
    
    # Create the output directory if it doesn't exist
    save_dir.mkdir(parents=True, exist_ok=True)
    
    npz_files = list(tile_dir.glob("*.npz"))
    
    if not npz_files:
        print(f"No .npz files found in {tile_dir}. Check your ingestion output path.")
        return
    
    samples = random.sample(npz_files, min(num_samples, len(npz_files)))
    
    # Optional: Set global Matplotlib parameters for publication aesthetics
    plt.rcParams.update({
        'font.size': 12,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'figure.facecolor': 'white'
    })
    
    for npz_path in samples:
        # 1. Load the tile
        data = np.load(npz_path)
        
        # 2. Extract Base Data
        bands = data["bands"] 
        ndvi = data["ndvi"]
        canopy = data["canopy"]
        row, col = data["row"], data["col"]
        bounds = data["bounds"]
        
        has_mask = "mask" in data
        has_chm = "chm" in data
        
        # 3. Prepare the False Color Composite (CIR)
        nir_stretched = robust_stretch(bands[2])
        red_stretched = robust_stretch(bands[1])
        green_stretched = robust_stretch(bands[0])
        cir_composite = np.dstack((nir_stretched, red_stretched, green_stretched))
        
        # 4. Setup the Plot dynamically
        num_plots = 3 + int(has_mask) + int(has_chm)
        fig, axes = plt.subplots(1, num_plots, figsize=(4 * num_plots, 5))
        if num_plots == 1: axes = [axes] 
        
        ax_idx = 0
        
        # Plot CIR
        axes[ax_idx].imshow(cir_composite)
        axes[ax_idx].set_title("CIR Composite")
        axes[ax_idx].axis("off")
        ax_idx += 1
        
        # Plot NDVI
        im_ndvi = axes[ax_idx].imshow(ndvi, cmap="RdYlGn", vmin=-1, vmax=1)
        axes[ax_idx].set_title("NDVI")
        axes[ax_idx].axis("off")
        fig.colorbar(im_ndvi, ax=axes[ax_idx], fraction=0.046, pad=0.04)
        ax_idx += 1
        
        # Plot Canopy
        axes[ax_idx].imshow(canopy, cmap="gray")
        axes[ax_idx].set_title("Canopy Proxy")
        axes[ax_idx].axis("off")
        ax_idx += 1
        
        # Plot OSM Road Mask
        if has_mask:
            axes[ax_idx].imshow(data["mask"], cmap="magma")
            axes[ax_idx].set_title("OSM Road Label")
            axes[ax_idx].axis("off")
            ax_idx += 1
            
        # Plot CHM
        if has_chm:
            im_chm = axes[ax_idx].imshow(data["chm"], cmap="viridis")
            axes[ax_idx].set_title("CHM (Height)")
            axes[ax_idx].axis("off")
            fig.colorbar(im_chm, ax=axes[ax_idx], fraction=0.046, pad=0.04)
            ax_idx += 1
            
        plt.tight_layout()
        
        # 5. Save as Publication-Grade Image (300 DPI PNG)
        # Naming the file based on its original tile name
        save_name = f"{npz_path.stem}_plot.png"
        save_path = save_dir / save_name
        
        fig.savefig(save_path, dpi=300, bbox_inches="tight", format="png", transparent=False)
        plt.close(fig) # Critical: frees memory so notebook doesn't crash
        
        # 6. Print Metadata to Console
        print(f"Saved: {save_path}")
        print(f"  Origin (Row, Col): ({row}, {col})")
        print(f"  Bounds: [{bounds[0]:.5f}, {bounds[1]:.5f}, {bounds[2]:.5f}, {bounds[3]:.5f}]")
        if has_mask:
            road_pixels = data["mask"].sum()
            total_pixels = data["mask"].size
            print(f"  Road Density: {100 * road_pixels / total_pixels:.2f}%")
        print("-" * 50)

# Execute the function
save_publication_tiles(data_dir="data/tiles", out_dir="output/tiles_visualization", num_samples=5)