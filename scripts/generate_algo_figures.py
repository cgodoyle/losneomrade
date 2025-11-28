import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import os

def create_grid(size=10):
    return np.zeros((size, size), dtype=int)

def plot_grid(ax, grid, title, legend_elements=None):
    # Colormap
    # 0: Background (White)
    # 1: Release (Red)
    # 2: Candidate (Blue)
    # 3: Rejected (Gray)
    # 4: Redundant Candidate (Purple)
    # 5: Checked/Visited Marker (Dark Green - used for border/overlay conceptually, but here distinct color for simplicity)
    
    cmap = mcolors.ListedColormap(['white', '#ff4d4d', '#4d79ff', '#b3b3b3', '#9933ff'])
    bounds = [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5]
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    ax.imshow(grid, cmap=cmap, norm=norm, origin='upper')
    ax.grid(which='major', axis='both', linestyle='-', color='k', linewidth=0.5)
    ax.set_xticks(np.arange(-0.5, 10, 1))
    ax.set_yticks(np.arange(-0.5, 10, 1))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_title(title, fontsize=10, fontweight='bold')

    if legend_elements:
        ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.0, -0.05), ncol=2, fontsize=8)

def generate_original_algo_figure(output_path):
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    # --- Panel 1: Initial State ---
    grid1 = create_grid()
    # Initial Release (Center)
    grid1[4:6, 4:6] = 1 
    # Candidates (Rim)
    grid1[3, 4:6] = 2; grid1[6, 4:6] = 2
    grid1[4:6, 3] = 2; grid1[4:6, 6] = 2
    # Corners
    grid1[3, 3] = 2; grid1[3, 6] = 2; grid1[6, 3] = 2; grid1[6, 6] = 2
    
    plot_grid(axes[0], grid1, "Step 0: Initial State")

    # --- Panel 2: Iteration 1 (Expand & Check) ---
    grid2 = grid1.copy()
    # Assume some candidates pass (become Release - 1)
    # Top and Bottom pass
    grid2[3, 4:6] = 1; grid2[6, 4:6] = 1
    # Left and Right fail (become Rejected - 3)
    grid2[4:6, 3] = 3; grid2[4:6, 6] = 3
    # Corners fail
    grid2[3, 3] = 3; grid2[3, 6] = 3; grid2[6, 3] = 3; grid2[6, 6] = 3
    
    plot_grid(axes[1], grid2, "Step 1: Expand & Check")

    # --- Panel 3: Iteration 2 (Redundancy) ---
    grid3 = grid2.copy()
    # In original algo, we dilate the ENTIRE release area again.
    # The release area now includes the original center + top/bottom rows.
    # Neighbors of the original center (Left/Right) are checked AGAIN.
    
    # Re-mark the rejected pixels as "Redundant Candidates" (4)
    # Left and Right neighbors of original center
    grid3[4:6, 3] = 4; grid3[4:6, 6] = 4
    # Corners (neighbors of the new top/bottom release pixels)
    grid3[3, 3] = 4; grid3[3, 6] = 4; grid3[6, 3] = 4; grid3[6, 6] = 4
    
    # New candidates from the new release pixels (Top/Bottom)
    # Top of Top
    grid3[2, 4:6] = 2
    # Bottom of Bottom
    grid3[7, 4:6] = 2
    
    plot_grid(axes[2], grid3, "Step 2: Redundant Re-check")

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#ff4d4d', edgecolor='k', label='Release Area'),
        Patch(facecolor='#4d79ff', edgecolor='k', label='Candidate'),
        Patch(facecolor='#b3b3b3', edgecolor='k', label='Rejected'),
        Patch(facecolor='#9933ff', edgecolor='k', label='Redundant Check'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=4, bbox_to_anchor=(0.5, -0.05))
    
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"Generated {output_path}")

def generate_bfs_algo_figure(output_path):
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    # --- Panel 1: Initial State ---
    grid1 = create_grid()
    # Initial Release
    grid1[4:6, 4:6] = 1
    # Candidates
    grid1[3, 4:6] = 2; grid1[6, 4:6] = 2
    grid1[4:6, 3] = 2; grid1[4:6, 6] = 2
    grid1[3, 3] = 2; grid1[3, 6] = 2; grid1[6, 3] = 2; grid1[6, 6] = 2
    
    plot_grid(axes[0], grid1, "Step 0: Initial State")

    # --- Panel 2: Iteration 1 (Check & Mark Visited) ---
    grid2 = grid1.copy()
    # Same pass/fail logic
    # Top/Bottom pass -> Release (1)
    grid2[3, 4:6] = 1; grid2[6, 4:6] = 1
    # Left/Right fail -> Rejected (3)
    grid2[4:6, 3] = 3; grid2[4:6, 6] = 3
    grid2[3, 3] = 3; grid2[3, 6] = 3; grid2[6, 3] = 3; grid2[6, 6] = 3
    
    # Visually, we want to imply these are all "Visited". 
    # The color coding (Red/Gray) implies visited.
    
    plot_grid(axes[1], grid2, "Step 1: Check & Mark Visited")

    # --- Panel 3: Iteration 2 (Efficient Expansion) ---
    grid3 = grid2.copy()
    # BFS only expands from NEWLY added pixels (Top/Bottom rows).
    # It does NOT expand from the original center again.
    # So Left/Right neighbors (Rejected) are NOT candidates again.
    
    # New candidates from Top
    grid3[2, 4:6] = 2
    # New candidates from Bottom
    grid3[7, 4:6] = 2
    
    # The rejected pixels stay Rejected (Gray), they do NOT turn Purple.
    
    plot_grid(axes[2], grid3, "Step 2: Efficient Expansion")

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#ff4d4d', edgecolor='k', label='Release Area'),
        Patch(facecolor='#4d79ff', edgecolor='k', label='Candidate'),
        Patch(facecolor='#b3b3b3', edgecolor='k', label='Rejected (Visited)'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=3, bbox_to_anchor=(0.5, -0.05))

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"Generated {output_path}")

def generate_memory_explosion_figure(output_path):
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    
    # --- Panel 1: The Problem (Full Matrix) ---
    ax1 = axes[0]
    ax1.set_title("Original Approach: Memory Explosion", fontsize=12, fontweight='bold')
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 10)
    ax1.axis('off')
    
    # Draw RAM Container
    from matplotlib.patches import Rectangle, FancyArrowPatch
    
    # RAM Limit (dashed line)
    ax1.hlines(y=6, xmin=0, xmax=10, colors='k', linestyles='dashed', linewidth=2)
    ax1.text(0.5, 6.2, "RAM Limit (e.g., 16 GB)", fontsize=10, color='k')
    
    # Full Matrix (Overflowing)
    # Represents N_pixels (height) x M_points (width)
    rect_full = Rectangle((2, 1), 6, 8, facecolor='#ff6666', edgecolor='k', linewidth=1.5)
    ax1.add_patch(rect_full)
    
    ax1.text(5, 5, "Full Distance Matrix\n(N Pixels x M Points)\n\n~36 GB RAM Required", 
             ha='center', va='center', fontsize=11, fontweight='bold', color='white')
    
    # --- Panel 2: The Solution (Chunking) ---
    ax2 = axes[1]
    ax2.set_title("Optimized Approach: Chunking", fontsize=12, fontweight='bold')
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 10)
    ax2.axis('off')
    
    # RAM Limit (dashed line) - same reference
    ax2.hlines(y=6, xmin=0, xmax=10, colors='k', linestyles='dashed', linewidth=2)
    ax2.text(0.5, 6.2, "RAM Limit", fontsize=10, color='k')
    
    # Outline of full matrix (dashed, ghost)
    rect_ghost = Rectangle((2, 1), 6, 8, facecolor='none', edgecolor='#b3b3b3', linestyle='--', linewidth=1)
    ax2.add_patch(rect_ghost)
    ax2.text(5, 8.5, "Total Data to Process", ha='center', va='center', fontsize=9, color='#b3b3b3')
    
    # One Chunk (Small slice)
    # Processed sequentially
    rect_chunk = Rectangle((2, 1), 6, 1.5, facecolor='#66b3ff', edgecolor='k', linewidth=1.5)
    ax2.add_patch(rect_chunk)
    
    ax2.text(5, 1.75, "Current Chunk\n(1000 Points)", ha='center', va='center', fontsize=10, fontweight='bold', color='white')
    
    # Arrow to Accumulator
    arrow = FancyArrowPatch((8, 1.75), (9, 4), connectionstyle="arc3,rad=-0.2", 
                            arrowstyle='->', mutation_scale=15, color='k')
    ax2.add_patch(arrow)
    
    # Accumulator (Result Vector)
    rect_acc = Rectangle((8.5, 3), 1, 4, facecolor='#66ff66', edgecolor='k', linewidth=1.5)
    ax2.add_patch(rect_acc)
    ax2.text(9, 7.5, "Max Slope\nResult", ha='center', va='center', fontsize=9)
    
    ax2.text(5, 4, "Memory Usage:\n< 500 MB", ha='center', va='center', fontsize=11, fontweight='bold', color='#009900')

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"Generated {output_path}")

if __name__ == "__main__":
    output_dir = r"c:\Users\yaredbe\Documents\losneomrade\docs\images"
    os.makedirs(output_dir, exist_ok=True)
    
    generate_original_algo_figure(os.path.join(output_dir, "original_algorithm_redundancy.png"))
    generate_bfs_algo_figure(os.path.join(output_dir, "bfs_algorithm_efficiency.png"))
    generate_memory_explosion_figure(os.path.join(output_dir, "memory_explosion_chunking.png"))
