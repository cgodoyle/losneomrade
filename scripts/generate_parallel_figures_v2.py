import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os

def setup_plot(ax, title):
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_aspect('equal')
    ax.set_title(title, fontsize=10, fontweight='bold')
    ax.set_xticks([])
    ax.set_yticks([])
    # Subtle grid
    for x in range(0, 101, 10):
        ax.axvline(x, color='#f0f0f0', linestyle='-', linewidth=0.5)
    for y in range(0, 101, 10):
        ax.axhline(y, color='#f0f0f0', linestyle='-', linewidth=0.5)

def draw_stream(ax, x, y, length, vertical=True, label=None):
    if vertical:
        ax.plot([x, x], [y - length/2, y + length/2], color='blue', linewidth=2, label=label)
    else:
        ax.plot([x - length/2, x + length/2], [y, y], color='blue', linewidth=2, label=label)

def generate_cutoff_figure(output_path):
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    
    # --- Panel 1: Simple Parallel (Fixed Buffer) ---
    ax1 = axes[0]
    setup_plot(ax1, "Problem: Fixed Small Buffer (Simple Parallel)")
    
    # Stream
    draw_stream(ax1, 50, 50, 20, vertical=True, label='Stream (Release)')
    
    # Tight Crop (e.g. 50px buffer -> small visual box)
    rect_crop = patches.Rectangle((35, 30), 30, 40, linewidth=2, edgecolor='k', facecolor='none', linestyle='--', label='Fixed Crop')
    ax1.add_patch(rect_crop)
    
    # Landslide hitting edge
    # Draw a shape that fills the crop but stops at the edge
    # A rounded rect or ellipse clipped
    landslide = patches.Ellipse((50, 50), 40, 50, facecolor='#ff6666', edgecolor='none', alpha=0.7, label='Landslide')
    landslide.set_clip_path(rect_crop)
    ax1.add_patch(landslide)
    
    # Re-draw crop on top
    ax1.add_patch(patches.Rectangle((35, 30), 30, 40, linewidth=2, edgecolor='k', facecolor='none', linestyle='--'))
    
    ax1.text(50, 25, "Artificial Cutoff", color='red', ha='center', fontweight='bold')
    ax1.text(50, 75, "Runout Truncated", color='red', ha='center', fontsize=9)
    
    ax1.legend(loc='upper right', fontsize=8)

    # --- Panel 2: Adaptive/Grouped (Dynamic Buffer) ---
    ax2 = axes[1]
    setup_plot(ax2, "Solution: Dynamic Buffer (Adaptive/Grouped)")
    
    # Stream
    draw_stream(ax2, 50, 50, 20, vertical=True)
    
    # Large Crop (Calculated from max_length)
    rect_large = patches.Rectangle((10, 10), 80, 80, linewidth=2, edgecolor='green', facecolor='none', linestyle='-', label='Dynamic Crop')
    ax2.add_patch(rect_large)
    
    # Full Landslide
    landslide_full = patches.Ellipse((50, 50), 40, 50, facecolor='#ff6666', edgecolor='k', alpha=0.7)
    ax2.add_patch(landslide_full)
    
    ax2.text(50, 15, "Full Propagation", color='green', ha='center', fontweight='bold')
    
    ax2.legend(loc='upper right', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"Generated {output_path}")

def generate_grouping_figure(output_path):
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    
    # Setup: Two nearby streams
    s1_pos = (40, 50)
    s2_pos = (60, 50)
    
    # --- Panel 1: Ungrouped (Separate Processing) ---
    ax1 = axes[0]
    setup_plot(ax1, "Ungrouped: Missed Interaction")
    
    draw_stream(ax1, s1_pos[0], s1_pos[1], 15, vertical=True)
    draw_stream(ax1, s2_pos[0], s2_pos[1], 15, vertical=True)
    
    # Separate Crops (even if large enough for individual, they are separate)
    # Crop 1
    rect1 = patches.Rectangle((20, 30), 35, 40, linewidth=1.5, edgecolor='gray', facecolor='none', linestyle='--')
    ax1.add_patch(rect1)
    # Crop 2
    rect2 = patches.Rectangle((45, 30), 35, 40, linewidth=1.5, edgecolor='gray', facecolor='none', linestyle='--')
    ax1.add_patch(rect2)
    
    # Resulting Landslides (Overlapping but calculated separately)
    # Just two ellipses
    l1 = patches.Ellipse(s1_pos, 25, 30, facecolor='#ff9999', edgecolor='k', alpha=0.6)
    l2 = patches.Ellipse(s2_pos, 25, 30, facecolor='#ff9999', edgecolor='k', alpha=0.6)
    ax1.add_patch(l1)
    ax1.add_patch(l2)
    
    ax1.text(50, 80, "Processed Separately", ha='center')
    ax1.text(50, 20, "Potential Interaction Lost", ha='center', color='orange')

    # --- Panel 2: Grouped (Combined Processing) ---
    ax2 = axes[1]
    setup_plot(ax2, "Grouped: Correct Merging")
    
    draw_stream(ax2, s1_pos[0], s1_pos[1], 15, vertical=True)
    draw_stream(ax2, s2_pos[0], s2_pos[1], 15, vertical=True)
    
    # Single Group Crop
    rect_group = patches.Rectangle((15, 25), 70, 50, linewidth=2, edgecolor='green', facecolor='none', linestyle='-')
    ax2.add_patch(rect_group)
    
    # Merged Landslide
    # A single shape covering both
    # We can draw a large rounded rect or two ellipses merged
    # Merged appearance:
    l_merged = patches.FancyBboxPatch((27, 35), 46, 30, boxstyle="Round,pad=0.5", facecolor='#ff6666', edgecolor='k', alpha=0.8)
    ax2.add_patch(l_merged)
    
    ax2.text(50, 80, "Processed Together", ha='center')
    ax2.text(50, 20, "Physically Correct Merge", ha='center', color='green', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"Generated {output_path}")

def generate_adaptive_strategies_figure(output_path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Scenario: 3 streams. A and B are close (e.g. 200m). C is medium far (e.g. 1000m).
    # Scale: 0-100 units. Let's say 1 unit = 10m.
    # A: 30, B: 35 (dist 50m). C: 60 (dist 250m from B).
    # Wait, let's make it clearer.
    # A: 20. B: 30. C: 70.
    # Dist A-B = 10. Dist B-C = 40.
    
    sA = 20; sB = 30; sC = 70
    y = 50
    
    def draw_scenario(ax, title, grouping_dist, strategy_name):
        setup_plot(ax, title)
        draw_stream(ax, sA, y, 10, vertical=True)
        draw_stream(ax, sB, y, 10, vertical=True)
        draw_stream(ax, sC, y, 10, vertical=True)
        
        ax.text(sA, 40, "A", ha='center', fontsize=8)
        ax.text(sB, 40, "B", ha='center', fontsize=8)
        ax.text(sC, 40, "C", ha='center', fontsize=8)
        
        # Visualize Grouping
        # A-B dist is 10. B-C dist is 40.
        
        groups = []
        # Simple logic for viz
        if 10 <= grouping_dist:
            # A and B group
            if 40 <= grouping_dist:
                # B and C group -> A, B, C all group
                groups.append([sA, sB, sC])
            else:
                groups.append([sA, sB])
                groups.append([sC])
        else:
            groups.append([sA])
            groups.append([sB])
            groups.append([sC])
            
        colors = ['#ccffcc', '#ccccff', '#ffcccc']
        
        for i, group in enumerate(groups):
            min_x = min(group) - 10
            max_x = max(group) + 10
            rect = patches.Rectangle((min_x, 20), max_x - min_x, 60, 
                                     linewidth=2, edgecolor=colors[i], facecolor=colors[i], alpha=0.3)
            ax.add_patch(rect)
            # Label
            ax.text((min_x+max_x)/2, 85, f"Group {i+1}", ha='center', fontsize=9, fontweight='bold')

        ax.text(50, 10, f"{strategy_name}\n(Dist: {grouping_dist})", ha='center', fontweight='bold')

    # Panel 1: Speed (Small Dist)
    # A and B group (dist 10). C separate (dist 40).
    # Let's say Speed Dist = 15.
    draw_scenario(axes[0], "Speed Mode", 15, "Groups nearby only")
    
    # Panel 2: Balanced (Medium Dist)
    # A, B, C group.
    # Let's say Balanced Dist = 50.
    draw_scenario(axes[1], "Balanced Mode", 50, "Groups neighborhood")
    
    # Panel 3: Serial / Accuracy (Infinite Dist)
    # All group.
    # Actually Balanced usually groups everything in a sub-catchment.
    # Let's make C really far for Speed to miss it.
    # In the viz, C is at 70. B at 30. Dist 40.
    # If Speed=15, Balanced=50.
    # Speed: {A,B}, {C}.
    # Balanced: {A,B,C}.
    
    # Let's add a D really far away at 95.
    # Dist C-D = 25.
    # If Balanced=50, C-D group too?
    # Let's stick to 3 streams for clarity.
    
    # Serial comparison
    # Serial is just "One Thread".
    # But visually, Balanced often results in large groups.
    # Let's label Panel 3 as "Accuracy/Serial" -> All in one.
    draw_scenario(axes[2], "Accuracy / Serial", 100, "Groups everything")

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"Generated {output_path}")

if __name__ == "__main__":
    output_dir = r"c:\Users\yaredbe\Documents\losneomrade\docs\images"
    os.makedirs(output_dir, exist_ok=True)
    
    generate_cutoff_figure(os.path.join(output_dir, "parallel_cutoff_problem.png"))
    generate_grouping_figure(os.path.join(output_dir, "parallel_grouping_concept.png"))
    generate_adaptive_strategies_figure(os.path.join(output_dir, "parallel_adaptive_strategies.png"))
