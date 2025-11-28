"""
Plot Performance Metrics
Reads summary metrics CSVs and generates performance plots for the presentation.
"""
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

def plot_metrics():
    # File paths
    file_orig_vs_opt = "output/comparison_landslide_retrogression_original_v_optimized/summary_metrics.csv"
    file_ser_vs_par = "output/comparison_serial_optimized_vs_parallel_adaptive_speed/summary_metrics.csv"
    
    output_dir = "docs/images"
    os.makedirs(output_dir, exist_ok=True)
    
    # Set style
    plt.style.use('ggplot')
    
    # 1. Original vs Optimized
    if os.path.exists(file_orig_vs_opt):
        df1 = pd.read_csv(file_orig_vs_opt)
        
        # Plot Time Comparison
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(df1['subset'], df1['time_original'], marker='o', label='Original landslide retrogression', linewidth=2)
        ax.plot(df1['subset'], df1['time_optimized'], marker='o', label='Optimized landslide retrogression', linewidth=2)
        
        ax.set_title('Performance: Original vs Optimized', fontsize=14)
        ax.set_xlabel('Number of Streams (Subset Size)', fontsize=12)
        ax.set_ylabel('Execution Time (seconds)', fontsize=12)
        ax.legend(fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "perf_original_vs_optimized_time.png"), dpi=300)
        print(f"Saved perf_original_vs_optimized_time.png")
        
        # Plot Speedup
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(df1['subset'].astype(str), df1['speedup'], color='teal', alpha=0.7)
        
        ax.set_title('Speedup: Optimized vs Original', fontsize=14)
        ax.set_xlabel('Number of Streams (Subset Size)', fontsize=12)
        ax.set_ylabel('Speedup Factor (x)', fontsize=12)
        ax.bar_label(bars, fmt='%.1fx')
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "perf_original_vs_optimized_speedup.png"), dpi=300)
        print(f"Saved perf_original_vs_optimized_speedup.png")
        
    else:
        print(f"Warning: File not found {file_orig_vs_opt}")

    # 2. Serial Optimized vs Parallel Adaptive (Speed)
    if os.path.exists(file_ser_vs_par):
        df2 = pd.read_csv(file_ser_vs_par)
        
        # Plot Time Comparison
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(df2['subset'], df2['time_serial_optimized'], marker='o', label='Serial with optimized landslide retrogression', linewidth=2, color='tab:orange')
        ax.plot(df2['subset'], df2['time_parallel_adaptive'], marker='o', label='Parallel Adaptive (Speed)', linewidth=2, color='tab:purple')
        
        ax.set_title('Performance: Serial vs Parallel Adaptive (Speed Mode)', fontsize=14)
        ax.set_xlabel('Number of Streams (Subset Size)', fontsize=12)
        ax.set_ylabel('Execution Time (seconds)', fontsize=12)
        ax.legend(fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "perf_serial_vs_parallel_time.png"), dpi=300)
        print(f"Saved perf_serial_vs_parallel_time.png")
        
        # Plot Speedup
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(df2['subset'].astype(str), df2['speedup'], color='purple', alpha=0.7)
        
        ax.set_title('Speedup: Parallel Adaptive (Speed) vs Serial', fontsize=14)
        ax.set_xlabel('Number of Streams (Subset Size)', fontsize=12)
        ax.set_ylabel('Speedup Factor (x)', fontsize=12)
        ax.bar_label(bars, fmt='%.1fx')
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "perf_serial_vs_parallel_speedup.png"), dpi=300)
        print(f"Saved perf_serial_vs_parallel_speedup.png")

    else:
        print(f"Warning: File not found {file_ser_vs_par}")

    # 3. Deduced: Original vs Parallel Adaptive (Speed)
    if os.path.exists(file_orig_vs_opt) and os.path.exists(file_ser_vs_par):
        df1 = pd.read_csv(file_orig_vs_opt)
        df2 = pd.read_csv(file_ser_vs_par)
        
        # Merge on subset
        df_merged = pd.merge(df1[['subset', 'time_original']], 
                             df2[['subset', 'time_parallel_adaptive']], 
                             on='subset')
        
        df_merged['total_speedup'] = df_merged['time_original'] / df_merged['time_parallel_adaptive']
        
        # Plot Time Comparison
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(df_merged['subset'], df_merged['time_original'], marker='o', label='Original landslide retrogression', linewidth=2, color='tab:blue')
        ax.plot(df_merged['subset'], df_merged['time_parallel_adaptive'], marker='o', label='Parallel Adaptive (Speed)', linewidth=2, color='tab:purple')
        
        ax.set_title('Total Performance Improvement: Original vs Parallel', fontsize=14)
        ax.set_xlabel('Number of Streams (Subset Size)', fontsize=12)
        ax.set_ylabel('Execution Time (seconds)', fontsize=12)
        ax.legend(fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "perf_total_original_vs_parallel_time.png"), dpi=300)
        print(f"Saved perf_total_original_vs_parallel_time.png")
        
        # Plot Speedup
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(df_merged['subset'].astype(str), df_merged['total_speedup'], color='green', alpha=0.7)
        
        ax.set_title('Total Speedup: Parallel Adaptive vs Original', fontsize=14)
        ax.set_xlabel('Number of Streams (Subset Size)', fontsize=12)
        ax.set_ylabel('Speedup Factor (x)', fontsize=12)
        ax.bar_label(bars, fmt='%.1fx')
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "perf_total_original_vs_parallel_speedup.png"), dpi=300)
        print(f"Saved perf_total_original_vs_parallel_speedup.png")

if __name__ == "__main__":
    plot_metrics()
