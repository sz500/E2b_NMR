import pandas as pd
import matplotlib.pyplot as plt
import glob
import os
import sys

def plot_nmr_t2_data(filepath, figures_dir='figures'):
    """Plot a single NMR T2 relaxation CSV file."""
    # Read the CSV file - skip the units row and blank row
    df = pd.read_csv(filepath, skiprows=[1, 2])
    
    # Strip whitespace from column names
    df.columns = df.columns.str.strip()
    
    # Drop any potential NaN values to ensure a clean plot
    df = df.dropna()

    # Create the plot
    plt.figure(figsize=(10, 6))
    # Plot Channel A data
    plt.plot(df['Time'], df['Channel A'], color='navy', linewidth=1)
    
    # Formatting: extract filename for title
    filename = os.path.basename(filepath)
    plt.title(f'T2 Relaxation Data: {filename}', fontsize=14)
    plt.xlabel('Time (ms)', fontsize=12)
    plt.ylabel('Channel A (V)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    # Create figures directory if it doesn't exist
    os.makedirs(figures_dir, exist_ok=True)
    # Save with the filename to avoid overwriting
    output_name = os.path.splitext(filename)[0] + '_plot.png'
    output_path = os.path.join(figures_dir, output_name)
    # plt.savefig(output_path, dpi=300)
    plt.show()

def plot_all_t2_files(data_folder='data/T2', figures_dir='figures'):
    """
    Recursively find and plot all CSV files in the T2 data folder.
    
    Args:
        data_folder (str): Path to the T2 data folder (default: 'data/T2')
        figures_dir (str): Path to the folder where figures will be saved (default: 'figures')
    """
    # Find all CSV files recursively in the T2 folder
    csv_files = glob.glob(os.path.join(data_folder, '**/*.csv'), recursive=True)
    
    if not csv_files:
        print(f"No CSV files found in {data_folder}")
        return
    
    print(f"Found {len(csv_files)} CSV file(s) to plot:")
    for filepath in csv_files:
        print(f"  - Plotting: {filepath}")
        try:
            plot_nmr_t2_data(filepath, figures_dir=figures_dir)
        except Exception as e:
            print(f"  Error plotting {filepath}: {e}")

# Main execution
if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("NMR T2 Data Plotter")
        print("=" * 70)
        print("Usage:")
        print("  python plot.py all [directory] [subfolder]  - Plot all files in directory")
        print("  python plot.py <filepath> [subfolder]       - Plot a single file")
        print("=" * 70)
        print("\nParameters:")
        print("  directory:  Data folder path (default: data/T2)")
        print("  subfolder:  Subfolder inside 'figures' to save plots (default: figures)")
        print("\nExample:")
        print("  python plot.py all")
        print("  python plot.py all data/T2")
        print("  python plot.py all data/T2 raw_plots")
        print("  python plot.py data/T2/file.csv")
        print("  python plot.py data/T2/file.csv my_plots")
        sys.exit(0)
    
    arg = sys.argv[1].strip().strip("'\"")
    figures_subfolder = 'figures'  # Default subfolder
    
    if arg.lower() == 'all':
        # If a directory is specified, use it; otherwise use default
        data_folder = 'data/T2'
        if len(sys.argv) > 2:
            data_folder = sys.argv[2].strip().strip("'\"")
        # Check if subfolder is specified
        if len(sys.argv) > 3:
            subfolder_name = sys.argv[3].strip().strip("'\"")
            figures_subfolder = os.path.join('figures', subfolder_name)
        plot_all_t2_files(data_folder, figures_subfolder)
    else:
        # Treat as filepath
        if os.path.exists(arg):
            # Check if subfolder is specified
            if len(sys.argv) > 2:
                subfolder_name = sys.argv[2].strip().strip("'\"")
                figures_subfolder = os.path.join('figures', subfolder_name)
            plot_nmr_t2_data(arg, figures_subfolder)
        else:
            print(f"Error: File '{arg}' not found.")