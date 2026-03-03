import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import sys
import re

def extract_tau_from_filename(filename):
    """
    Extract tau value from filename. Supports:
    - Simple number filenames: '10.csv', '5.5.csv' (treated as tau in ms)
    - Named patterns: 'T2_tau=5.5ms_4.csv' or 'T2_tau=10ms_07.csv'
    
    Args:
        filename: Filename string
    
    Returns:
        tau_value: Tau value in ms (float), or None if not found
    """
    # Remove extension
    name_without_ext = os.path.splitext(filename)[0]
    
    # First try to match patterns like "tau=5.5ms" or "tau=10ms"
    match = re.search(r'tau=([0-9.]+)\s*ms', filename, re.IGNORECASE)
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            return None
    
    # If filename is just a number (e.g., "10.csv" or "5.5.csv")
    try:
        return float(name_without_ext)
    except ValueError:
        return None

def plot_t2_vs_tau(results, figures_dir='figures'):
    """
    Create a plot of T2 vs tau (delta time).
    
    Args:
        results: List of result dictionaries with tau, T2, and T2_err
        figures_dir: Directory to save the figure
    """
    # Sort by tau value
    results_sorted = sorted(results, key=lambda x: x['tau'])
    
    taus = [r['tau'] for r in results_sorted]
    t2_values = [r['T2'] for r in results_sorted]
    t2_errs = [r['T2_err'] for r in results_sorted]
    
    # Create figure
    plt.figure(figsize=(10, 6))
    plt.errorbar(taus, t2_values, yerr=t2_errs, fmt='o-', markersize=8, 
                 capsize=5, capthick=2, linewidth=2, color='#1f77b4', 
                 ecolor='#1f77b4', alpha=0.8, label='T2 measurements')
    
    plt.xlabel('τ (ms)', fontsize=12)
    plt.ylabel('T2 (ms)', fontsize=12)
    plt.title('T2 Relaxation vs Pulse Spacing (τ)', fontsize=13, fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(fontsize=11)
    plt.tight_layout()
    
    # Save the figure
    os.makedirs(figures_dir, exist_ok=True)
    output_path = os.path.join(figures_dir, 'T2_vs_tau.png')
    plt.savefig(output_path, dpi=300)
    plt.close()
    
    print(f"T2 vs tau plot saved to: {output_path}")

def load_curve_fit_results(figures_dir='figures/fitted'):
    """
    Load T2 results from curve_fit summary or CSV files.
    Looks for a summary CSV file created by curve_fit.py
    
    Args:
        figures_dir: Directory containing fitted curve figures
    
    Returns:
        List of result dictionaries with tau, T2, and T2_err
    """
    results = []
    
    # Try to read from summary file first
    summary_file = os.path.join(os.path.dirname(figures_dir), 'T2_summary.csv')
    if os.path.exists(summary_file):
        try:
            df = pd.read_csv(summary_file)
            for _, row in df.iterrows():
                filename = row['filename']
                tau = extract_tau_from_filename(filename)
                if tau is not None:
                    results.append({
                        'filename': filename,
                        'M0': row['M0'],
                        'T2': row['T2'],
                        'T2_err': row['T2_err'],
                        'tau': tau
                    })
            return results
        except Exception as e:
            print(f"Warning: Could not read summary file {summary_file}: {e}")
    
    # If no summary file, print instructions
    print("No summary file found.")
    print("Please run curve_fit.py first to generate results:")
    print("  python curve_fit.py all data/T2")
    return []

# Main execution
if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("T2 vs Tau Plotting Tool")
        print("=" * 70)
        print("Usage:")
        print("  python plot_t2_vs_tau.py <summary_file>  - Plot from summary CSV")
        print("  python plot_t2_vs_tau.py all [directory] - Analyze directory and plot")
        print("=" * 70)
        print("\nExample:")
        print("  python plot_t2_vs_tau.py figures/T2_summary.csv")
        print("  python plot_t2_vs_tau.py all data/T2")
        sys.exit(0)
    
    arg = sys.argv[1].strip().strip("'\"")
    
    if arg.lower() == 'all':
        # Analyze CSV files directly
        data_folder = 'data/T2'
        if len(sys.argv) > 2:
            data_folder = sys.argv[2].strip().strip("'\"")
        
        # Find all CSV files
        csv_files = glob.glob(os.path.join(data_folder, '**/*.csv'), recursive=True)
        
        if not csv_files:
            print(f"No CSV files found in {data_folder}")
            sys.exit(1)
        
        print(f"Found {len(csv_files)} CSV file(s)")
        print("Note: This mode extracts tau values and creates the plot,")
        print("      but does not perform curve fitting.")
        print("\nFor best results, run curve_fit.py first and use the summary CSV:")
        print(f"  python curve_fit.py all {data_folder}")
        print("  python plot_t2_vs_tau.py figures/T2_summary.csv")
        
        # Extract tau values from filenames
        results = []
        for filepath in csv_files:
            filename = os.path.basename(filepath)
            tau = extract_tau_from_filename(filename)
            if tau is not None:
                results.append({
                    'filename': filename,
                    'tau': tau,
                    'T2': None,
                    'T2_err': None
                })
        
        if results:
            print(f"\nExtracted tau values:")
            for r in sorted(results, key=lambda x: x['tau']):
                print(f"  {r['filename']:<40} τ = {r['tau']} ms")
        
    elif os.path.isfile(arg):
        # Load from summary CSV file
        print(f"Loading results from: {arg}")
        try:
            df = pd.read_csv(arg)
            results = []
            for _, row in df.iterrows():
                filename = row['filename'] if 'filename' in df.columns else ""
                T2 = row['T2'] if 'T2' in df.columns else None
                T2_err = row['T2_err'] if 'T2_err' in df.columns else 0
                
                # Extract tau from filename
                tau = extract_tau_from_filename(filename) if filename else None
                
                if tau is not None and T2 is not None:
                    results.append({
                        'filename': filename,
                        'T2': T2,
                        'T2_err': T2_err,
                        'tau': tau
                    })
            
            if results:
                print(f"\nLoaded {len(results)} results with tau values")
                plot_t2_vs_tau(results)
                
                # Print summary
                print("\nData points:")
                for r in sorted(results, key=lambda x: x['tau']):
                    print(f"  τ = {r['tau']:8.2f} ms | T2 = {r['T2']:8.4f} ± {r['T2_err']:8.4f} ms | {r['filename']}")
            else:
                print("No results with valid tau values found in the file.")
        except Exception as e:
            print(f"Error reading file: {e}")
    else:
        print(f"Error: File '{arg}' not found.")
