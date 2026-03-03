import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import os
import sys
import glob
import re

def mz_model(t, m0, t1):
    """
    Model for T1 relaxation: Mz(t) = M0 * (1 - 2 * exp(-t / T1))
    
    Args:
        t: Delay time (ms)
        m0: Maximum magnetization
        t1: T1 relaxation time constant (spin-lattice relaxation)
    
    Returns:
        Mz: Magnetization at time t
    """
    return m0 * (1.0 - 2.0 * np.exp(-t / t1))

def get_mirrored_figures_path(filepath):
    """
    Get the output figures path that mirrors the data directory structure.
    
    Args:
        filepath: Path to the data file
    
    Returns:
        output_dir: Path to save figures with same subfolder structure as data
    """
    # Find E2b_NMR directory
    file_abs = os.path.abspath(filepath)
    
    # Look for E2b_NMR in the path
    parts = file_abs.split(os.sep)
    e2b_index = None
    for i, part in enumerate(parts):
        if part == 'E2b_NMR':
            e2b_index = i
            break
    
    if e2b_index is None:
        # Fallback: just use a 'figures' subdirectory in current location
        return os.path.join(os.getcwd(), 'figures')
    
    # Get the path relative to E2b_NMR directory
    e2b_root = os.sep.join(parts[:e2b_index + 1])
    
    # Get the directory structure after 'data'
    data_index = None
    for i in range(e2b_index, len(parts)):
        if parts[i] == 'data':
            data_index = i
            break
    
    if data_index is None:
        return os.path.join(os.getcwd(), 'figures')
    
    # Get relative path after 'data'
    relative_path = os.sep.join(parts[data_index + 1:-1])  # Exclude filename
    
    # Create figures path with same structure
    figures_path = os.path.join(e2b_root, 'figures', relative_path)
    
    return figures_path

def extract_concentration_and_temperature(filename):
    """
    Extract concentration (after 1st '_') and temperature (after 2nd '_') from filename.
    
    Args:
        filename: Filename (without path)
    
    Returns:
        concentration: Extracted concentration value or None
        temperature: Extracted temperature value or None
    """
    # Remove extension
    name = os.path.splitext(filename)[0]
    
    # Split by underscore
    parts = name.split('_')
    
    concentration = None
    temperature = None
    
    # Extract first underscore value (concentration)
    if len(parts) > 1:
        try:
            concentration = float(parts[1])
        except ValueError:
            pass
    
    # Extract second underscore value (temperature)
    if len(parts) > 2:
        try:
            temperature = float(parts[2])
        except ValueError:
            pass
    
    return concentration, temperature

def plot_t1_vs_parameter(results, parameter_name='temperature'):
    """
    Plot T1 values against temperature or concentration.
    
    Args:
        results: List of result dictionaries with t1, t1_err, and filename
        parameter_name: 'temperature' or 'concentration'
    """
    parameter_values = []
    t1_values = []
    t1_errors = []
    
    for result in results:
        filename = result['filename']
        concentration, temperature = extract_concentration_and_temperature(filename)
        
        if parameter_name == 'temperature' and temperature is not None:
            parameter_values.append(temperature)
            t1_values.append(result['t1'])
            t1_errors.append(result['t1_err'])
        elif parameter_name == 'concentration' and concentration is not None:
            parameter_values.append(concentration)
            t1_values.append(result['t1'])
            t1_errors.append(result['t1_err'])
    
    if not parameter_values:
        print(f"Warning: No valid {parameter_name} values found in filenames")
        return
    
    # Sort by parameter values
    sorted_data = sorted(zip(parameter_values, t1_values, t1_errors))
    parameter_values, t1_values, t1_errors = zip(*sorted_data)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.errorbar(parameter_values, t1_values, yerr=t1_errors, fmt='o-', capsize=5, markersize=8, linewidth=2)
    
    if parameter_name == 'temperature':
        ax.set_xlabel('Temperature (K)', fontsize=12)
        plot_name = 'T1_vs_temperature.png'
    else:
        ax.set_xlabel('Concentration (M)', fontsize=12)
        plot_name = 'T1_vs_concentration.png'
    
    ax.set_ylabel('T1 (ms)', fontsize=12)
    ax.set_title(f'T1 Relaxation Time vs {parameter_name.capitalize()}', fontsize=13)
    ax.grid(True, alpha=0.3)
    

    # Get the directory of the first result filepath to use as basis for figures path
    if results:
        first_filepath = results[0]['filepath']
        figures_dir = get_mirrored_figures_path(first_filepath)

    # Save plot
    os.makedirs(figures_dir, exist_ok=True)
    output_path = os.path.join(figures_dir, plot_name)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.show()
    plt.close()
    
    print(f"\n✓ Plot saved: {output_path}")

def fit_curve(filepath):
    """
    Perform exponential recovery curve fitting on NMR T1 data.
    
    Args:
        filepath: Path to the Excel file
    
    Returns:
        m0: Fitted M0 parameter
        t1: Fitted T1 parameter
        t1_err: Error in T1 parameter
    """
    try:
        df = pd.read_excel(filepath, header=None)
        
        # Read M0 and delta M0 from row 1 (index 1)
        m0_ref = df.iloc[1, 0]  # M0/mV value
        
        # Read T (measured T1 value) from row 1, find "T" or "T1" label and get the value after it
        t1_ref = None
        for col in range(df.shape[1]):
            cell_value = str(df.iloc[0, col]).strip()
            if cell_value.upper() in ['T', 'T1']:
                # Get the T value from the next column
                if col + 1 < df.shape[1]:
                    t1_ref = pd.to_numeric(df.iloc[0, col + 1], errors='coerce')
                break
        
        # Read data starting from row 3 (index 3), with headers at row 2 (index 2)
        data_start_row = 3
        headers = df.iloc[2, :5].values  # Get headers from row 2
        data = df.iloc[data_start_row:, :4].dropna(how='all')
        
        # Extract columns by position
        t_data = pd.to_numeric(data.iloc[:, 0], errors='coerce').values  # Column A: t/ms
        mz_data = pd.to_numeric(data.iloc[:, 1], errors='coerce').values  # Column B: V/mV
        sigma_data = np.full_like(mz_data, 400, dtype=float) # !!! error = 400mV
        
        # Remove NaN values
        mask = ~(np.isnan(t_data) | np.isnan(mz_data) | np.isnan(sigma_data))
        t_data = t_data[mask]
        mz_data = mz_data[mask]
        sigma_data = sigma_data[mask]
        
        if len(t_data) == 0:
            print(f"Error reading {filepath}: No valid numeric data found")
            return None, None, None
    except FileNotFoundError:
        print(f"Error: Could not find '{filepath}'.")
        return None, None, None
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return None, None, None
    
    # Provide initial guesses for the optimization algorithm
    m0_guess = m0_ref  # Use measured M0 as initial guess
    t1_guess = t1_ref if (t1_ref is not None and not np.isnan(t1_ref)) else 30.0  # Use measured T1 if available, else default to 30.0
    initial_guess = [m0_guess, t1_guess]

    try:
        # Perform the Non-Linear Fit
        popt, pcov = curve_fit(mz_model, t_data, mz_data, p0=initial_guess, 
                               sigma=sigma_data, absolute_sigma=True)
        
        m0_fit, t1_fit = popt
        m0_err, t1_err = np.sqrt(np.diag(pcov))
    except Exception as e:
        print(f"Curve fitting failed for {filepath}: {e}")
        return None, None, None
    
    # Calculate Residuals
    mz_fit = mz_model(t_data, m0_fit, t1_fit)
    residuals = mz_data - mz_fit
    
    # Create and save the plot with mirrored directory structure
    figures_dir = get_mirrored_figures_path(filepath)
    os.makedirs(figures_dir, exist_ok=True)
    filename = os.path.basename(filepath)
    output_name = os.path.splitext(filename)[0] + '_fitted.png'
    output_path = os.path.join(figures_dir, output_name)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
    
    # Main Fit Plot
    t_smooth = np.linspace(min(t_data), max(t_data), 200)
    ax1.errorbar(t_data, mz_data, yerr=sigma_data, fmt='ko', label='Experimental Data', capsize=4)
    ax1.plot(t_smooth, mz_model(t_smooth, m0_fit, t1_fit), 'r-', label=f'Fit: $M_0$ = {m0_fit:.3f}, $T_1$ = {t1_fit:.3f} ± {t1_err:.3f} ms')
    ax1.set_ylabel('Magnetization ($M_z$)')
    ax1.set_title(f'T1 Inversion Recovery Fit: {filename}')
    ax1.axhline(0, color='gray', linestyle='--', linewidth=1)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Residuals Plot
    ax2.errorbar(t_data, residuals, yerr=sigma_data, fmt='ko', capsize=4)
    ax2.axhline(0, color='r', linestyle='--')
    ax2.set_xlabel('Delay Time (ms)')
    ax2.set_ylabel('Residuals')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.show()
    plt.close()
    
    print(f"✓ {filename}")
    print(f"  M0 = {m0_fit:.6f} ± {m0_err:.6f}")
    print(f"  T1 = {t1_fit:.6f} ± {t1_err:.6f} ms")
    
    return m0_fit, t1_fit, t1_err

def fit_multiple_files(data_folder='.', plot_vs_temperature=False, plot_vs_concentration=False):
    """
    Fit curves for all Excel files in a folder.
    
    Args:
        data_folder: Folder containing Excel files
        plot_vs_temperature: Whether to plot T1 vs temperature
        plot_vs_concentration: Whether to plot T1 vs concentration
    """
    # Search for Excel files recursively
    excel_files = glob.glob(os.path.join(data_folder, '**/*.xlsx'), recursive=True)
    excel_files += glob.glob(os.path.join(data_folder, '**/*.xls'), recursive=True)
    
    if not excel_files:
        print(f"No Excel files found in {data_folder}")
        return
    
    results = []
    for filepath in excel_files:
        print(f"\nProcessing: {filepath}")
        m0, t1, t1_err = fit_curve(filepath)
        
        if m0 is not None and t1 is not None:
            filename = os.path.basename(filepath)
            results.append({
                'filename': filename,
                'filepath': filepath,
                'm0': m0,
                't1': t1,
                't1_err': t1_err
            })
    
    # Summary table
    if results:
        print(f"\n{'='*100}")
        print("SUMMARY OF ALL FITS")
        print(f"{'='*100}")
        print(f"{'Filename':<40} {'M0':<20} {'T1 (ms)':<30}")
        print(f"{'-'*100}")
        for result in results:
            t1_str = f"{result['t1']:.6f} ± {result['t1_err']:.6f}"
            print(f"{result['filename']:<40} {result['m0']:<20.6f} {t1_str:<30}")
        print(f"{'='*100}\n")
        
        # Extract all concentration and temperature values
        concentrations = []
        temperatures = []
        for result in results:
            concentration, temperature = extract_concentration_and_temperature(result['filename'])
            concentrations.append(concentration)
            temperatures.append(temperature)
        
        # Determine which parameters actually vary
        unique_concentrations = set([c for c in concentrations if c is not None])
        unique_temperatures = set([t for t in temperatures if t is not None])
        
        has_varying_conc = len(unique_concentrations) > 1
        has_varying_temp = len(unique_temperatures) > 1
        
        # Auto-decide which plot to make
        if plot_vs_temperature or plot_vs_concentration:
            # Explicit flags provided
            if plot_vs_temperature:
                plot_t1_vs_parameter(results, parameter_name='temperature')
            if plot_vs_concentration:
                plot_t1_vs_parameter(results, parameter_name='concentration')
        else:
            # Auto-decide based on which parameter varies
            if has_varying_temp and not has_varying_conc:
                print("Detected varying temperature, constant concentration → Plotting T1 vs Temperature")
                plot_t1_vs_parameter(results, parameter_name='temperature')
            elif has_varying_conc and not has_varying_temp:
                print("Detected varying concentration, constant temperature → Plotting T1 vs Concentration")
                plot_t1_vs_parameter(results, parameter_name='concentration')
            elif has_varying_conc and has_varying_temp:
                print("Both concentration and temperature vary. Use -temp, -conc, or -both flags to specify plots.")

# Main execution
if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("T1 Curve Fitting Tool for NMR Data")
        print("=" * 70)
        print("Usage:")
        print("  python T1_exp_fit_CuSO4.py <filepath>                      - Fit single file")
        print("  python T1_exp_fit_CuSO4.py <folder> [flags]                - Fit all files in folder")
        print("=" * 70)
        print("\nParameters:")
        print("  filepath:   Path to Excel file (.xlsx or .xls)")
        print("  folder:     Path to folder containing Excel files")
        print("\nOptional Flags (for folder processing):")
        print("  -temp       Plot T1 vs temperature (extracted from 2nd '_' in filename)")
        print("  -conc       Plot T1 vs concentration (extracted from 1st '_' in filename)")
        print("  -both       Plot both T1 vs temperature and T1 vs concentration")
        print("\nAutomatic behavior (without flags):")
        print("  - If temperature is constant and concentration varies → plots T1 vs concentration")
        print("  - If concentration is constant and temperature varies → plots T1 vs temperature")
        print("  - If both vary → requires explicit flag (-temp, -conc, or -both)")
        print("\nFigures are saved under: E2b_NMR/figures/ with the same subfolder structure as data")
        print("\nExample:")
        print("  python T1_exp_fit_CuSO4.py data/t1_data.xlsx")
        print("  python T1_exp_fit_CuSO4.py data/                    (auto-plot based on variation)")
        print("  python T1_exp_fit_CuSO4.py data/ -both              (explicit: plot both)")
        sys.exit(0)
    
    arg = sys.argv[1].strip().strip("'\"")
    plot_vs_temperature = False
    plot_vs_concentration = False
    
    # Check for flags
    for i in range(2, len(sys.argv)):
        arg_lower = sys.argv[i].lower()
        if arg_lower == '-temp':
            plot_vs_temperature = True
        elif arg_lower == '-conc':
            plot_vs_concentration = True
        elif arg_lower == '-both':
            plot_vs_temperature = True
            plot_vs_concentration = True
    
    if os.path.isfile(arg):
        # Single file processing
        fit_curve(arg)
    elif os.path.isdir(arg):
        # Folder processing
        fit_multiple_files(arg, 
                          plot_vs_temperature=plot_vs_temperature, 
                          plot_vs_concentration=plot_vs_concentration)
    else:
        print(f"Error: Path '{arg}' not found.")