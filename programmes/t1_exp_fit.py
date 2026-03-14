import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import os
import sys
import glob

def mz_model(t, m0, t1, c):
    """
    Model for T1 relaxation: Mz(t) = M0 * (1 - 2 * exp(-t / T1)) + C
    
    Args:
        t: Delay time (ms)
        m0: Maximum magnetization
        t1: T1 relaxation time constant (spin-lattice relaxation)
        c: Constant offset (background signal)
    
    Returns:
        Mz: Magnetization at time t
    """
    return m0 * (1.0 - 2.0 * np.exp(-t / t1)) + c

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

def _extract_t1_reference(df):
    """
    Extract measured T1 value from metadata row if available.
    Looks for cell labels 'T' or 'T1' and reads the next column value.
    """
    t1_ref = None
    if df.shape[0] == 0:
        return t1_ref

    for col in range(df.shape[1]):
        cell_value = str(df.iloc[0, col]).strip()
        if cell_value.upper() in ['T', 'T1']:
            if col + 1 < df.shape[1]:
                t1_ref = pd.to_numeric(df.iloc[0, col + 1], errors='coerce')
            break

    return t1_ref

def _parse_format_tau_header(df):
    """
    Parse format with a header row containing 'tau' and data columns for V and delta V.
    Returns (m0_ref, t1_ref, t_data, mz_data, sigma_data) or None if parsing fails.
    """
    if df.shape[0] == 0:
        return None

    # Read M0 by searching 'M0' label in first row metadata
    m0_ref = None
    for col in range(df.shape[1]):
        cell_value = str(df.iloc[0, col]).strip()
        if 'M0' in cell_value:
            if col + 1 < df.shape[1]:
                m0_ref = pd.to_numeric(df.iloc[0, col + 1], errors='coerce')
            break

    t1_ref = _extract_t1_reference(df)

    # Find data start row by looking for "tau" header in first column
    data_start_row = None
    for row in range(df.shape[0]):
        if 'tau' in str(df.iloc[row, 0]).lower():
            data_start_row = row + 1
            break

    if data_start_row is None:
        return None

    data = df.iloc[data_start_row:, :7].dropna(how='all')
    if data.empty:
        return None

    header_row = df.iloc[data_start_row - 1, :7].values
    tau_col = None
    v_col = None

    for i, header in enumerate(header_row):
        header_str = str(header).lower().strip()
        if 'tau' in header_str:
            tau_col = i
        elif 'v' in header_str and 'delta' not in header_str:
            v_col = i

    if tau_col is None:
        tau_col = 0
    if v_col is None:
        v_col = 2

    t_data = pd.to_numeric(data.iloc[:, tau_col], errors='coerce').values
    mz_data = pd.to_numeric(data.iloc[:, v_col], errors='coerce').values
    sigma_data = np.full_like(mz_data, 100, dtype=float)

    mask = ~(np.isnan(t_data) | np.isnan(mz_data) | np.isnan(sigma_data))
    t_data = t_data[mask]
    mz_data = mz_data[mask]
    sigma_data = sigma_data[mask]

    if len(t_data) == 0:
        return None

    return m0_ref, t1_ref, t_data, mz_data, sigma_data

def _parse_format_fixed_columns(df):
    """
    Parse fixed-column format used in t1_exp_fit_C.py:
    - M0 in row 1, col 0
    - data starts at row 3 with tau in col 0 and V in col 1
    Returns (m0_ref, t1_ref, t_data, mz_data, sigma_data) or None if parsing fails.
    """
    if df.shape[0] < 4 or df.shape[1] < 2:
        return None

    m0_ref = pd.to_numeric(df.iloc[1, 0], errors='coerce')
    t1_ref = _extract_t1_reference(df)

    data_start_row = 3
    data = df.iloc[data_start_row:, :4].dropna(how='all')
    if data.empty:
        return None

    t_data = pd.to_numeric(data.iloc[:, 0], errors='coerce').values
    mz_data = pd.to_numeric(data.iloc[:, 1], errors='coerce').values
    sigma_data = np.full_like(mz_data, 100, dtype=float)

    mask = ~(np.isnan(t_data) | np.isnan(mz_data) | np.isnan(sigma_data))
    t_data = t_data[mask]
    mz_data = mz_data[mask]
    sigma_data = sigma_data[mask]

    if len(t_data) == 0:
        return None

    return m0_ref, t1_ref, t_data, mz_data, sigma_data

def parse_t1_excel_data(filepath):
    """
    Auto-detect and parse supported T1 Excel formats.
    Tries the tau-header format first, then the fixed-column format.
    """
    df = pd.read_excel(filepath, header=None)

    parsed = _parse_format_tau_header(df)
    if parsed is not None:
        return parsed

    parsed = _parse_format_fixed_columns(df)
    if parsed is not None:
        return parsed

    raise ValueError('Unsupported Excel format or no valid numeric T1 data found.')

def plot_t1_vs_parameter(results, parameter_name='concentration'):
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
    parameter_values = np.array(parameter_values)
    t1_values = np.array(t1_values)
    t1_errors = np.array(t1_errors)
    
    # concentration theory
    # t1_errors = t1_errors/t1_values**2
    # t1_values = 1/t1_values

    # temperature theory
    # parameter_values += 273
    # y = np.log(t1_values / parameter_values)
    # x = 1/parameter_values
    # t1_errors = t1_errors / t1_values  # error propagation
    # t1_values = y
    # parameter_values = x
    
    with plt.rc_context({
        'font.family': 'serif',
        'font.size': 9,
        'axes.labelsize': 10,
        'axes.titlesize': 10,
        'axes.linewidth': 0.8,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'xtick.major.size': 4,
        'ytick.major.size': 4,
        'xtick.minor.size': 2.5,
        'ytick.minor.size': 2.5,
        'legend.fontsize': 8,
        'savefig.bbox': 'tight'
    }):
        fig, ax = plt.subplots(figsize=(4.8, 3.6))
        ax.errorbar(
            parameter_values,
            t1_values,
            yerr=t1_errors,
            fmt='o',
            color='black',
            ecolor='black',
            elinewidth=0.8,
            capsize=2,
            capthick=0.8,
            markersize=3,
            markeredgewidth=1,
            zorder=2
        )
        ax.plot(
            parameter_values,
            t1_values,
            linestyle='--',
            color='steelblue',
            linewidth=1.5,
            zorder=1
        )

        ######################### linear fit ############################
        slope, intercept = np.polyfit(parameter_values, t1_values, 1)
        y_fit = slope * np.array(parameter_values) + intercept
        # 2. Calculate R-squared
        residuals = t1_values - y_fit
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((t1_values - np.mean(t1_values))**2)
        r_squared = 1 - (ss_res / ss_tot)
        x_min = np.min(parameter_values)
        x_max = np.max(parameter_values)
        margin = (x_max - x_min) * 0.10
        x_fit_extended = np.linspace(x_min - margin, x_max + margin, 100)
        y_fit_extended = slope * x_fit_extended + intercept
        ax.plot(
            x_fit_extended,
            y_fit_extended,
            linestyle='-.',
            color='gray',
            linewidth=1.5,
            label=f'Linear Fit ($R^2 = {r_squared:.3f}$)',
            zorder=1
        )
        # r^2
        ax.text(0.05, 0.95, f'$R^2 = {r_squared:.4f}$',
            transform=ax.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
        ###################################################################

        if parameter_name == 'temperature':
            # ax.set_xlabel('Temperature (°C)')
            # plot_name = 'T1_vs_temperature.png'
            
            ax.set_xlabel('1 / Temperature (K$^{-1}$)')
            plot_name = 'linear_fit_T1_vs_temperature.png'
        else:
            ax.set_xlabel('Concentration (%)')
            plot_name = 'T1_vs_concentration.png'

        # ax.set_ylabel('T1 (ms)')
        # ax.set_ylabel('ln(T1) (ln(ms))')   # Gly conc theory
        # ax.set_ylabel('1/T1 (ms$^{-1}$)')   # CuSO4 conc theory
        ax.set_ylabel('ln($T_1/T$) (ln(ms K$^{-1}$))')  # temp theory, same for both

        # ax.set_title(f'T1 Relaxation Time vs {parameter_name.capitalize()}')
        ax.minorticks_on()
        ax.grid(True, which='major', linestyle='--', linewidth=0.5, alpha=0.25)
    
    # Get the directory of the first result filepath to use as basis for figures path
    if results:
        first_filepath = results[0]['filepath']
        figures_dir = get_mirrored_figures_path(first_filepath)
    
        # Save plot
        os.makedirs(figures_dir, exist_ok=True)
        output_path = os.path.join(figures_dir, plot_name)
        plt.tight_layout()
        plt.savefig(output_path, dpi=600)
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
        m0_ref, t1_ref, t_data, mz_data, sigma_data = parse_t1_excel_data(filepath)
    except FileNotFoundError:
        print(f"Error: Could not find '{filepath}'.")
        return None, None, None
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return None, None, None
    
    # Provide initial guesses for the optimization algorithm
    m0_guess = pd.to_numeric(m0_ref, errors='coerce')
    if pd.isna(m0_guess):
        m0_guess = np.nanmax(np.abs(mz_data))
    t1_guess = t1_ref if (t1_ref is not None and not np.isnan(t1_ref)) else 30.0  # Use measured T1 if available, else default to 30.0
    c_guess = (np.nanmax(mz_data) + np.nanmin(mz_data)) / 2.0
    initial_guess = [m0_guess, t1_guess, c_guess]
    print(m0_guess)
    print(t1_guess)

    try:
        # Perform the Non-Linear Fit
        popt, pcov = curve_fit(mz_model, t_data, mz_data, p0=initial_guess, 
                               sigma=sigma_data, absolute_sigma=True)
        
        m0_fit, t1_fit, c_fit = popt
        m0_err, t1_err, c_err = np.sqrt(np.diag(pcov))
    except Exception as e:
        print(f"Curve fitting failed for {filepath}: {e}")
        return None, None, None
    
    # Calculate Residuals
    mz_fit = mz_model(t_data, m0_fit, t1_fit, c_fit)
    residuals = mz_data - mz_fit

    # Generate uncertainty band for T1 uncertainty only
    t_smooth = np.linspace(min(t_data), max(t_data), 200)
    t1_upper = t1_fit + t1_err
    t1_lower = t1_fit - t1_err
    curve_upper = mz_model(t_smooth, m0_fit, t1_upper, c_fit)
    curve_lower = mz_model(t_smooth, m0_fit, t1_lower, c_fit)
    
    # Create and save the plot with mirrored directory structure
    figures_dir = get_mirrored_figures_path(filepath)
    os.makedirs(figures_dir, exist_ok=True)
    filename = os.path.basename(filepath)
    output_name = os.path.splitext(filename)[0] + '_fitted.png'
    output_path = os.path.join(figures_dir, output_name)
    
    with plt.rc_context({
        'font.family': 'serif',
        'font.size': 9,
        'axes.labelsize': 10,
        'axes.titlesize': 10,
        'axes.linewidth': 0.8,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'xtick.major.size': 4,
        'ytick.major.size': 4,
        'xtick.minor.size': 2.5,
        'ytick.minor.size': 2.5,
        'legend.fontsize': 8,
        'savefig.bbox': 'tight'
    }):
        fig, (ax1, ax2) = plt.subplots(
            2, 1, figsize=(4.8, 5.0),
            gridspec_kw={'height_ratios': [3, 1]}, sharex=True
        )

        # Main Fit Plot
        ax1.errorbar(
            t_data, mz_data, yerr=sigma_data,
            fmt='o', color='black', ecolor='black',
            elinewidth=0.8, capsize=2, capthick=0.8,
            markersize=3, markeredgewidth=1, zorder=2,
            label='Experimental Data'
        )
        ax1.fill_between(
            t_smooth,
            curve_lower,
            curve_upper,
            color='indianred',
            alpha=0.16,
            zorder=0,
            label=f'T1 uncertainty (±1σ: ±{t1_err:.2g} ms)'
        )
        ax1.plot(
            t_smooth, mz_model(t_smooth, m0_fit, t1_fit, c_fit),
            linestyle='--', color='firebrick', linewidth=1, zorder=1,
            label=f'Fit: $T_1$ = {t1_fit:.3g} ± {t1_err:.2g} ms'
        )
        ax1.set_ylabel('Signal Amplitude $\propto M_z$ (mV)')
        # ax1.set_title(f'T1 Inversion Recovery Fit: {filename}')
        ax1.axhline(0, color='gray', linewidth=0.9)
        ax1.minorticks_on()
        ax1.grid(True, which='major', linewidth=0.5, alpha=0.25)
        
        ax1.legend(frameon=False)

        # Residuals Plot
        ax2.errorbar(
            t_data, residuals, yerr=sigma_data,
            fmt='o', color='black', ecolor='black',
            elinewidth=0.8, capsize=2, capthick=0.8,
            markersize=3, markeredgewidth=1
        )
        ax2.axhline(0, color='firebrick', linestyle = '--', linewidth=1.0)
        ax2.set_xlabel(r'Delay Time $\tau$ (ms)')
        ax2.set_ylabel('Residuals (mV)')
        ax2.minorticks_on()
        ax2.grid(True, which='major', linestyle='--', linewidth=0.5, alpha=0.25)

        plt.tight_layout()
        plt.savefig(output_path, dpi=600)
        plt.show()
        plt.close()
    
    print(f"✓ {filename}")
    print(f"  M0 = {m0_fit:.6f} ± {m0_err:.6f}")
    print(f"  T1 = {t1_fit:.6f} ± {t1_err:.6f} ms")
    print(f"  C  = {c_fit:.6f} ± {c_err:.6f}")

    return m0_fit, t1_fit, t1_err, c_fit, c_err

def fit_multiple_files(data_folder='.'):
    """
    Fit curves for all Excel files in a folder.
    
    Args:
        data_folder: Folder containing Excel files
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
        m0, t1, t1_err, c, c_err = fit_curve(filepath)
        
        if m0 is not None and t1 is not None:
            filename = os.path.basename(filepath)
            results.append({
                'filename': filename,
                'filepath': filepath,
                'm0': m0,
                't1': t1,
                't1_err': t1_err,
                'c': c,
                'c_err': c_err
            })
    
    # Summary table
    if results:
        print(f"\n{'='*100}")
        print("SUMMARY OF ALL FITS")
        print(f"{'='*100}")
        print(f"{'Filename':<40} {'M0':<20} {'T1 (ms)':<30} {'C':<20}")
        print(f"{'-'*100}")
        for result in results:
            t1_str = f"{result['t1']:.6f} ± {result['t1_err']:.6f}"
            c_str = f"{result['c']:.6f} ± {result['c_err']:.6f}"
            print(f"{result['filename']:<40} {result['m0']:<20.6f} {t1_str:<30} {c_str:<20}")
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
        
        # Auto-decide which plot to make based on which parameter varies
        if has_varying_conc and not has_varying_temp:
            print("Detected varying concentration, constant temperature → Plotting T1 vs Concentration")
            plot_t1_vs_parameter(results, parameter_name='concentration')
        elif has_varying_temp and not has_varying_conc:
            print("Detected varying temperature, constant concentration → Plotting T1 vs Temperature")
            plot_t1_vs_parameter(results, parameter_name='temperature')
        elif has_varying_conc and has_varying_temp:
            print("Detected both varying parameters → Plotting T1 vs Concentration")
            plot_t1_vs_parameter(results, parameter_name='concentration')

# Main execution
if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("T1 Curve Fitting Tool for NMR Data (Concentration Dependence)")
        print("=" * 70)
        print("Usage:")
        print("  python T1_exp_fit_Gly_conc.py <filepath>  - Fit single file")
        print("  python T1_exp_fit_Gly_conc.py <folder>    - Fit all files in folder")
        print("=" * 70)
        print("\nParameters:")
        print("  filepath:   Path to Excel file (.xlsx or .xls)")
        print("  folder:     Path to folder containing Excel files")
        print("\nFeatures:")
        print("  • Automatically extracts concentration from filename (e.g., Gly_100_*.xlsx)")
        print("  • Plots T1 vs Concentration if concentration varies in the dataset")
        print("  • Figures saved with same directory structure as data files")
        print("\nExample:")
        print("  python T1_exp_fit_Gly_conc.py /path/to/folder/")
        sys.exit(0)
    
    arg = sys.argv[1].strip().strip("'\"")
    
    if os.path.isfile(arg):
        # Single file processing
        fit_curve(arg)
    elif os.path.isdir(arg):
        # Folder processing
        fit_multiple_files(arg)
    else:
        print(f"Error: Path '{arg}' not found.")