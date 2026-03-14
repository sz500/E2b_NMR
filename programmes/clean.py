import pandas as pd
import glob
import os
import sys

def remove_negative_time(filepath, time_min=None, time_max=None):
    """
    Remove rows outside a specified time range from a CSV file and save it back.
    
    Args:
        filepath: Path to the CSV file
        time_min: Minimum time value to keep (default: None, no lower limit)
        time_max: Maximum time value to keep (default: None, no upper limit)
    
    Returns:
        Tuple (success, removed_count, original_count, new_count) or (False, error_msg)
    """
    try:
        # Read the CSV file, skipping units and blank rows
        df = pd.read_csv(filepath, skiprows=[1, 2])
        
        # Strip whitespace from column names
        df.columns = df.columns.str.strip()
        
        # Store original row count
        original_count = len(df)
        
        # Apply time range filter
        if time_min is not None:
            df = df[df['Time'] >= time_min]
        if time_max is not None:
            df = df[df['Time'] <= time_max]
        
        # Store new row count
        new_count = len(df)
        removed_count = original_count - new_count
        
        # Save back to the same file
        df.to_csv(filepath, index=False)
        
        return True, removed_count, original_count, new_count
    
    except Exception as e:
        return False, str(e)

def clean_all_files(data_folder='data/T2', time_min=None, time_max=None):
    """
    Remove data outside time range from all CSV files in a folder recursively.
    
    Args:
        data_folder: Path to the folder containing CSV files
        time_min: Minimum time value to keep (default: None, no lower limit)
        time_max: Maximum time value to keep (default: None, no upper limit)
    """
    # Find all CSV files recursively
    csv_files = glob.glob(os.path.join(data_folder, '**/*.csv'), recursive=True)
    
    if not csv_files:
        print(f"No CSV files found in {data_folder}")
        return
    
    print(f"Found {len(csv_files)} CSV file(s) to clean")
    if time_min is not None or time_max is not None:
        time_range_str = f"Time range: "
        if time_min is not None:
            time_range_str += f"{time_min}"
        else:
            time_range_str += "-∞"
        time_range_str += " to "
        if time_max is not None:
            time_range_str += f"{time_max}"
        else:
            time_range_str += "+∞"
        print(time_range_str)
    else:
        print("Time range: No filtering (keeping all data)")
    print("=" * 90)
    
    total_removed = 0
    successful = 0
    failed = 0
    
    for filepath in csv_files:
        filename = os.path.basename(filepath)
        result = remove_negative_time(filepath, time_min=time_min, time_max=time_max)
        
        if result[0]:  # Success
            removed, original, new = result[1], result[2], result[3]
            print(f"✓ {filename:<40} | Removed: {removed:6d} rows | {original} → {new}")
            total_removed += removed
            successful += 1
        else:  # Failure
            error = result[1]
            print(f"✗ {filename:<40} | Error: {error}")
            failed += 1
    
    print("=" * 90)
    print(f"Summary: {successful} files processed successfully, {failed} failed")
    print(f"Total rows removed: {total_removed}")

# Main execution
if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Data Cleaning Tool - Remove Data Outside Time Range")
        print("=" * 70)
        print("Usage:")
        print("  python clean_data.py all [directory] [min] [max]")
        print("  python clean_data.py <filepath> [min] [max]")
        print("=" * 70)
        print("\nParameters:")
        print("  directory: Folder path (default: data/T2)")
        print("  min:       Minimum time value to keep (optional)")
        print("  max:       Maximum time value to keep (optional)")
        print("\nExample:")
        print("  python clean_data.py all                    # Keep all data")
        print("  python clean_data.py all data/T2            # Keep all data in data/T2")
        print("  python clean_data.py all data/T2 0 200      # Keep 0 ≤ time ≤ 200")
        print("  python clean_data.py data/T2/file.csv 0     # Keep time ≥ 0")
        print("  python clean_data.py data/T2/file.csv 0 150 # Keep 0 ≤ time ≤ 150")
        sys.exit(0)
    
    arg = sys.argv[1].strip().strip("'\"")
    time_min = None
    time_max = None
    
    # Parse time range if provided
    if len(sys.argv) > 2 and arg.lower() == 'all':
        # For 'all' command, parameters start at index 2
        if len(sys.argv) > 2:
            data_folder = sys.argv[2].strip().strip("'\"")
            # Check if parameters are numbers
            if len(sys.argv) > 3:
                try:
                    time_min = float(sys.argv[3])
                except ValueError:
                    pass
            if len(sys.argv) > 4:
                try:
                    time_max = float(sys.argv[4])
                except ValueError:
                    pass
        clean_all_files(data_folder, time_min=time_min, time_max=time_max)
    elif arg.lower() == 'all':
        # Just 'all' with no parameters
        data_folder = 'data/T2'
        clean_all_files(data_folder, time_min=time_min, time_max=time_max)
    else:
        # Treat as filepath - handle both single file and 'all' together
        if os.path.isfile(arg):
            # Single file processing
            # Parse time parameters for single file
            if len(sys.argv) > 2:
                try:
                    time_min = float(sys.argv[2])
                except ValueError:
                    pass
            if len(sys.argv) > 3:
                try:
                    time_max = float(sys.argv[3])
                except ValueError:
                    pass
            
            print(f"Cleaning: {arg}")
            result = remove_negative_time(arg, time_min=time_min, time_max=time_max)
            if result[0]:
                removed, original, new = result[1], result[2], result[3]
                print(f"Success! Removed {removed} rows outside range")
                print(f"Total rows: {original} → {new}")
            else:
                error = result[1]
                print(f"Error: {error}")
        elif os.path.isdir(arg):
            # Directory provided instead of file
            data_folder = arg
            if len(sys.argv) > 2:
                try:
                    time_min = float(sys.argv[2])
                except ValueError:
                    pass
            if len(sys.argv) > 3:
                try:
                    time_max = float(sys.argv[3])
                except ValueError:
                    pass
            clean_all_files(data_folder, time_min=time_min, time_max=time_max)
        else:
            print(f"Error: File or directory '{arg}' not found.")
