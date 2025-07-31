# analysis_tools.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy import stats
from scipy.integrate import trapezoid
import pwlf  # pip install pwlf

# --- Data Analysis Functions ---
def generate_report(csv_file, ax=None):
    """
    Loads a CSV, calculates material properties, and plots the stress-strain curve.
    Uses strain as a unitless ratio throughout with dynamic Young's modulus calculation
    that accounts for possible toe regions.
    """
    # --- Configuration ---
    StrainCol = "Strain (m/m)"
    StressCol = "Stress (Pa)"
    OffsetStrain = 0.002

    df = pd.read_csv(csv_file)
    
    # --- Dynamic Young's Modulus Calculation ---
    # Use piecewise linear fit to find the elastic region automatically
    x = df[StrainCol].values
    y = df[StressCol].values
    
    # Filter out any initial zero or very low strain/stress points
    valid_idx = y > np.max(y) * 0.01  # Filter points with stress > 1% of max
    if np.sum(valid_idx) < len(y):  # Only filter if necessary
        x = x[valid_idx]
        y = y[valid_idx]
    
    # Try fitting with 3 segments to account for possible toe region
    my_pwlf = pwlf.PiecewiseLinFit(x, y)
    breaks_3 = my_pwlf.fit(3)  # 3 segments: toe region, elastic, plastic
    slopes_3 = my_pwlf.slopes
    
    # Also try with 2 segments as backup
    breaks_2 = my_pwlf.fit(2)
    slopes_2 = my_pwlf.slopes
    
    # Determine which model to use based on segment characteristics
    if len(breaks_3) >= 4 and slopes_3[1] > slopes_3[0]:
        # If 3-segment model has increasing slope in second segment, likely has toe region
        # Use the second segment (index 1) as the elastic region
        youngs_modulus = slopes_3[1]
        elastic_start = breaks_3[1]
        elastic_limit = breaks_3[2]
        model_used = "3-segment"
    else:
        # Otherwise use 2-segment model (simpler case)
        youngs_modulus = slopes_2[0]  # First segment is elastic region
        elastic_start = breaks_2[0]
        elastic_limit = breaks_2[1]
        model_used = "2-segment"
    
    # --- Fix 1: Properly position the elastic region line ---
    # Get actual data points in the elastic region
    elastic_region_data = df[(df[StrainCol] >= elastic_start) & (df[StrainCol] <= elastic_limit)]
    
    if len(elastic_region_data) > 1:
        # Recalculate slope and intercept directly from the data points
        slope, intercept, _, _, _ = stats.linregress(
            elastic_region_data[StrainCol], 
            elastic_region_data[StressCol]
        )
        # Update Young's modulus with this more precise value
        youngs_modulus = slope
    else:
        # Fallback if too few points in the elastic region
        intercept = 0
    
    # Define the elastic region fit function using slope-intercept form
    elastic_region_fit = lambda x: youngs_modulus * x + intercept
    
    # --- Rest of calculations ---
    ultimate_tensile_strength = df[StressCol].max()
    uts_index = df[StressCol].idxmax()
    strain_at_uts = df.loc[uts_index, StrainCol]
    
    # --- Fix 2: Properly calculate the 0.2% offset line ---
    # Standard 0.2% offset method: shift the elastic line to the right by 0.002 strain
    offset_intercept = intercept - youngs_modulus * OffsetStrain
    df['Offset Stress'] = youngs_modulus * df[StrainCol] + offset_intercept
    
    # --- Fix 3: Better intersection calculation for yield point ---
    yield_strength = np.nan
    yield_strain = np.nan
    intersection_index = -1
    
    try:
        # Look for where stress curve crosses above offset line, after the toe region
        min_strain = max(elastic_start, OffsetStrain * 2)  # Avoid very early data
        search_df = df[df[StrainCol] > min_strain].copy()
        
        # Calculate difference between stress and offset line
        search_df['diff'] = search_df[StressCol] - search_df['Offset Stress']
        
        # Fixed approach: Manually find where the difference changes sign
        search_df['diff_sign'] = np.sign(search_df['diff'])
        search_df['sign_change'] = search_df['diff_sign'].diff() != 0
        
        # Find where diff goes from negative to positive (crosses zero)
        cross_points = search_df[(search_df['sign_change']) & (search_df['diff'] >= 0)]
        
        if len(cross_points) > 0:
            # Get first crossing point
            intersection_index = cross_points.index[0]
            yield_strength = df.loc[intersection_index, StressCol]
            yield_strain = df.loc[intersection_index, StrainCol]
        else:
            # If no sign change, find point where difference is closest to zero
            closest_idx = search_df['diff'].abs().idxmin()
            if closest_idx and search_df.loc[closest_idx, 'diff'] < youngs_modulus * 0.001:  # Reasonable tolerance
                intersection_index = closest_idx
                yield_strength = df.loc[intersection_index, StressCol]
                yield_strain = df.loc[intersection_index, StrainCol]
    except (ValueError, IndexError):
        print(f"Warning: 0.2% offset yield strength could not be determined for {csv_file}.")

    # --- Resilience and Toughness (Integration) ---
    resilience = np.nan
    if intersection_index != -1:
        df_to_yield = df.loc[:intersection_index]
        resilience = trapezoid(df_to_yield[StressCol], x=df_to_yield[StrainCol])

    df_to_uts = df.loc[:uts_index]
    toughness = trapezoid(df_to_uts[StressCol], x=df_to_uts[StrainCol])

    # --- Results Dictionary ---
    results = {
        'Youngs Modulus (Pa)': youngs_modulus, 
        'UTS (Pa)': ultimate_tensile_strength,
        'Strain at UTS': strain_at_uts, 
        'Yield Strength (Pa)': yield_strength,
        'Yield Strain': yield_strain, 
        'Elastic Limit Strain': elastic_limit,
        'Model Type': model_used,
        'Resilience (J/m^3)': resilience,
        'Toughness to UTS (J/m^3)': toughness, 
        'File': csv_file
    }

    # --- Plotting ---
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.set_title(f'Stress-Strain Curve for {os.path.basename(csv_file)}')

    label = f'{os.path.basename(csv_file)} (E = {youngs_modulus / 1e9:.2f} GPa)'
    ax.plot(df[StrainCol], df[StressCol], label=label)
    
    # Plot elastic region fit
    plot_strain_min = max(0, elastic_start - 0.001)  # Start slightly before elastic region
    plot_strain_max = min(elastic_limit + 0.002, strain_at_uts)  # End slightly after
    elastic_strains = np.linspace(plot_strain_min, plot_strain_max, 100)
    elastic_stresses = [elastic_region_fit(s) for s in elastic_strains]
    ax.plot(elastic_strains, elastic_stresses, 'g--', linewidth=1.5, 
            label=f'Elastic Region ({model_used})')
    
    if not np.isnan(yield_strength):
        # Plot offset line with proper positioning
        offset_strains = np.linspace(OffsetStrain, yield_strain + 0.002, 100)
        offset_stresses = [youngs_modulus * s + offset_intercept for s in offset_strains]
        ax.plot(offset_strains, offset_stresses, 'r--', linewidth=1.5, label='0.2% Offset Line')
        
        # Add yield point marker
        ax.scatter(yield_strain, yield_strength, color='red', s=50, zorder=5)
        ax.annotate(f'Yield Point ({yield_strain:.4f}, {yield_strength/1e6:.1f} MPa)', 
                    xy=(yield_strain, yield_strength), 
                    xytext=(yield_strain + 0.005, yield_strength * 0.8))
        
        # Add offset strain marker
        ax.scatter(OffsetStrain, 0, color='blue', s=30, zorder=5)
        ax.annotate(f'0.2% Offset', xy=(OffsetStrain, 0), 
                    xytext=(OffsetStrain, ultimate_tensile_strength*0.05),
                    arrowprops=dict(facecolor='blue', shrink=0.05),
                    horizontalalignment='center')
    
    ax.set_xlabel('Strain (m/m)')
    ax.set_ylabel('Stress (Pa)')
    ax.set_ylim(bottom=0)
    ax.grid(True)
    ax.legend()

    return results
# --- Format and Display Results ---
def format_results(results_dict):
    """Helper function to make results more readable."""
    formatted = {}
    for key, value in results_dict.items():
        if 'Youngs Modulus' in key or 'UTS' in key or 'Yield Strength' in key:
            # Convert Pascals to GigaPascals (GPa)
            formatted[key.replace('(Pa)', '(GPa)')] = f"{value / 1e9:.3f}" if pd.notna(value) else "N/A"
        elif '(J/m^3)' in key:
             # Convert J/m^3 to kJ/m^3
            formatted[key.replace('(J/m^3)', '(kJ/m^3)')] = f"{value / 1e3:.2f}" if pd.notna(value) else "N/A"
        elif 'Strain' in key and 'at' in key:
            # Display strain as unitless value
            formatted[key] = f"{value:.4f}" if pd.notna(value) else "N/A"
        elif 'Yield Strain' == key or 'Elastic Limit Strain' == key:
            # Display strain as unitless value
            formatted[key] = f"{value:.4f}" if pd.notna(value) else "N/A"
        elif 'File' in key:
            # Shorten the file path for display
            formatted[key] = os.path.basename(value)
    return formatted