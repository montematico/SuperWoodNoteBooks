# analysis_tools.py
"""
Stress-strain analysis tools for wood materials.

This module provides functions for analyzing stress-strain data from wood samples,
including calculating Young's modulus, yield strength, and other mechanical properties.
"""

import ipywidgets as widgets
from IPython.display import display, clear_output
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy import stats
from scipy.integrate import trapezoid
import pwlf

# --- Configuration ---
StrainCol = "Strain (m/m)"
StressCol = "Stress (Pa)"
OffsetStrain = 0.002

def _analyze_stress_strain(df, ax=None, elastic_bounds_override=None):
    """
    Analyzes stress-strain data and calculates material properties.

    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing stress-strain data
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, no plotting is done.
    elastic_bounds_override : tuple, optional
        (start, end) of elastic region to override automatic detection

    Returns:
    --------
    dict
        Dictionary of calculated properties
    """
    x_full = df[StrainCol].values
    y_full = df[StressCol].values

    # --- Step 1: Determine Elastic Region ---
    if elastic_bounds_override:
        # User is providing the bounds manually via a slider
        elastic_start, elastic_limit = elastic_bounds_override
        model_used = "Manual Override"
    else:
        # Automatic detection using pwlf (the original logic)
        x_filt = x_full[y_full > np.max(y_full) * 0.01]
        y_filt = y_full[y_full > np.max(y_full) * 0.01]

        my_pwlf = pwlf.PiecewiseLinFit(x_filt, y_filt)
        breaks_3 = my_pwlf.fit(3)
        slopes_3 = my_pwlf.slopes
        breaks_2 = my_pwlf.fit(2)
        slopes_2 = my_pwlf.slopes

        if len(breaks_3) >= 4 and slopes_3[1] > slopes_3[0]:
            elastic_start, elastic_limit = breaks_3[1], breaks_3[2]
            model_used = "3-segment"
        else:
            elastic_start, elastic_limit = breaks_2[0], breaks_2[1]
            model_used = "2-segment"

    # --- Step 2: Calculate Young's Modulus from the determined/provided region ---
    elastic_region_data = df[(df[StrainCol] >= elastic_start) & (df[StrainCol] <= elastic_limit)]

    # --- Quality Metrics ---
    quality_metrics = {}

    if len(elastic_region_data) > 1:
        # Calculate Young's modulus and R² for the elastic region
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            elastic_region_data[StrainCol],
            elastic_region_data[StressCol]
        )
        youngs_modulus = slope
        quality_metrics['R_squared'] = r_value**2

        # Calculate ratio of elastic modulus to overall curve slope (should be > 1 for good fit)
        overall_slope, _, _, _, _ = stats.linregress(x_full, y_full)
        quality_metrics['Modulus_ratio'] = youngs_modulus / overall_slope if overall_slope != 0 else float('inf')

        # Calculate consistency with expected material behavior
        # 1. Elastic region should be in the beginning part of the curve
        quality_metrics['Early_region_score'] = 1.0 - (elastic_start / max(x_full))

        # 2. Points in the elastic region should have low deviation from the linear fit
        predicted = youngs_modulus * elastic_region_data[StrainCol] + intercept
        residuals = elastic_region_data[StressCol] - predicted
        quality_metrics['Residual_score'] = 1.0 / (1.0 + np.std(residuals) / np.mean(elastic_region_data[StressCol]))

        # 3. Region should have enough points for statistical significance
        quality_metrics['Data_points'] = len(elastic_region_data)
        quality_metrics['Data_coverage'] = len(elastic_region_data) / len(df)

        # Composite quality score (weighted average of metrics)
        quality_metrics['Overall_score'] = (
            0.4 * quality_metrics['R_squared'] + 
            0.2 * min(1.0, quality_metrics['Modulus_ratio'] / 2) +
            0.2 * quality_metrics['Early_region_score'] + 
            0.1 * quality_metrics['Residual_score'] +
            0.1 * min(1.0, quality_metrics['Data_points'] / 20)
        )
    else:
        # Fallback if bounds are invalid
        youngs_modulus, intercept = 0, 0
        quality_metrics = {
            'R_squared': 0, 'Modulus_ratio': 0, 'Early_region_score': 0,
            'Residual_score': 0, 'Data_points': 0, 'Data_coverage': 0,
            'Overall_score': 0
        }

    # --- Step 3: Calculate All Other Properties ---
    # These calculations use the Young's modulus from the current elastic region fit
    # (either automatic or manual override)
    ultimate_tensile_strength = df[StressCol].max()
    uts_index = df[StressCol].idxmax()
    strain_at_uts = df.loc[uts_index, StrainCol]

    # Offset line and yield strength calculation
    # Calculate the offset line using the current Young's modulus (from the elastic region fit)
    # This ensures the offset line is always parallel to the elastic region fit
    offset_intercept = intercept - youngs_modulus * OffsetStrain
    df['Offset Stress'] = youngs_modulus * df[StrainCol] + offset_intercept

    yield_strength, yield_strain, intersection_index = np.nan, np.nan, -1
    try:
        # Start searching for the yield point from the elastic limit or the offset strain,
        # whichever is greater. This ensures we find the yield point after the elastic region.
        search_df = df[df[StrainCol] > max(elastic_limit, OffsetStrain)].copy()
        search_df['diff'] = search_df[StressCol] - search_df['Offset Stress']

        # Look for where the actual stress curve crosses above the offset line
        cross_points = search_df[search_df['diff'] > 0]
        if len(cross_points) > 0:
            intersection_index = cross_points.index[0]
            yield_strength = df.loc[intersection_index, StressCol]
            yield_strain = df.loc[intersection_index, StrainCol]
    except (ValueError, IndexError):
        pass # Fail silently if no yield point found

    # Resilience and Toughness
    resilience = trapezoid(df.loc[:intersection_index][StressCol], x=df.loc[:intersection_index][StrainCol]) if intersection_index != -1 else np.nan
    toughness = trapezoid(df.loc[:uts_index][StressCol], x=df.loc[:uts_index][StrainCol])

    results = {
        'Youngs Modulus (Pa)': youngs_modulus, 'UTS (Pa)': ultimate_tensile_strength,
        'Yield Strength (Pa)': yield_strength, 'Elastic Limit Strain': elastic_limit, 'Model Type': model_used,
        'Resilience (J/m^3)': resilience, 'Toughness (J/m^3)': toughness,
        # Quality metrics
        'R_squared': quality_metrics['R_squared'],
        'Modulus_ratio': quality_metrics['Modulus_ratio'],
        'Early_region_score': quality_metrics['Early_region_score'],
        'Residual_score': quality_metrics['Residual_score'],
        'Data_points': quality_metrics['Data_points'],
        'Data_coverage': quality_metrics['Data_coverage'],
        'Overall_quality_score': quality_metrics['Overall_score']
    }

    # --- Step 4: Plotting ---
    # Clear the axes on each update to prevent over-plotting
    ax.clear()

    # Plot main curve
    ax.plot(x_full, y_full, label=f'Data (E = {youngs_modulus / 1e9:.2f} GPa)')

    # Plot linear fit line
    elastic_strains = np.array([elastic_start, elastic_limit])
    ax.plot(elastic_strains, youngs_modulus * elastic_strains + intercept, 'g--', lw=2,
            label=f'Elastic Region Fit (E = {youngs_modulus/1e9:.2f} GPa, {model_used})')

    # Plot vertical lines for bounds
    ax.axvline(elastic_start, color='r', linestyle=':', lw=1.5, label=f'Lower Bound: {elastic_start:.4f}')
    ax.axvline(elastic_limit, color='r', linestyle=':', lw=1.5, label=f'Upper Bound: {elastic_limit:.4f}')

    # Plot offset line and yield point
    if not np.isnan(yield_strength):
        # Use the current Young's modulus to ensure the offset line is parallel to the elastic region fit
        # Start the offset line at the offset strain (0.2%) and extend it past the yield point
        offset_strains = np.linspace(OffsetStrain, yield_strain * 1.1, 2)
        ax.plot(offset_strains, youngs_modulus * offset_strains + offset_intercept, 'r--', lw=1.5, 
                label=f'0.2% Offset (Parallel to E = {youngs_modulus/1e9:.2f} GPa)')
        ax.scatter(yield_strain, yield_strength, color='red', s=50, zorder=5, 
                  label=f'Yield: {yield_strength/1e6:.1f} MPa at ε = {yield_strain:.4f}')

    ax.set_xlabel('Strain (m/m)')
    ax.set_ylabel('Stress (Pa)')
    ax.grid(True)
    ax.legend(loc='best')

    # Add quality metrics to the plot title
    quality_text = f"Quality Score: {quality_metrics['Overall_score']:.2f}"
    ax.set_title(f"Interactive Stress-Strain Analysis - {quality_text}")

    # Add a text box with detailed quality metrics
    textstr = '\n'.join((
        f"Quality Metrics:",
        f"R² = {quality_metrics['R_squared']:.3f}",
        f"Modulus Ratio = {min(quality_metrics['Modulus_ratio'], 10):.2f}",
        f"Early Region = {quality_metrics['Early_region_score']:.2f}",
        f"Residual Score = {quality_metrics['Residual_score']:.2f}",
        f"Data Points = {quality_metrics['Data_points']}",
        f"Coverage = {quality_metrics['Data_coverage']:.2f}"
    ))

    # Position the text box in the upper right corner with a light background
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', bbox=props)

    ax.set_ylim(bottom=0)

    return results


def _format_results(results_dict):
    """
    Format results for display with appropriate units and precision.

    Parameters:
    -----------
    results_dict : dict
        Dictionary of raw calculation results

    Returns:
    --------
    dict
        Dictionary of formatted results for display
    """
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
        # Format quality metrics
        elif key == 'R_squared':
            formatted['R² Value'] = f"{value:.3f}" if pd.notna(value) else "N/A"
        elif key == 'Modulus_ratio':
            formatted['Modulus Ratio'] = f"{min(value, 10):.2f}" if pd.notna(value) else "N/A"
        elif key == 'Early_region_score':
            formatted['Early Region Score'] = f"{value:.2f}" if pd.notna(value) else "N/A"
        elif key == 'Residual_score':
            formatted['Residual Score'] = f"{value:.2f}" if pd.notna(value) else "N/A"
        elif key == 'Data_points':
            formatted['Data Points'] = f"{int(value)}" if pd.notna(value) else "N/A"
        elif key == 'Data_coverage':
            formatted['Data Coverage'] = f"{value:.2f}" if pd.notna(value) else "N/A"
        elif key == 'Overall_quality_score':
            formatted['Overall Quality Score'] = f"{value:.2f}" if pd.notna(value) else "N/A"
    return formatted


def analyze_stress_strain(file_path, interactive=True, format_output=False):
    """
    Analyze stress-strain data from a CSV file and display results.

    This is the main function of the module that provides a unified interface
    for stress-strain analysis. It can work in both interactive and non-interactive modes.

    Parameters:
    -----------
    file_path : str
        Path to the CSV file containing stress-strain data
    interactive : bool, default=True
        If True, displays an interactive plot with sliders for adjusting the elastic region
        If False, performs automatic analysis and returns the results
    format_output : bool, default=False
        If True, formats the results for display (only in non-interactive mode)

    Returns:
    --------
    dict
        Dictionary of analysis results (only in non-interactive mode)
        If format_output=True, returns formatted results for display
        In interactive mode, the function displays the plot and returns None
    """
    # Load data
    df = pd.read_csv(file_path)

    if not interactive:
        # Non-interactive mode: perform analysis and return results
        fig, ax = plt.subplots(figsize=(10, 7))
        results = _analyze_stress_strain(df.copy(), ax)
        plt.show()

        # Format results if requested
        if format_output:
            return _format_results(results)
        return results

    # Interactive mode
    max_strain = df[StrainCol].max()

    # --- Run initial automatic analysis to get starting points ---
    # We need a dummy figure/ax to call the function
    _, dummy_ax = plt.subplots()
    initial_results = _analyze_stress_strain(df.copy(), ax=dummy_ax)
    plt.close(_) # Close the dummy plot to prevent it from displaying

    initial_start_strain = df[StrainCol].iloc[1] # A safe default
    initial_limit_strain = initial_results.get('Elastic Limit Strain', max_strain / 2)

    # Get the *actual* start from the fit, not just the limit
    if 'Model Type' in initial_results and initial_results['Model Type'] != "Manual Override":
        elastic_region_data = df[(df[StrainCol] <= initial_limit_strain)]
        if not elastic_region_data.empty:
            # Find where the automatic fit started
            my_pwlf = pwlf.PiecewiseLinFit(elastic_region_data[StrainCol], elastic_region_data[StressCol])
            breaks = my_pwlf.fit(2)
            if len(breaks) > 1:
                initial_start_strain = breaks[0]

    # --- Create the Interactive Widget ---
    bounds_slider = widgets.FloatRangeSlider(
        value=[initial_start_strain, initial_limit_strain],
        min=0,
        max=max_strain,
        step=max_strain / 2000, # A reasonable step size
        description='Elastic Region:',
        disabled=False,
        continuous_update=False, # Only update when mouse is released
        orientation='horizontal',
        readout=True,
        readout_format='.4f',
        layout=widgets.Layout(width='80%')
    )

    # --- Create the figure and output widget ---
    plt.ioff()  # Turn off interactive mode to prevent automatic display
    fig, ax = plt.subplots(figsize=(10, 7))

    # Create an output widget to capture the plot
    output_widget = widgets.Output()

    # Define update function that will refresh the plot when slider changes
    def update_plot(change):
        bounds = change.new if hasattr(change, 'new') else change

        with output_widget:
            # Clear previous output
            clear_output(wait=True)

            # Update the plot
            _analyze_stress_strain(df.copy(), ax, elastic_bounds_override=bounds)

            # Display the updated figure
            display(fig)

    # Connect the slider to our update function
    bounds_slider.observe(update_plot, names='value')

    # Initial update to ensure plot is displayed
    update_plot(bounds_slider.value)

    # Display the slider and the plot output
    print("Adjust the slider to refine the linear elastic region. The plot and calculations will update.")
    display(widgets.VBox([bounds_slider, output_widget]))

    # Return None in interactive mode
    return None
