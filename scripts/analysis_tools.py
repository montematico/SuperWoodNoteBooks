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
import warnings


class Analysis:
    """
    Class for performing stress-strain analysis on a given dataset.
    """
    #Configuartion Params
    __strain_col = "Strain (m/m)"  # Columns as they are named in the .csv
    __stress_col = "Stress (Pa)"  # "
    __lowerBndGuess = 0.02 #Where the lower bound of the linear elastic region is guessed -- optimized in __findLowerElastic
    __rThreshhold = 0.96 #Lower limit of residuals to be considered "good"
    __stressScale = 1e-6 #Conversion factor to use on plotting Pa -> MPa

    def __init__(self,df,name="",Ax = None):
        """
        Initialize the analysis object.
        Params:
        df : pandas.DataFrame
            DataFrame containing stress-strain data
        name : str, optional
            Names the curve, used in conjuntion with the Ax parameter
        ax : matplotlib.axes.Axes, optional
            Axes to plot on. If None, no plotting is done.
        """

        #Create local copy of the data
        self.df = df[[self.__stress_col,self.__strain_col]].copy() #creates a copy of the dataframe
        self.df.rename(columns={self.__stress_col:'Stress',self.__strain_col:'Strain'},inplace=True) #renames the columns
        self.df.drop_duplicates(subset=["Strain"], keep='first',inplace=True) #drops duplicates

        #Init params
        if name == "":
            warnings.warn("No name provided for the curve. Using the default name.")
            self.__Name = "Curve"
        else:
            self.__Name = name

        self.__Ax = Ax
        self.__ElasticBound = [None,None] #contains upper and lower limit of elastic bounds


        self.Yield = { "Stress": None,"Strain": None, "idx": None}
        self.__findElastic()

        if self.__Ax is not None:
            #Axes passed, assume plotting
            self.__plot()
        else:
            warnings.warn("No Axes provided. Plotting disabled.")

        #Material Properties Dictionary
        self.Properties = {
            "YoungsModulus": None,
            "YieldStrength": None,
            "UltimateStrength": None,
        }


    def __plot(self):
        #Plots Stress Strain curve, converting to MPa
        self.__Ax.plot(self.df["Strain"],self.df["Stress"]*self.__stressScale,label="Experimental Data",color='b')
        self.__Ax.set_xlabel("Strain")
        self.__Ax.set_ylabel("Stress (MPa)")
        self.__Ax.set_title(self.__Name)
        self.__Ax.grid(True)
        self.__DrawLinearElastic()
        self.__Ax.scatter(self.Yield["Strain"],self.Yield["Stress"]*self.__stressScale,color="red",s=150,label=f"Yield Point: {(self.Yield["Stress"]*self.__stressScale):.1f} MPa")

        self.__Ax.legend()
        return self.__Ax

    def __DrawLinearElastic(self):
        Youngs = pd.DataFrame({
            "X": self.df["Strain"],
            "Y": (self.df["Strain"]*self.LinElastic["Slope"] + self.LinElastic["Intercept"])
        })
        Youngs = Youngs.iloc[self.__ElasticBound[0]:self.__ElasticBound[1]]
        self.__Ax.plot(Youngs["X"],Youngs["Y"]*self.__stressScale,color="red",linestyle="-.",label=f"Lin Elastic: Y = {(self.LinElastic["Slope"]*self.__stressScale):.1f} MPa")
        return self.__Ax


    def __findElastic(self):
        #Finds the Upper and Lower bounds of the linear elastic region
        UpperIdx = self.__findYield() #Finds the Upper bound based on the first decrease in stress
        #Finds the lower bounds using PWLF
        self.__findLowerElastic(UpperIdx)

        #Checks residuals to ensure they are above __rThreshhold
        #todo add r^2 check here to ensure it is beyond some threshhold value
        elastic_df = self.df.iloc[self.__ElasticBound[0]:self.__ElasticBound[1]]
        LinReg = stats.linregress(elastic_df["Strain"], elastic_df["Stress"])

        if LinReg.rvalue**2 < self.__rThreshhold:
            warnings.warn("Residuals are below threshold. Check data.")

        #Will still create object even with poor fitting. Caution
        self.LinElastic = {
            "Slope": LinReg.slope,
            "Intercept": LinReg.intercept,
            "R2": LinReg.rvalue**2,
            "UpperBound": self.__ElasticBound[0],
            "LowerBound": self.__ElasticBound[1]
        }





    def __findLowerElastic(self,Upper):
        CroppedData = self.df.iloc[0:Upper].copy()
        my_pwlf = pwlf.PiecewiseLinFit(CroppedData["Strain"],CroppedData["Stress"])

        LStrain = my_pwlf.fit_guess([self.__lowerBndGuess])[1]
        self.__ElasticBound[0] = (self.df["Strain"] - LStrain).abs().idxmin()




    def __findYield(self):
        """
        Finds the first point at which stress decreases, indicating the yielding point.
        This is done via finding the first negative point in the local derivative

        """
        self.df["StressGradient"] = np.gradient(self.df["Stress"],self.df["Strain"]) #Calculates so called "Stress Rate"
        try:
            yieldIdx = self.df[self.df["StressGradient"]<0].index[0] #Finds the first negative point in the local derivative (i.e. first yielding)
        except:
            raise Exception("No yield point found")
        else:
            self.__ElasticBound[1] = yieldIdx
            self.Yield["Stress"]= self.df.at[yieldIdx, "Stress"] #saves the yield stress
            self.Yield["Strain"]= self.df.at[yieldIdx, "Strain"] #saves the yield strain
            self.Yield["idx"]= yieldIdx #saves the index of the yield point
            return yieldIdx #used by __findElastic to find the lower bounds








# def _analyze_stress_strain(df, ax=None, elastic_bounds_override=None):
#     """
#     Analyzes stress-strain data and calculates material properties.
#
#     Parameters:
#     -----------
#     df : pandas.DataFrame
#         DataFrame containing stress-strain data
#     ax : matplotlib.axes.Axes, optional
#         Axes to plot on. If None, no plotting is done.
#     elastic_bounds_override : tuple, optional
#         (start, end) of elastic region to use for analysis
#
#     Returns:
#     --------
#     dict
#         Dictionary of calculated properties
#     """
#     x_full = df[StrainCol].values
#     y_full = df[StressCol].values
#
#     # --- Step 1: Determine Elastic Region ---
#     # Always use provided bounds or default values
#     if elastic_bounds_override:
#         elastic_start, elastic_limit = elastic_bounds_override
#     else:
#         # Use simple defaults if no bounds provided
#         elastic_start = x_full[1]  # Skip the first point which might be zero
#         elastic_limit = np.max(x_full) / 3  # Use first third as a default
#
#     model_used = "Manual Selection"
#
#     # --- Step 2: Calculate Young's Modulus from the determined/provided region ---
#     elastic_region_data = df[(df[StrainCol] >= elastic_start) & (df[StrainCol] <= elastic_limit)]
#
#     # --- Quality Metrics ---
#     quality_metrics = {}
#
#     if len(elastic_region_data) > 1:
#         # Calculate Young's modulus and R² for the elastic region
#         slope, intercept, r_value, p_value, std_err = stats.linregress(
#             elastic_region_data[StrainCol],
#             elastic_region_data[StressCol]
#         )
#         youngs_modulus = slope
#         quality_metrics['R_squared'] = r_value**2
#
#         # Calculate ratio of elastic modulus to overall curve slope (should be > 1 for good fit)
#         overall_slope, _, _, _, _ = stats.linregress(x_full, y_full)
#         quality_metrics['Modulus_ratio'] = youngs_modulus / overall_slope if overall_slope != 0 else float('inf')
#
#         # Calculate consistency with expected material behavior
#         # 1. Elastic region should be in the beginning part of the curve
#         quality_metrics['Early_region_score'] = 1.0 - (elastic_start / max(x_full))
#
#         # 2. Points in the elastic region should have low deviation from the linear fit
#         predicted = youngs_modulus * elastic_region_data[StrainCol] + intercept
#         residuals = elastic_region_data[StressCol] - predicted
#         quality_metrics['Residual_score'] = 1.0 / (1.0 + np.std(residuals) / np.mean(elastic_region_data[StressCol]))
#
#         # 3. Region should have enough points for statistical significance
#         quality_metrics['Data_points'] = len(elastic_region_data)
#         quality_metrics['Data_coverage'] = len(elastic_region_data) / len(df)
#
#         # Composite quality score (weighted average of metrics)
#         quality_metrics['Overall_score'] = (
#             0.4 * quality_metrics['R_squared'] +
#             0.2 * min(1.0, quality_metrics['Modulus_ratio'] / 2) +
#             0.2 * quality_metrics['Early_region_score'] +
#             0.1 * quality_metrics['Residual_score'] +
#             0.1 * min(1.0, quality_metrics['Data_points'] / 20)
#         )
#     else:
#         # Fallback if bounds are invalid
#         youngs_modulus, intercept = 0, 0
#         quality_metrics = {
#             'R_squared': 0, 'Modulus_ratio': 0, 'Early_region_score': 0,
#             'Residual_score': 0, 'Data_points': 0, 'Data_coverage': 0,
#             'Overall_score': 0
#         }
#
#     # --- Step 3: Calculate All Other Properties ---
#     # These calculations use the Young's modulus from the current elastic region fit
#     # (either automatic or manual override)
#     ultimate_tensile_strength = df[StressCol].max()
#     uts_index = df[StressCol].idxmax()
#     strain_at_uts = df.loc[uts_index, StrainCol]
#
#     # Offset line and yield strength calculation
#     # Calculate the offset line using the current Young's modulus (from the elastic region fit)
#     # This ensures the offset line is always parallel to the elastic region fit
#     # The offset line is shifted by 0.2% strain to the right of the elastic fit line
#     offset_intercept = intercept - youngs_modulus * OffsetStrain
#
#     # Print debug information about the offset line calculation
#     print(f"Young's modulus: {youngs_modulus/1e9:.2f} GPa")
#     print(f"Intercept: {intercept}")
#     print(f"Offset strain: {OffsetStrain}")
#     print(f"Offset intercept: {offset_intercept}")
#
#     # Calculate the offset stress for each strain value
#     df['Offset Stress'] = youngs_modulus * df[StrainCol] + offset_intercept
#
#     yield_strength, yield_strain, intersection_index = np.nan, np.nan, -1
#     try:
#         # Start searching for the yield point from the offset strain.
#         # The yield point can be beyond the elastic region bounds.
#         # search_df = df[df[StrainCol] >= OffsetStrain].copy()
#         # CORRECTED: Start searching for the yield point immediately after the elastic region ends.
#         search_df = df[df[StrainCol] >= elastic_limit].copy()
#
#         # Calculate the difference between actual stress and offset line stress
#         search_df['diff'] = search_df[StressCol] - search_df['Offset Stress']
#
#         # Debug information
#         print(f"Searching for yield point. Min strain: {search_df[StrainCol].min()}, Max strain: {search_df[StrainCol].max()}")
#         print(f"Min diff: {search_df['diff'].min()}, Max diff: {search_df['diff'].max()}")
#
#         # Print the first few rows of search_df to help diagnose issues
#         print("\nFirst few rows of search data:")
#         print(search_df[['Strain (m/m)', 'Stress (Pa)', 'Offset Stress', 'diff']].head())
#
#         # Find where the stress-strain curve transitions from below to above the offset line
#         # This is a more accurate way to find the true intersection point
#         # First, find points where diff changes from negative to positive
#         search_df['diff_sign'] = np.sign(search_df['diff'])
#         search_df['sign_change'] = search_df['diff_sign'].diff()
#
#         # Look for sign changes from negative to positive (value of 2)
#         # or points that start positive (first point has no diff, so we check if it's positive)
#         crossing_indices = search_df[(search_df['sign_change'] == 2) |
#                                     ((search_df.index == search_df.index[0]) &
#                                      (search_df['diff_sign'] == 1))].index.tolist()
#
#         print(f"Found {len(crossing_indices)} true crossing points")
#
#         if len(crossing_indices) > 0:
#             # Get the first true crossing point
#             first_cross_idx = crossing_indices[0]
#
#             # If this is the first point and it's already positive, use it directly
#             if first_cross_idx == search_df.index[0]:
#                 print(f"Yield point occurs at or before the elastic limit. Interpolating...")
#                 # Find the index of the point in the *original dataframe* immediately before the search started
#                 prev_idx_loc = df.index.get_loc(first_cross_idx) - 1
#                 if prev_idx_loc >= 0:
#                     prev_idx = df.index[prev_idx_loc]
#
#                     # Get coordinates of the point before and the point at the elastic limit
#                     strain1, stress1 = df.loc[prev_idx, [StrainCol, StressCol]]
#                     offset1 = df.loc[prev_idx, 'Offset Stress']
#                     strain2, stress2 = df.loc[first_cross_idx, [StrainCol, StressCol]]
#                     offset2 = df.loc[first_cross_idx, 'Offset Stress']
#
#                     # Perform the same interpolation as the main case
#                     if strain2 != strain1:
#                         t = (offset1 - stress1) / ((stress2 - stress1) - (offset2 - offset1))
#                         yield_strain = strain1 + t * (strain2 - strain1)
#                         yield_strength = stress1 + t * (stress2 - stress1)
#                         intersection_index = first_cross_idx
#                 else:
#                     # Fallback if there's no previous point (unlikely)
#                     intersection_index = first_cross_idx
#                     yield_strength = df.loc[intersection_index, StressCol]
#                     yield_strain = df.loc[intersection_index, StrainCol]
#             else:
#                 # Get the point before the crossing (below the line) and the point after (above the line)
#                 # to interpolate the exact crossing point
#                 prev_idx = search_df.index[search_df.index.get_loc(first_cross_idx) - 1]
#
#                 # Get the coordinates of the two points
#                 strain1 = df.loc[prev_idx, StrainCol]
#                 stress1 = df.loc[prev_idx, StressCol]
#                 offset1 = youngs_modulus * strain1 + offset_intercept
#
#                 strain2 = df.loc[first_cross_idx, StrainCol]
#                 stress2 = df.loc[first_cross_idx, StressCol]
#                 offset2 = youngs_modulus * strain2 + offset_intercept
#
#                 # Interpolate to find the exact crossing point
#                 # At the crossing point, stress = offset_line
#                 # We can use linear interpolation between the two points
#                 if strain2 != strain1:  # Avoid division by zero
#                     # Calculate the fraction of the way between the two points
#                     # where the crossing occurs
#                     t = (offset1 - stress1) / ((stress2 - stress1) - (offset2 - offset1))
#
#                     # Interpolate the strain and stress at the crossing point
#                     yield_strain = strain1 + t * (strain2 - strain1)
#                     yield_strength = stress1 + t * (stress2 - stress1)
#                     #yield_strength = youngs_modulus * yield_strain + offset_intercept
#
#                     # Use the index of the point after crossing for reference
#                     intersection_index = first_cross_idx
#
#                     print(f"Interpolated yield point between indices {prev_idx} and {first_cross_idx}")
#                 else:
#                     # Fallback if the strains are identical (unlikely)
#                     intersection_index = first_cross_idx
#                     yield_strength = df.loc[intersection_index, StressCol]
#                     yield_strain = df.loc[intersection_index, StrainCol]
#                     print(f"Could not interpolate (identical strains). Using point after crossing.")
#
#             print(f"Yield strength found: {yield_strength/1e6:.2f} MPa at strain {yield_strain:.4f}")
#
#             # Print the crossing point details
#             print(f"Crossing point details:")
#             print(f"  Strain: {yield_strain}")
#             print(f"  Stress: {yield_strength/1e6:.2f} MPa")
#             print(f"  Offset line value: {(youngs_modulus * yield_strain + offset_intercept)/1e6:.2f} MPa")
#             print(f"  Difference: {(yield_strength - (youngs_modulus * yield_strain + offset_intercept))/1e6:.2f} MPa")
#         else:
#             # If no crossing point is found, try the original method as a fallback
#             cross_points = search_df[search_df['diff'] > 0]
#             print(f"No sign change found. Checking for points above offset line: {len(cross_points)} found")
#
#             if len(cross_points) > 0:
#                 # Get the first point where stress is above the offset line
#                 intersection_index = cross_points.index[0]
#                 yield_strength = df.loc[intersection_index, StressCol]
#                 yield_strain = df.loc[intersection_index, StrainCol]
#                 print(f"Using first point above offset line: {yield_strength/1e6:.2f} MPa at strain {yield_strain:.4f}")
#             else:
#                 # If still no crossing point is found, try to find the point where the difference is closest to zero
#                 # This is a fallback method that might help in cases where the curves are very close but don't cross
#                 closest_index = search_df['diff'].abs().idxmin()
#                 if closest_index is not None:
#                     yield_strength = df.loc[closest_index, StressCol]
#                     yield_strain = df.loc[closest_index, StrainCol]
#                     print(f"No crossing point found. Using closest point: {yield_strength/1e6:.2f} MPa at strain {yield_strain:.4f}")
#
#                     # Print the closest point details
#                     print(f"Closest point details:")
#                     print(f"  Strain: {yield_strain}")
#                     print(f"  Stress: {yield_strength/1e6:.2f} MPa")
#                     print(f"  Offset line value: {(youngs_modulus * yield_strain + offset_intercept)/1e6:.2f} MPa")
#                     print(f"  Difference: {(yield_strength - (youngs_modulus * yield_strain + offset_intercept))/1e6:.2f} MPa")
#     except (ValueError, IndexError) as e:
#         print(f"Error calculating yield strength: {str(e)}")
#         # Fail silently if no yield point found
#
#     # Resilience and Toughness
#     resilience = trapezoid(df.loc[:intersection_index][StressCol], x=df.loc[:intersection_index][StrainCol]) if intersection_index != -1 else np.nan
#     toughness = trapezoid(df.loc[:uts_index][StressCol], x=df.loc[:uts_index][StrainCol])
#
#     results = {
#         'Youngs Modulus (Pa)': youngs_modulus, 'UTS (Pa)': ultimate_tensile_strength,
#         'Yield Strength (Pa)': yield_strength, 'Elastic Limit Strain': elastic_limit, 'Model Type': model_used,
#         'Resilience (J/m^3)': resilience, 'Toughness (J/m^3)': toughness,
#         # Quality metrics
#         'R_squared': quality_metrics['R_squared'],
#         'Modulus_ratio': quality_metrics['Modulus_ratio'],
#         'Early_region_score': quality_metrics['Early_region_score'],
#         'Residual_score': quality_metrics['Residual_score'],
#         'Data_points': quality_metrics['Data_points'],
#         'Data_coverage': quality_metrics['Data_coverage'],
#         'Overall_quality_score': quality_metrics['Overall_score']
#     }
#
#     # --- Step 4: Plotting ---
#     # Clear the axes on each update to prevent over-plotting
#     ax.clear()
#
#     # Plot main curve
#     ax.plot(x_full, y_full, label=f'Data (E = {youngs_modulus / 1e9:.2f} GPa)')
#
#     # Plot linear fit line
#     elastic_strains = np.array([elastic_start, elastic_limit])
#     ax.plot(elastic_strains, youngs_modulus * elastic_strains + intercept, 'g--', lw=2,
#             label=f'Elastic Region Fit (E = {youngs_modulus/1e9:.2f} GPa, {model_used})')
#
#     # Plot vertical lines for bounds
#     ax.axvline(elastic_start, color='r', linestyle=':', lw=1.5, label=f'Lower Bound: {elastic_start:.4f}')
#     ax.axvline(elastic_limit, color='r', linestyle=':', lw=1.5, label=f'Upper Bound: {elastic_limit:.4f}')
#
#     # Plot offset line and yield point
#     # Always draw the offset line, even if no yield strength is calculated
#     # Use the current Young's modulus to ensure the offset line is parallel to the elastic region fit
#     # Start the offset line at the offset strain (0.2%) and extend it to a reasonable point
#     # Make sure the offset line extends far enough to be visible
#     max_strain_for_offset = max(x_full) * 0.8  # Use a larger portion of the max strain to ensure visibility
#     offset_strains = np.linspace(OffsetStrain, max_strain_for_offset, 2)
#     ax.plot(offset_strains, youngs_modulus * offset_strains + offset_intercept, 'r--', lw=2,
#             label=f'0.2% Offset (Parallel to E = {youngs_modulus/1e9:.2f} GPa)')
#
#     # Only plot the yield point if it was found
#     if not np.isnan(yield_strength):
#         # Make the yield point more prominent
#         ax.scatter(yield_strain, yield_strength, color='red', s=100, zorder=5,
#                   label=f'Yield: {yield_strength/1e6:.1f} MPa at ε = {yield_strain:.4f}')
#
#         # Add a vertical line at the yield strain to make it more visible
#         ax.axvline(yield_strain, color='red', linestyle='--', alpha=0.5, lw=1)
#
#         # Print the yield point coordinates for debugging
#         print(f"\nYield point plotted at: ({yield_strain}, {yield_strength/1e6:.2f} MPa)")
#
#     ax.set_xlabel('Strain (m/m)')
#     ax.set_ylabel('Stress (Pa)')
#     ax.grid(True)
#     ax.legend(loc='best')
#
#     # Add quality metrics to the plot title
#     quality_text = f"Quality Score: {quality_metrics['Overall_score']:.2f}"
#     ax.set_title(f"Interactive Stress-Strain Analysis - {quality_text}")
#
#     # Add a text box with detailed quality metrics
#     textstr = '\n'.join((
#         f"Quality Metrics:",
#         f"R² = {quality_metrics['R_squared']:.3f}",
#         f"Modulus Ratio = {min(quality_metrics['Modulus_ratio'], 10):.2f}",
#         f"Early Region = {quality_metrics['Early_region_score']:.2f}",
#         f"Residual Score = {quality_metrics['Residual_score']:.2f}",
#         f"Data Points = {quality_metrics['Data_points']}",
#         f"Coverage = {quality_metrics['Data_coverage']:.2f}"
#     ))
#
#     # Position the text box in the upper right corner with a light background
#     props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
#     ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=9,
#             verticalalignment='top', bbox=props)
#
#     # Set axis limits after all elements have been plotted
#     # Ensure all important elements are visible
#
#     # Set y-axis to start at 0 and extend to include all data points
#     y_max = max(y_full) * 1.1  # Default to 110% of max stress
#
#     # If yield point was found, ensure it's included in the visible range
#     if not np.isnan(yield_strength):
#         y_max = max(y_max, yield_strength * 1.1)
#
#     ax.set_ylim(bottom=0, top=y_max)
#
#     # Set x-axis to include the entire range needed
#     # Start from 0 and extend to include the offset line and yield point (if found)
#     x_max = max(max_strain_for_offset * 1.1, max(x_full) * 0.9)
#
#     # If yield point was found, ensure it's included in the visible range
#     if not np.isnan(yield_strain):
#         x_max = max(x_max, yield_strain * 1.1)
#
#     ax.set_xlim(left=0, right=x_max)
#
#     # Print the final axis limits for debugging
#     print(f"\nFinal plot limits: X: {ax.get_xlim()}, Y: {ax.get_ylim()}")
#
#     return results


# def _format_results(results_dict):
#     """
#     Format results for display with appropriate units and precision.
#
#     Parameters:
#     -----------
#     results_dict : dict
#         Dictionary of raw calculation results
#
#     Returns:
#     --------
#     dict
#         Dictionary of formatted results for display
#     """
#     formatted = {}
#     for key, value in results_dict.items():
#         if 'Youngs Modulus' in key or 'UTS' in key or 'Yield Strength' in key:
#             # Convert Pascals to GigaPascals (GPa)
#             formatted[key.replace('(Pa)', '(GPa)')] = f"{value / 1e9:.3f}" if pd.notna(value) else "N/A"
#         elif '(J/m^3)' in key:
#              # Convert J/m^3 to kJ/m^3
#             formatted[key.replace('(J/m^3)', '(kJ/m^3)')] = f"{value / 1e3:.2f}" if pd.notna(value) else "N/A"
#         elif 'Strain' in key and 'at' in key:
#             # Display strain as unitless value
#             formatted[key] = f"{value:.4f}" if pd.notna(value) else "N/A"
#         elif 'Yield Strain' == key or 'Elastic Limit Strain' == key:
#             # Display strain as unitless value
#             formatted[key] = f"{value:.4f}" if pd.notna(value) else "N/A"
#         elif 'File' in key:
#             # Shorten the file path for display
#             formatted[key] = os.path.basename(value)
#         # Format quality metrics
#         elif key == 'R_squared':
#             formatted['R² Value'] = f"{value:.3f}" if pd.notna(value) else "N/A"
#         elif key == 'Modulus_ratio':
#             formatted['Modulus Ratio'] = f"{min(value, 10):.2f}" if pd.notna(value) else "N/A"
#         elif key == 'Early_region_score':
#             formatted['Early Region Score'] = f"{value:.2f}" if pd.notna(value) else "N/A"
#         elif key == 'Residual_score':
#             formatted['Residual Score'] = f"{value:.2f}" if pd.notna(value) else "N/A"
#         elif key == 'Data_points':
#             formatted['Data Points'] = f"{int(value)}" if pd.notna(value) else "N/A"
#         elif key == 'Data_coverage':
#             formatted['Data Coverage'] = f"{value:.2f}" if pd.notna(value) else "N/A"
#         elif key == 'Overall_quality_score':
#             formatted['Overall Quality Score'] = f"{value:.2f}" if pd.notna(value) else "N/A"
#     return formatted

