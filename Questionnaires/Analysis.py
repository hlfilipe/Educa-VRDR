#!/usr/bin/env python3
import os
import csv
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import matplotlib
from factor_analyzer import FactorAnalyzer
from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity, calculate_kmo
import pingouin as pg  # For Cronbach's alpha using Pingouin
from scipy.stats import chi2  # To compute p-value for chi-square

def format_p_value(p):
    """
    Format the p-value:
      - If p < 0.001, return "<0.001"
      - Otherwise, return the value in scientific notation with 6 decimals.
    """
    if p < 0.001:
        return "<0.001"
    else:
        return f"{p:.6f}"

def load_data(file_path):
    """
    Load data from a file, automatically detecting the file type and CSV delimiter.
    
    Supports:
      - Excel files (.xls, .xlsx)
      - CSV files with detected delimiter (comma, semicolon, or space). 
        If the CSV loads into a single column (values separated by commas),
        the column is split into multiple columns and the first row is used as header.
    
    Parameters:
        file_path (str): Path to the data file.
        
    Returns:
        DataFrame: Loaded data or None if an error occurs.
    """
    ext = os.path.splitext(file_path)[1].lower()
    
    try:
        if ext in ['.xls', '.xlsx']:
            data = pd.read_excel(file_path)
        elif ext == '.csv':
            with open(file_path, 'r', encoding='utf-8') as f:
                sample = f.read(2048)  # Increase sample size for better detection
            try:
                dialect = csv.Sniffer().sniff(sample)
                delimiter = dialect.delimiter
            except csv.Error:
                delimiter = ','
            data = pd.read_csv(file_path, delimiter=delimiter)
            # If the DataFrame has a single column and its values contain commas,
            # split the column into multiple columns and use the first row as header.
            if data.shape[1] == 1:
                split_data = data.iloc[:, 0].str.split(',', expand=True)
                new_header = split_data.iloc[0]
                data = split_data[1:]
                data.columns = new_header
        else:
            try:
                data = pd.read_csv(file_path)
            except Exception:
                try:
                    data = pd.read_csv(file_path, delimiter=';')
                except Exception:
                    data = pd.read_csv(file_path, delimiter=' ')
        print("Data loaded successfully.\n", data.head())
        return data
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def cronbach_alpha(data):
    """
    Calculate Cronbach's alpha for the questionnaire data.
    Also computes the alpha if each item is deleted and marks with '<'
    those items for which deletion increases the alpha by at least 0.2 points.
    
    Parameters:
        data (DataFrame): The questionnaire data with items as columns.
        
    Returns:
        DataFrame: A table showing for each item its variance, the alpha if the item is deleted,
                   and an indicator ("<") if that deletion increases the alpha by at least 0.2 points.
                   Also includes summary rows for total variance and overall Cronbach's alpha.
    """
    try:
        n = data.shape[1]  # Number of items
        print("\nn is:", n)
        variances = data.var(axis=0)
        total_var = data.sum(axis=1).var()
        alpha = (n / (n - 1)) * (1 - variances.sum() / total_var)

        alpha_if_deleted = []
        for item in data.columns:
            data_without_item = data.drop(columns=[item])
            n_without = data_without_item.shape[1]
            variances_without = data_without_item.var(axis=0)
            total_var_without = data_without_item.sum(axis=1).var()
            alpha_without = (n_without / (n_without - 1)) * (1 - variances_without.sum() / total_var_without)
            alpha_if_deleted.append(alpha_without)

        indicator = []
        for a_del in alpha_if_deleted:
            if (a_del - alpha) >= 0.2:
                indicator.append("<")
            else:
                indicator.append("")

        results = pd.DataFrame({
            'Item': data.columns,
            'Variance': variances, 
            'Alpha if deleted': alpha_if_deleted,
            'Indicator': indicator
        })
        results.loc['Total'] = ['Total', total_var, '', '']
        results.loc['Alpha'] = ["Cronbach's Alpha", alpha, '', '']

        return results
    except Exception as e:
        print(f"Error calculating Cronbach's alpha: {e}")
        return None

def cronbach_alpha_pingouin(data):
    """
    Calculate Cronbach's alpha using the Pingouin library.
    
    Parameters:
        data (DataFrame): The questionnaire data with items as columns.
        
    Returns:
        dict: A dictionary with Cronbach's alpha and its 95% confidence interval.
    """
    try:
        df = data.astype(float)
        alpha_results = pg.cronbach_alpha(df)
        results = {
            "Cronbach's Alpha": alpha_results[0],
            "95% Confidence Interval": alpha_results[1]
        }
        return results
    except Exception as e:
        print(f"Error calculating Cronbach's alpha with Pingouin: {e}")
        return None

def descriptive_statistics(data):
    """
    Calculate descriptive statistics for Likert scale questionnaire data.
    
    This function uses the output of data.describe() and modifies it:
      - Renames the "50%" column to "Median".
      - Removes the "mean" column since it may not be as meaningful.
      - Adds a new column "IQR" computed as the difference between the 75% and 25% percentiles.
      - Keeps the remaining original values (count, std, min, 25%, Median, 75%, max).
    
    Parameters:
        data (DataFrame): The questionnaire data.
        
    Returns:
        DataFrame: A table with count, std, min, 25%, Median, 75%, max, and IQR for each item.
    """
    try:
        d = data.describe().T
        d = d.rename(columns={"50%": "Median"})
        if "mean" in d.columns:
            d = d.drop(columns=["mean"])
        d["IQR"] = d["75%"] - d["25%"]
        return d
    except Exception as e:
        print(f"Error calculating descriptive statistics: {e}")
        return None

def correlation_matrix(data):
    """
    Calculate the correlation matrix for the questionnaire data.
    
    Parameters:
        data (DataFrame): The questionnaire data.
        
    Returns:
        DataFrame: The correlation matrix.
    """
    try:
        corr_matrix = data.corr()
        return corr_matrix
    except Exception as e:
        print(f"Error calculating correlation matrix: {e}")
        return None

def factor_analysis(data, n_factors=None):
    """
    Perform factor analysis on the data.
    If n_factors is None, the optimal number of factors is determined using Kaiser's criterion (eigenvalue > 1).
    The analysis is performed only if the KMO index is at least 0.5.
    A message is printed beforehand indicating whether the analysis can be performed.
    
    Parameters:
        data (DataFrame): The questionnaire data.
        n_factors (int or None): Number of factors to extract. If None, the optimal number is used.
        
    Returns:
        dict or None: Results of the factor analysis including factor loadings, explained variance,
                      Bartlett's test, KMO, eigenvalues, and the optimal number of factors.
                      Returns None if the KMO index is below 0.5.
    """
    try:
        # Calculate eigenvalues (without rotation) to determine the optimal number of factors.
        fa_tmp = FactorAnalyzer(rotation=None)
        fa_tmp.fit(data)
        eigen_values, _ = fa_tmp.get_eigenvalues()
        optimal_n_factors = sum(eigen_values > 1)
        print("Eigenvalues:", eigen_values)
        print("Optimal number of factors (Kaiser's criterion):", optimal_n_factors)
        
        if n_factors is None:
            n_factors = optimal_n_factors
            print("Using the optimal number of factors:", n_factors)
        else:
            print("Number of factors specified by the user:", n_factors)
            if n_factors > data.shape[1]:
                raise ValueError(f"The number of factors ({n_factors}) cannot exceed the number of columns ({data.shape[1]}).")
        
        # Perform Bartlett's test and calculate the KMO index.
        chi_square, p_value = calculate_bartlett_sphericity(data)
        kmo_all, kmo_model = calculate_kmo(data)
        print(f"Bartlett's test: chi² = {chi_square:.6f}, p = {p_value}")
        print(f"KMO Index: {kmo_model:.3f}")
        
        # Verify the KMO index before proceeding.
        if kmo_model < 0.5:
            print(f"The KMO index is too low ({kmo_model:.3f}). Factor analysis will not be performed.")
            return None
        else:
            print(f"The KMO index is {kmo_model:.3f}, so factor analysis can be performed.")
        
        # Proceed with factor analysis using Varimax rotation.
        fa = FactorAnalyzer(n_factors=n_factors, rotation='varimax')
        fa.fit(data)
        
        loadings = pd.DataFrame(fa.loadings_, index=data.columns, 
                                columns=[f"Factor {i+1}" for i in range(n_factors)])
        loadings['Assigned Factor'] = loadings.abs().idxmax(axis=1)
        
        variance = pd.DataFrame(fa.get_factor_variance(), 
                        index=['Variance', 'Proportional Variance', 'Cumulative Variance'], 
                        columns=[f"Factor {i+1}" for i in range(n_factors)])
        
        results = {
            'Factor Loadings': loadings,
            'Explained Variance': variance,
            'Bartlett': (chi_square, p_value),
            'KMO': kmo_model,
            'Eigenvalues': eigen_values,
            'Optimal Number of Factors': optimal_n_factors
        }
        return results
    except Exception as e:
        print(f"Error during factor analysis: {e}")
        return None

def validate_factor_analysis(data, n_factors, rotation='varimax'):
    """
    Validate the factor analysis model by calculating various goodness-of-fit indices.
    This function manually computes:
      - Model chi-square, degrees of freedom (df_model), and p-value based on the discrepancy between the observed
        and reproduced correlation matrices.
      - SRMSR: Standardized Root Mean Square Residual.
      - TLI (Tucker-Lewis Index) and CFI (Comparative Fit Index) using the null model.
      - PNFI is computed as (df_model/df_null)*CFI.
      - RMSEA (Root Mean Square Error of Approximation).
      - ECVI (Expected Cross-Validation Index) is calculated as: (model_chi_square + 2*df_model)/(N - 1).
    
    The model degrees of freedom are computed as:
    
        df_model = [p(p+1)/2] - [p*m - m(m-1)/2]
    
    where p is the number of variables and m is the number of factors.
    
    Parameters:
         data (DataFrame): The questionnaire data.
         n_factors (int): Number of factors to extract.
         rotation (str): Rotation method (default 'varimax').
         
    Returns:
         dict: A dictionary with the computed goodness-of-fit indices.
    """
    try:
        N = data.shape[0]      # Sample size
        p = data.shape[1]      # Number of variables
        m = n_factors          # Number of factors

        # Fit the factor analysis model.
        fa = FactorAnalyzer(n_factors=n_factors, rotation=rotation)
        fa.fit(data)
        L = fa.loadings_
        communalities = fa.get_communalities()
        uniquenesses = 1 - communalities
        R = data.corr().values
        # Reproduced correlation matrix: R_hat = L L' + diag(uniquenesses)
        R_hat = np.dot(L, L.T) + np.diag(uniquenesses)
        
        # Compute SRMSR: sqrt(mean((R - R_hat)^2))
        srmr = np.sqrt(np.mean((R - R_hat)**2))
        
        # Compute the discrepancy function F_min.
        det_R = np.linalg.det(R)
        det_R_hat = np.linalg.det(R_hat)
        inv_R_hat = np.linalg.inv(R_hat)
        trace_term = np.trace(np.dot(R, inv_R_hat))
        F_min = np.log(det_R_hat) + trace_term - np.log(det_R) - p
        model_chi_square = (N - 1) * F_min
        
        # Compute degrees of freedom for the model using the standard formula:
        # df_model = [p(p+1)/2] - [p*m - m(m-1)/2]
        df_model = (p * (p + 1) / 2) - (p * m - m * (m - 1) / 2)
        
        # Compute p-value for the model chi-square.
        if df_model > 0:
            p_value = chi2.sf(model_chi_square, df_model)
        else:
            p_value = None
        
        # Compute the null model chi-square and degrees of freedom.
        chi_square_null = - (N - 1) * np.log(np.linalg.det(R))
        df_null = p * (p - 1) / 2
        
        # Compute TLI (Tucker-Lewis Index)
        if df_model > 0 and df_null > 0:
            tli = ((chi_square_null / df_null - model_chi_square / df_model) /
                   (chi_square_null / df_null - 1))
        else:
            tli = None

        # Compute CFI (Comparative Fit Index)
        numerator = max(model_chi_square - df_model, 0)
        denominator = max(chi_square_null - df_null, model_chi_square - df_model, 0)
        if denominator > 0:
            cfi = 1 - numerator / denominator
        else:
            cfi = None

        # Compute RMSEA (Root Mean Square Error of Approximation)
        if df_model > 0:
            rmsea = np.sqrt(max((model_chi_square - df_model) / (df_model * (N - 1)), 0))
        else:
            rmsea = None

        # Compute PNFI (Parsimonious Normed Fit Index)
        if cfi is not None and df_null > 0:
            pnfi = (df_model / df_null) * cfi
        else:
            pnfi = None

        # Compute ECVI (Expected Cross-Validation Index)
        if N - 1 > 0:
            ecvi = (model_chi_square + 2 * df_model) / (N - 1)
        else:
            ecvi = None

        indices = {
            "model_chi_square": model_chi_square,
            "df_model": df_model,
            "p_value": p_value,
            "SRMSR": srmr,
            "TLI": tli,
            "CFI": cfi,
            "RMSEA": rmsea,
            "PNFI": pnfi,
            "ECVI": ecvi
        }
        return indices
    except Exception as e:
        print(f"Error validating factor analysis: {e}")
        return None

def plot_scree_from_corr(data):
    """
    Calculate the correlation matrix of the items, obtain its eigenvalues,
    and generate a scree plot.
    A horizontal line is drawn at y=1 (Kaiser's criterion) and the optimal number of components is annotated.
    
    Parameters:
        data (DataFrame): The questionnaire data.
    """
    corr_matrix = data.corr()
    eigenvalues, _ = np.linalg.eig(corr_matrix)
    eigenvalues = np.sort(eigenvalues)[::-1]
    optimal_n = sum(eigenvalues > 1)
    
    plt.figure(figsize=(8,6))
    plt.plot(range(1, len(eigenvalues) + 1), eigenvalues, marker='o', linestyle='-')
    plt.axhline(y=1, color='r', linestyle='--', label="Line at y=1")
    plt.xlabel("Component / Factor Number")
    plt.ylabel("Eigenvalue")
    plt.title("Scree Plot (Correlation Matrix)")
    plt.xticks(range(1, len(eigenvalues) + 1))
    plt.legend()
    plt.text(0.5, 0.9, f"Optimal Components (eigenvalue > 1): {optimal_n}",
             transform=plt.gca().transAxes, fontsize=10, color='blue')
    plt.show()

def plot_correlation_heatmap(data):
    """
    Generate a heatmap of the correlation matrix of the DataFrame,
    displaying numeric correlation values in each cell.
    
    Parameters:
        data (DataFrame): The questionnaire data.
    """
    corr = data.corr()
    plt.figure(figsize=(8, 6))
    plt.imshow(corr, cmap='viridis', interpolation='none', aspect='auto')
    plt.colorbar()
    plt.xticks(ticks=np.arange(len(corr.columns)), labels=corr.columns, rotation=90)
    plt.yticks(ticks=np.arange(len(corr.index)), labels=corr.index)
    for i in range(len(corr.index)):
        for j in range(len(corr.columns)):
            plt.text(j, i, f"{corr.iloc[i, j]:.2f}", ha='center', va='center',
                     color='black', fontsize=1)
    plt.title("Correlation Matrix")
    plt.tight_layout()
    plt.show()

def plot_factor_tree_grouped(loadings):
    """
    Generate a tree diagram where each factor and its associated items are grouped.
    Each factor (and its items) is assigned a unique color and positioned so that
    factors appear in column 0 and items in column 1, with vertical spacing based on
    the circle's diameter (2*node_radius). This ensures that the circles (nodes) do not overlap.
    
    Parameters:
        loadings (DataFrame): The factor loadings matrix with an 'Assigned Factor' column.
    """
    import networkx as nx
    G = nx.DiGraph()

    # Get unique factors (sorted alphabetically)
    factors = sorted(loadings['Assigned Factor'].unique())
    
    # Create a dictionary mapping each factor to its items.
    factor_to_items = {}
    for factor in factors:
        factor_to_items[factor] = loadings.index[loadings['Assigned Factor'] == factor].tolist()

    # Get a colormap and sample colors for each factor.
    cmap = matplotlib.colormaps['tab10']
    if len(factors) > 1:
        colors = [cmap(i/(len(factors)-1)) for i in range(len(factors))]
    else:
        colors = [cmap(0.5)]
    factor_color_dict = {factor: color for factor, color in zip(factors, colors)}

    # Add nodes for factors and items.
    for factor in factors:
        G.add_node(factor, type='factor', color=factor_color_dict[factor])
        for item in factor_to_items[factor]:
            G.add_node(item, type='item', color=factor_color_dict[factor])
            weight = abs(loadings.loc[item, factor]) if factor in loadings.columns else 0
            G.add_edge(factor, item, weight=weight)

    # Set parameters for spacing:
    node_radius = 0.5        # radius of each node in coordinate units
    diameter = 2 * node_radius
    group_gap = diameter     # gap between groups equal to one diameter

    # Position nodes manually:
    # - Factors in column 0; Items in column 1; 
    # - Nodes within a group are separated vertically by the diameter.
    pos = {}
    y_offset = 0
    for factor in factors:
        items = factor_to_items[factor]
        n_items = len(items)
        # Total group height: one factor node + n_items item nodes; each node takes a vertical space of 'diameter'
        group_height = (n_items + 1) * diameter
        # Position factor at column 0, centered vertically in the group.
        pos[factor] = (0, y_offset + group_height / 2)
        # Position each item in column 1, spaced evenly from top.
        for j, item in enumerate(items):
            pos[item] = (1, y_offset + group_height - node_radius - j * diameter)
        y_offset += group_height + group_gap

    # Extract node colors.
    node_colors = [G.nodes[node]['color'] for node in G.nodes()]

    plt.figure(figsize=(10, 8))
    nx.draw(G, pos, with_labels=True, node_color=node_colors, node_size=1500, arrows=True)
    edge_labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels={k: f"{v:.2f}" for k, v in edge_labels.items()}, font_size=9)
    plt.title("Grouped Factor Tree Diagram")
    plt.axis('off')
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description=("Analyze questionnaire data.\n"
                     "Example usage: python stat_factorial_auto.py --file data.csv --cronbach --descriptive --correlation --factorial 5 --scree --tree"),
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('--file', type=str, required=True,
                        help="Path to the CSV file containing the questionnaire data.")
    parser.add_argument('--cronbach', action='store_true',
                        help="Calculate Cronbach's alpha for the questionnaire data.")
    parser.add_argument('--descriptive', action='store_true',
                        help="Calculate descriptive statistics for the data.")
    parser.add_argument('--correlation', action='store_true',
                        help="Calculate the correlation matrix for the data.")
    parser.add_argument('--factorial', type=int, nargs='?', const=-1,
                        help=("Perform factor analysis. If a number is provided, that number of factors is used; \n"
                              "if not provided (or if -1 is used), the optimal number of factors will be determined."))
    parser.add_argument('--scree', action='store_true',
                        help="Generate a scree plot based on the correlation matrix.")
    parser.add_argument('--tree', action='store_true',
                        help="Generate a grouped factor tree diagram with each factor and its items together.")
    
    args = parser.parse_args()

    data = load_data(args.file)
    if data is None:
        return

    # Convert all columns to numeric to ensure correct calculations.
    data = data.apply(pd.to_numeric, errors='coerce')

    results_df = pd.DataFrame()

    if args.cronbach:
        cronbach_results = cronbach_alpha(data)
        if cronbach_results is not None:
            print("\nCronbach's Alpha (Manual) Results:")
            print(cronbach_results)
            results_df = pd.concat([results_df, cronbach_results], axis=1)
        cronbach_pingouin_results = cronbach_alpha_pingouin(data)
        if cronbach_pingouin_results is not None:
            print("\nCronbach's Alpha (Pingouin) Results:")
            print(f"Cronbach's Alpha: {cronbach_pingouin_results['Cronbach\'s Alpha']}")
            print(f"95% Confidence Interval: {cronbach_pingouin_results['95% Confidence Interval']}\n")

    if args.descriptive:
        stats = descriptive_statistics(data)
        if stats is not None:
            print("\nDescriptive Statistics:")
            print(stats)
            results_df = pd.concat([results_df, stats], axis=1)

    if args.correlation:
        corr_matrix_result = correlation_matrix(data)
        if corr_matrix_result is not None:
            # print("\nCorrelation Matrix:")
            # print(corr_matrix_result)
            results_df = pd.concat([results_df, corr_matrix_result], axis=1)
            plot_correlation_heatmap(data)

    factorial_results = None
    if args.factorial is not None:
        n_factors = None if args.factorial == -1 else args.factorial
        factorial_results = factor_analysis(data, n_factors=n_factors)
        if factorial_results is not None:
            print("\nFactor Analysis Results:")
            print("Optimal number of factors (based on eigenvalues > 1):", factorial_results['Optimal Number of Factors'])
            print("Eigenvalues:", factorial_results['Eigenvalues'])
            print("\nFactor Loadings:")
            print(factorial_results['Factor Loadings'])
            print("\nExplained Variance:")
            print(factorial_results['Explained Variance'])
            chi_sq, p_val = factorial_results['Bartlett']
            print(f"\nBartlett's Test: chi² = {chi_sq:.6f}, p = {format_p_value(p_val)}")
            print(f"KMO Index: {factorial_results['KMO']}")
            
            # Validate the factor analysis model and display goodness-of-fit indices.
            print("\nGoodness-of-Fit Indices:")
            fit_indices = validate_factor_analysis(data, n_factors=n_factors)
            if fit_indices is not None:
                if fit_indices["model_chi_square"] is not None:
                    print(f"Model Chi-square: {fit_indices['model_chi_square']:.6f}")
                else:
                    print("Model Chi-square: Not calculated")
                print(f"Degrees of Freedom: {fit_indices['df_model']}")
                if fit_indices.get("p_value", None) is not None:
                    print(f"p-value: {format_p_value(fit_indices['p_value'])}")
                else:
                    print("p-value: Not calculated")
                print(f"SRMSR: {fit_indices['SRMSR']}")
                print(f"TLI: {fit_indices['TLI']}")
                print(f"CFI: {fit_indices['CFI']}")
                print(f"RMSEA: {fit_indices['RMSEA']}")
                print(f"PNFI: {fit_indices['PNFI']}")
                print(f"ECVI: {fit_indices['ECVI']}")
            
    if args.scree:
        plot_scree_from_corr(data)
            
    if args.tree:
        if factorial_results is not None:
            plot_factor_tree_grouped(factorial_results['Factor Loadings'])
        else:
            print("No factor analysis results available to generate the tree diagram.")
            
    if not results_df.empty:
        results_df.to_csv("analysis_results.csv", index=False)
        print("\nResults saved in 'analysis_results.csv'.")

if __name__ == "__main__":
    main()
