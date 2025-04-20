import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.formula.api import ols
from scipy.stats import chi2_contingency
import numpy as np
from sklearn.cluster import KMeans
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Function to validate dataset
def validate_dataset(df, required_columns):
    """Validate that the dataset contains required columns and no missing values."""
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns: {missing_cols}")
    if df[required_columns].isnull().any().any():
        raise ValueError("Dataset contains missing values in required columns")
    return True

# Function to clean and standardize the Like column
def clean_like_column(like):
    """Convert Like column to numeric values (-5 to +5)."""
    if isinstance(like, str):
        if 'I love it!+5' in like:
            return 5
        if 'I hate it!-5' in like:
            return -5
        try:
            return int(like)
        except ValueError:
            return np.nan
    return like

# Function to create a mosaic plot equivalent (heatmap of chi-squared residuals)
def plot_chi_squared_heatmap(df, row_var, col_var, row_label, col_label, title):
    """Create a heatmap of chi-squared residuals for two categorical variables."""
    cross_tab = pd.crosstab(df[row_var], df[col_var])
    chi2, p, dof, expected = chi2_contingency(cross_tab)
    
    # Check for zero expected values to avoid division by zero
    if (expected == 0).any():
        print("Warning: Zero expected values detected. Residuals may be unreliable.")
        residuals = cross_tab - expected  # Fallback to raw differences
    else:
        residuals = (cross_tab - expected) / np.sqrt(expected)
    
    plt.figure(figsize=(10, 6))
    sns.heatmap(residuals, annot=True, cmap='coolwarm', center=0, fmt=".2f",
                xticklabels=['Male', 'Female'], yticklabels=sorted(df[row_var].unique()))
    plt.title(f"{title} (p-value: {p:.4f})")
    plt.xlabel(col_label)
    plt.ylabel(row_label)
    plt.tight_layout()
    plt.savefig('segment_vs_gender_heatmap.png')
    plt.close()

# Function to plot histograms by segment
def plot_histogram_by_segment(df, var, bins, title, xlabel, ylabel):
    """Plot histograms of a variable by segment with kernel density estimation."""
    plt.figure(figsize=(12, 8))
    sns.histplot(data=df, x=var, hue='Segment', bins=bins, stat='density', 
                 alpha=0.5, kde=True, palette='Set2', element='step')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('age_distribution_by_segment.png')
    plt.close()

# Function to perform ANOVA
def perform_anova(df, dependent_var, independent_var='Segment'):
    """Perform ANOVA and print formatted results."""
    try:
        model = ols(f'{dependent_var} ~ C({independent_var})', data=df).fit()
        anova_table = sm.stats.anova_lm(model, typ=2)
        print(f"\nANOVA for {dependent_var}:")
        print(anova_table)
        p_value = anova_table['PR(>F)'][0]
        print(f"Result: {'Significant' if p_value < 0.05 else 'Not significant'} (p={p_value:.4f})")
        # Save ANOVA results
        anova_table.to_csv(f'anova_{dependent_var}.csv')
    except Exception as e:
        print(f"Error in ANOVA for {dependent_var}: {str(e)}")

# Main execution
try:
    # Load the dataset
    data = pd.read_csv('mcdonalds.csv')
    
    # Validate dataset
    required_columns = ['yummy', 'convenient', 'spicy', 'fattening', 'greasy', 
                        'fast', 'cheap', 'tasty', 'expensive', 'healthy', 
                        'disgusting', 'Like', 'Age', 'Gender']
    validate_dataset(data, required_columns)
    
    # Data cleaning and preprocessing
    # Convert perception attributes to binary (0/1)
    binary_cols = ['yummy', 'convenient', 'spicy', 'fattening', 'greasy', 
                   'fast', 'cheap', 'tasty', 'expensive', 'healthy', 'disgusting']
    for col in binary_cols:
        data[col] = data[col].map({'Yes': 1, 'No': 0})
    
    # Clean Like column
    data['Like'] = data['Like'].apply(clean_like_column)
    
    # Create FEMALE column from Gender
    data['FEMALE'] = data['Gender'].map({'Female': 1, 'Male': 0})
    
    # Ensure Age is numeric
    data['Age'] = pd.to_numeric(data['Age'], errors='coerce')
    
    # Drop rows with missing values
    data = data.dropna(subset=binary_cols + ['Like', 'Age', 'FEMALE'])
    
    # Perform k-means clustering to create segments
    X = data[binary_cols]
    kmeans = KMeans(n_clusters=4, random_state=42)  # 4 segments as per common practice
    data['Segment'] = kmeans.fit_predict(X).astype(str)  # Convert to string for categorical
    
    # 1. Mosaic Plot Equivalent (Segment vs Gender)
    plot_chi_squared_heatmap(data, 'Segment', 'FEMALE', 'Segment', 'Gender',
                             'Segment vs Gender (Chi-squared Residuals)')
    
    # 2. Histogram of Age by Segment
    age_min, age_max = int(data['Age'].min()), int(data['Age'].max())
    bins = np.arange(age_min, age_max + 5, 5)
    plot_histogram_by_segment(data, 'Age', bins, 
                             'Age Distribution by Segment', 'Age', 'Density')
    
    # 3. ANOVA for Age and Like
    perform_anova(data, 'Age')
    perform_anova(data, 'Like')
    
except FileNotFoundError:
    print("Error: 'mcdonalds.csv' not found. Please ensure the file exists.")
except Exception as e:
    print(f"Error: {str(e)}")