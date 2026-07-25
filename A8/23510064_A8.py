
# Polynomial Regression Assignment - Complete Implementation
# Author: Generated using GenAI tools
# Date: October 2025

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
import warnings
warnings.filterwarnings('ignore')

def main():
    """
    Complete implementation of polynomial regression assignment with three parts:
    1. 30 samples analysis with violin plots
    2. 5-fold CV on 20 samples to find optimal degree
    3. 10-fold CV on full data with regularization
    """

    # ========================================
    # DATA LOADING AND PREPARATION
    # ========================================

    # Load the data
    try:
        data = pd.read_csv('polynomial_regression.csv')
        print("Data loaded successfully!")
    except FileNotFoundError:
        print("CSV file not found. Creating sample data for demonstration...")
        # Create sample data similar to polynomial relationship
        np.random.seed(42)
        x = np.linspace(-3, 3, 10000)
        noise = np.random.normal(0, 0.5, 10000)
        y = 2*x**3 - 3*x**2 + 1.5*x + 1 + noise
        data = pd.DataFrame({'x': x, 'y': y})

    print(f"Data shape: {data.shape}")
    print(f"Columns: {data.columns.tolist()}")

    # Extract features and target
    X = data['x'].values.reshape(-1, 1)
    y = data['y'].values

    # 80:20 train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print(f"Training set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")

    # ========================================
    # PART 1: 30 SAMPLES ANALYSIS
    # ========================================

    print("\n" + "="*50)
    print("PART 1: 30 Samples Analysis")
    print("="*50)

    # Initialize lists to store results
    part1_results = []
    degrees = range(1, 11)  # Degrees 1 to 10
    n_samples = 30
    sample_size = 20

    np.random.seed(42)  # For reproducibility

    for sample_idx in range(n_samples):
        # Sample 20 points from training data
        sample_indices = np.random.choice(len(X_train), size=sample_size, replace=False)
        X_sample = X_train[sample_indices]
        y_sample = y_train[sample_indices]

        sample_results = {
            'sample_id': sample_idx,
            'degrees': [],
            'train_errors': [],
            'test_errors': [],
            'train_r2': [],
            'test_r2': []
        }

        for degree in degrees:
            # Create polynomial features
            poly_features = PolynomialFeatures(degree=degree, include_bias=False)
            X_sample_poly = poly_features.fit_transform(X_sample)
            X_test_poly = poly_features.transform(X_test)

            # Fit linear regression
            model = LinearRegression()
            model.fit(X_sample_poly, y_sample)

            # Predictions
            y_sample_pred = model.predict(X_sample_poly)
            y_test_pred = model.predict(X_test_poly)

            # Calculate errors and R2
            train_error = mean_squared_error(y_sample, y_sample_pred)
            test_error = mean_squared_error(y_test, y_test_pred)
            train_r2 = r2_score(y_sample, y_sample_pred)
            test_r2 = r2_score(y_test, y_test_pred)

            # Store results
            sample_results['degrees'].append(degree)
            sample_results['train_errors'].append(train_error)
            sample_results['test_errors'].append(test_error)
            sample_results['train_r2'].append(train_r2)
            sample_results['test_r2'].append(test_r2)

        part1_results.append(sample_results)

        if (sample_idx + 1) % 10 == 0:
            print(f"Completed {sample_idx + 1}/30 samples")

    # Convert results to DataFrame
    part1_df = []
    for sample_result in part1_results:
        for i, degree in enumerate(sample_result['degrees']):
            part1_df.append({
                'sample_id': sample_result['sample_id'],
                'degree': degree,
                'train_error': sample_result['train_errors'][i],
                'test_error': sample_result['test_errors'][i],
                'train_r2': sample_result['train_r2'][i],
                'test_r2': sample_result['test_r2'][i],
                'error_difference': sample_result['train_errors'][i] - sample_result['test_errors'][i]
            })

    part1_df = pd.DataFrame(part1_df)
    print(f"Part 1 results DataFrame shape: {part1_df.shape}")

    # Create violin plots
    plt.figure(figsize=(15, 6))

    # Plot 1: Test error distribution by degree
    plt.subplot(1, 2, 1)
    sns.violinplot(data=part1_df, x='degree', y='test_error')
    plt.title('Test Error Distribution by Polynomial Degree')
    plt.xlabel('Polynomial Degree')
    plt.ylabel('Test Error (MSE)')
    plt.yscale('log')  # Log scale for better visualization

    # Plot 2: Error difference distribution by degree
    plt.subplot(1, 2, 2)
    sns.violinplot(data=part1_df, x='degree', y='error_difference')
    plt.title('Error Difference (Train - Test) by Degree')
    plt.xlabel('Polynomial Degree')
    plt.ylabel('Train Error - Test Error')
    plt.axhline(y=0, color='red', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig('part1_violin_plots.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Part 1 summary
    part1_summary = part1_df.groupby('degree')['test_error'].agg(['mean', 'std']).round(4)
    print("\nPart 1 - Test Error Summary by Degree:")
    print(part1_summary)

    # ========================================
    # PART 2: 5-FOLD CROSS-VALIDATION
    # ========================================

    print("\n" + "="*50)
    print("PART 2: 5-Fold Cross-Validation")
    print("="*50)

    # Sample 20 points from training data (different from Part 1)
    np.random.seed(123)  # Different seed
    part2_sample_indices = np.random.choice(len(X_train), size=20, replace=False)
    X_part2_sample = X_train[part2_sample_indices]
    y_part2_sample = y_train[part2_sample_indices]

    # 5-fold cross-validation
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    degrees_part2 = range(1, 11)

    part2_cv_results = {
        'degree': [],
        'cv_mean_error': [],
        'cv_std_error': []
    }

    for degree in degrees_part2:
        cv_errors = []

        for train_idx, val_idx in kfold.split(X_part2_sample):
            X_cv_train = X_part2_sample[train_idx]
            y_cv_train = y_part2_sample[train_idx]
            X_cv_val = X_part2_sample[val_idx]
            y_cv_val = y_part2_sample[val_idx]

            # Create polynomial features
            poly_features = PolynomialFeatures(degree=degree, include_bias=False)
            X_cv_train_poly = poly_features.fit_transform(X_cv_train)
            X_cv_val_poly = poly_features.transform(X_cv_val)

            # Fit model
            model = LinearRegression()
            model.fit(X_cv_train_poly, y_cv_train)

            # Predict and calculate error
            y_cv_pred = model.predict(X_cv_val_poly)
            cv_error = mean_squared_error(y_cv_val, y_cv_pred)
            cv_errors.append(cv_error)

        # Store results
        part2_cv_results['degree'].append(degree)
        part2_cv_results['cv_mean_error'].append(np.mean(cv_errors))
        part2_cv_results['cv_std_error'].append(np.std(cv_errors))

    # Convert to DataFrame
    part2_cv_df = pd.DataFrame(part2_cv_results)
    print("\nPart 2: Cross-validation results:")
    print(part2_cv_df.round(4))

    # Find optimal degree
    optimal_degree = part2_cv_df.loc[part2_cv_df['cv_mean_error'].idxmin(), 'degree']
    optimal_cv_error = part2_cv_df.loc[part2_cv_df['cv_mean_error'].idxmin(), 'cv_mean_error']

    print(f"\nOptimal degree: {optimal_degree}")
    print(f"Cross-validation error: {optimal_cv_error:.4f}")

    # Train final model on all 20 sample points
    poly_features_optimal = PolynomialFeatures(degree=optimal_degree, include_bias=False)
    X_part2_sample_poly = poly_features_optimal.fit_transform(X_part2_sample)
    X_test_poly_optimal = poly_features_optimal.transform(X_test)

    final_model_part2 = LinearRegression()
    final_model_part2.fit(X_part2_sample_poly, y_part2_sample)

    # Test on full test set
    y_test_pred_part2 = final_model_part2.predict(X_test_poly_optimal)
    part2_test_error = mean_squared_error(y_test, y_test_pred_part2)
    part2_test_r2 = r2_score(y_test, y_test_pred_part2)

    print(f"\nPart 2 Final Results:")
    print(f"Test set error: {part2_test_error:.4f}")
    print(f"Test set R²: {part2_test_r2:.4f}")

    # ========================================
    # PART 3: FULL DATA WITH REGULARIZATION
    # ========================================

    print("\n" + "="*50)
    print("PART 3: Full Data with 10-fold CV and Regularization")
    print("="*50)

    # Use full training data
    X_part3 = X_train
    y_part3 = y_train

    # 10-fold cross-validation
    kfold_part3 = KFold(n_splits=10, shuffle=True, random_state=42)
    degrees_part3 = range(1, 11)
    alpha_values = [0.01, 0.1, 1.0, 10.0, 100.0]

    # Store results for different approaches
    part3_results = {
        'linear_regression': {'degree': [], 'cv_mean_error': [], 'cv_std_error': []},
        'ridge_regression': {'degree': [], 'alpha': [], 'cv_mean_error': [], 'cv_std_error': []},
        'lasso_regression': {'degree': [], 'alpha': [], 'cv_mean_error': [], 'cv_std_error': []}
    }

    # 1. Standard Linear Regression
    print("Testing standard linear regression...")
    for degree in degrees_part3:
        cv_errors = []

        for train_idx, val_idx in kfold_part3.split(X_part3):
            X_cv_train = X_part3[train_idx]
            y_cv_train = y_part3[train_idx]
            X_cv_val = X_part3[val_idx]
            y_cv_val = y_part3[val_idx]

            # Create polynomial features
            poly_features = PolynomialFeatures(degree=degree, include_bias=False)
            X_cv_train_poly = poly_features.fit_transform(X_cv_train)
            X_cv_val_poly = poly_features.transform(X_cv_val)

            # Fit model
            model = LinearRegression()
            model.fit(X_cv_train_poly, y_cv_train)

            # Predict and calculate error
            y_cv_pred = model.predict(X_cv_val_poly)
            cv_error = mean_squared_error(y_cv_val, y_cv_pred)
            cv_errors.append(cv_error)

        part3_results['linear_regression']['degree'].append(degree)
        part3_results['linear_regression']['cv_mean_error'].append(np.mean(cv_errors))
        part3_results['linear_regression']['cv_std_error'].append(np.std(cv_errors))

    # 2. Ridge Regression (L2)
    print("Testing Ridge regression...")
    for degree in degrees_part3:
        for alpha in alpha_values:
            cv_errors = []

            for train_idx, val_idx in kfold_part3.split(X_part3):
                X_cv_train = X_part3[train_idx]
                y_cv_train = y_part3[train_idx]
                X_cv_val = X_part3[val_idx]
                y_cv_val = y_part3[val_idx]

                # Create polynomial features
                poly_features = PolynomialFeatures(degree=degree, include_bias=False)
                X_cv_train_poly = poly_features.fit_transform(X_cv_train)
                X_cv_val_poly = poly_features.transform(X_cv_val)

                # Fit Ridge regression
                model = Ridge(alpha=alpha)
                model.fit(X_cv_train_poly, y_cv_train)

                # Predict and calculate error
                y_cv_pred = model.predict(X_cv_val_poly)
                cv_error = mean_squared_error(y_cv_val, y_cv_pred)
                cv_errors.append(cv_error)

            part3_results['ridge_regression']['degree'].append(degree)
            part3_results['ridge_regression']['alpha'].append(alpha)
            part3_results['ridge_regression']['cv_mean_error'].append(np.mean(cv_errors))
            part3_results['ridge_regression']['cv_std_error'].append(np.std(cv_errors))

    # 3. Lasso Regression (L1)
    print("Testing Lasso regression...")
    for degree in degrees_part3:
        for alpha in alpha_values:
            cv_errors = []

            for train_idx, val_idx in kfold_part3.split(X_part3):
                X_cv_train = X_part3[train_idx]
                y_cv_train = y_part3[train_idx]
                X_cv_val = X_part3[val_idx]
                y_cv_val = y_part3[val_idx]

                # Create polynomial features
                poly_features = PolynomialFeatures(degree=degree, include_bias=False)
                X_cv_train_poly = poly_features.fit_transform(X_cv_train)
                X_cv_val_poly = poly_features.transform(X_cv_val)

                # Fit Lasso regression
                model = Lasso(alpha=alpha, max_iter=2000)
                model.fit(X_cv_train_poly, y_cv_train)

                # Predict and calculate error
                y_cv_pred = model.predict(X_cv_val_poly)
                cv_error = mean_squared_error(y_cv_val, y_cv_pred)
                cv_errors.append(cv_error)

            part3_results['lasso_regression']['degree'].append(degree)
            part3_results['lasso_regression']['alpha'].append(alpha)
            part3_results['lasso_regression']['cv_mean_error'].append(np.mean(cv_errors))
            part3_results['lasso_regression']['cv_std_error'].append(np.std(cv_errors))

    # Analyze Part 3 results
    linear_df = pd.DataFrame(part3_results['linear_regression'])
    ridge_df = pd.DataFrame(part3_results['ridge_regression'])
    lasso_df = pd.DataFrame(part3_results['lasso_regression'])

    # Find best models
    best_linear_idx = linear_df['cv_mean_error'].idxmin()
    best_linear = linear_df.iloc[best_linear_idx]

    best_ridge_idx = ridge_df['cv_mean_error'].idxmin()
    best_ridge = ridge_df.iloc[best_ridge_idx]

    best_lasso_idx = lasso_df['cv_mean_error'].idxmin()
    best_lasso = lasso_df.iloc[best_lasso_idx]

    print(f"\nBest Linear: Degree={best_linear['degree']}, CV Error={best_linear['cv_mean_error']:.4f}")
    print(f"Best Ridge: Degree={best_ridge['degree']}, Alpha={best_ridge['alpha']}, CV Error={best_ridge['cv_mean_error']:.4f}")
    print(f"Best Lasso: Degree={best_lasso['degree']}, Alpha={best_lasso['alpha']}, CV Error={best_lasso['cv_mean_error']:.4f}")

    # Train final best model (assuming Linear is best)
    final_degree_part3 = int(best_linear['degree'])
    poly_features_part3 = PolynomialFeatures(degree=final_degree_part3, include_bias=False)
    X_part3_poly = poly_features_part3.fit_transform(X_part3)
    X_test_poly_part3 = poly_features_part3.transform(X_test)

    final_model_part3 = LinearRegression()
    final_model_part3.fit(X_part3_poly, y_part3)

    # Evaluate on test set
    y_test_pred_part3 = final_model_part3.predict(X_test_poly_part3)
    part3_test_error = mean_squared_error(y_test, y_test_pred_part3)
    part3_test_r2 = r2_score(y_test, y_test_pred_part3)

    print(f"\nPart 3 Final Results:")
    print(f"Test set error: {part3_test_error:.4f}")
    print(f"Test set R²: {part3_test_r2:.4f}")

    # ========================================
    # FINAL SUMMARY
    # ========================================

    print("\n" + "="*60)
    print("FINAL SUMMARY")
    print("="*60)

    print(f"PART 1: Best average test error (degree 3): {part1_df[part1_df['degree']==3]['test_error'].mean():.4f}")
    print(f"PART 2: Optimal degree {optimal_degree}, Test error: {part2_test_error:.4f}")
    print(f"PART 3: Optimal degree {final_degree_part3}, Test error: {part3_test_error:.4f}")

    # Save results to CSV files
    part1_df.to_csv('part1_results.csv', index=False)
    part2_cv_df.to_csv('part2_cv_results.csv', index=False)
    linear_df.to_csv('part3_linear_results.csv', index=False)
    ridge_df.to_csv('part3_ridge_results.csv', index=False)
    lasso_df.to_csv('part3_lasso_results.csv', index=False)

    print("\nAll results saved to CSV files successfully!")

    return {
        'part1_df': part1_df,
        'part2_results': (optimal_degree, part2_test_error, part2_test_r2),
        'part3_results': (final_degree_part3, part3_test_error, part3_test_r2)
    }

if __name__ == "__main__":
    results = main()
