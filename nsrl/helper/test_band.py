import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV
from scipy.stats import gaussian_kde
import time
from sklearn.metrics import mean_squared_error
import seaborn as sns

def get_scott_bandwidth(data):
    """
    Calculate Scott's rule bandwidth
    h = n**(-1/(d+4)) * std
    """
    n_samples, n_dims = data.shape
    return np.power(n_samples, -1./(n_dims + 4)) * np.std(data, axis=0).mean()

def get_silverman_bandwidth(data):
    """
    Calculate Silverman's rule bandwidth
    h = (n * (d + 2) / 4)^(-1/(d + 4)) * std
    """
    n_samples, n_dims = data.shape
    return np.power(n_samples * (n_dims + 2) / 4., -1./(n_dims + 4)) * np.std(data, axis=0).mean()

def cross_validation_bandwidth(data, cv=5):
    """
    Get bandwidth using cross-validation
    """
    bandwidths = np.logspace(-1, 1, 20)
    grid = GridSearchCV(
        KernelDensity(kernel='gaussian'),
        {'bandwidth': bandwidths},
        cv=cv
    )
    grid.fit(data)
    return grid.best_params_['bandwidth']

def compare_bandwidth_methods(n_trials=5):
    """
    Compare different bandwidth selection methods
    """
    results = {
        'scott': {'times': [], 'scores': [], 'bandwidths': []},
        'silverman': {'times': [], 'scores': [], 'bandwidths': []},
        'cv': {'times': [], 'scores': [], 'bandwidths': []}
    }
    
    # Test on different data distributions
    distributions = [
        ('Gaussian', lambda n, d: np.random.normal(0, 1, (n, d))),
        ('Uniform', lambda n, d: np.random.uniform(-1, 1, (n, d))),
        ('Mixture', lambda n, d: np.vstack([
            np.random.normal(0, 1, (n//2, d)),
            np.random.normal(3, 0.5, (n//2, d))
        ]))
    ]
    
    for dist_name, dist_func in distributions:
        print(f"\nTesting on {dist_name} distribution:")
        
        for trial in range(n_trials):
            # Generate data
            n_samples = 1000
            n_dims = 4
            train_data = dist_func(n_samples, n_dims)
            test_data = dist_func(n_samples//5, n_dims)  # 20% size of train data
            
            # Standardize data
            mean = np.mean(train_data, axis=0)
            std = np.std(train_data, axis=0)
            train_data_scaled = (train_data - mean) / np.maximum(std, 1e-8)
            test_data_scaled = (test_data - mean) / np.maximum(std, 1e-8)
            
            # Test each method
            methods = {
                'scott': get_scott_bandwidth,
                'silverman': get_silverman_bandwidth,
                'cv': cross_validation_bandwidth
            }
            
            for method_name, method_func in methods.items():
                start_time = time.time()
                bandwidth = method_func(train_data_scaled)
                end_time = time.time()
                
                # Fit KDE and evaluate on test data
                kde = KernelDensity(bandwidth=bandwidth, kernel='gaussian')
                kde.fit(train_data_scaled)
                score = kde.score(test_data_scaled)
                
                results[method_name]['times'].append(end_time - start_time)
                results[method_name]['scores'].append(score)
                results[method_name]['bandwidths'].append(bandwidth)
                
            print(f"Trial {trial + 1} completed")
            
        # Plot results for this distribution
        plot_results(results, dist_name)
        print_statistics(results, dist_name)
        
        # Reset results for next distribution
        for method in results:
            results[method] = {'times': [], 'scores': [], 'bandwidths': []}

def plot_results(results, dist_name):
    """
    Plot comparison results
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f'Bandwidth Method Comparison for {dist_name} Distribution')
    
    # Plot bandwidths
    sns.boxplot(data=[results[m]['bandwidths'] for m in results], ax=axes[0])
    axes[0].set_xticklabels(results.keys())
    axes[0].set_title('Bandwidth Values')
    axes[0].set_ylabel('Bandwidth')
    
    # Plot scores
    sns.boxplot(data=[results[m]['scores'] for m in results], ax=axes[1])
    axes[1].set_xticklabels(results.keys())
    axes[1].set_title('Log-Likelihood Scores')
    axes[1].set_ylabel('Score')
    
    # Plot computation times
    sns.boxplot(data=[results[m]['times'] for m in results], ax=axes[2])
    axes[2].set_xticklabels(results.keys())
    axes[2].set_title('Computation Time')
    axes[2].set_ylabel('Time (seconds)')
    
    plt.tight_layout()
    plt.show()

def print_statistics(results, dist_name):
    """
    Print statistical summary of results
    """
    print(f"\nResults for {dist_name} distribution:")
    print("\nMean values:")
    for method in results:
        print(f"\n{method.capitalize()} method:")
        print(f"  Bandwidth: {np.mean(results[method]['bandwidths']):.6f} ± {np.std(results[method]['bandwidths']):.6f}")
        print(f"  Score: {np.mean(results[method]['scores']):.6f} ± {np.std(results[method]['scores']):.6f}")
        print(f"  Time: {np.mean(results[method]['times']):.6f} ± {np.std(results[method]['times']):.6f} seconds")

if __name__ == '__main__':
    compare_bandwidth_methods()