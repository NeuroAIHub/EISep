import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

def plot_subplot(ax, results_path, metric, title, y_label, is_log_y):

    try:
        with open(results_path, 'rb') as f:
            results = pickle.load(f)
        print(f"Successfully loaded results from {results_path}")
    except FileNotFoundError:
        print(f"Error: The file {results_path} was not found.")
        ax.text(0.5, 0.5, f'Data not found:\n{results_path}', ha='center', va='center')
        return None, None

    df = pd.DataFrame(results)
    if df.empty:
        print(f"The results file {results_path} is empty.")
        ax.text(0.5, 0.5, 'No data to visualize', ha='center', va='center')
        return None, None

    
    model_order = ['eisep', 'simple', 'eisep_ei_input', 'eisep_ei_output', 'eisep_ei_both', 'simple_ei_input', 'simple_ei_output', 'simple_ei_both']
    legend_map = {
        'eisep': 'EISep',
        'unconstrained_simple': 'Simple',
        'simple': 'Simple',
        'dann_sgd': 'DANN-SGD',
        'dann_adam': 'DANN-Adam',
        'colei_sgd': 'ColEI-SGD',
        'colei_adam': 'ColEI-Adam',
        'dbpn': 'DBPN',
        'entryei': 'EntryEI',
        'fixed_eisep': 'Fixed-EISep',
        'absolute_eisep': 'Abs-EISep',
        'eisep_ei_input': 'EISep-EI-Input',
        'eisep_ei_output': 'EISep-EI-Output',
        'eisep_ei_both': 'EISep-EI-Both',
        'simple_ei_input': 'Simple-EI-Input',
        'simple_ei_output': 'Simple-EI-Output',
        'simple_ei_both': 'Simple-EI-Both'
    }
    
    df = df[df['model_name'].isin(legend_map.keys())]
    df['model_name'] = df['model_name'].map(legend_map)
    
    new_order = [legend_map[m] for m in model_order if m in legend_map]

    df['model_name'] = pd.Categorical(df['model_name'], categories=new_order, ordered=True)

    df = df.sort_values('model_name')


    color_palette = {
        'EISep': 'C1',
        'DANN-SGD': 'C2',
        'DANN-Adam': 'C3',
        'ColEI-SGD': 'C4',
        'ColEI-Adam': 'C5',
        'Simple': 'C0',
        'DBPN': 'C6',
        'EntryEI': 'C9',
        'Fixed-EISep': 'C8',
        'Abs-EISep': 'C7',
        'EISep-EI-Input': '#bcbd22',
        'EISep-EI-Output': '#dc6060',
        'EISep-EI-Both': '#7f0000',
        'Simple-EI-Input': '#17becf',
        'Simple-EI-Output': "#9467bd",
        'Simple-EI-Both': '#003f5c'
    }

    eisep_df = df[df['model_name'] == 'EISep'].copy()
    other_df = df[df['model_name'] != 'EISep'].copy()
    

    sns.lineplot(
        data=other_df,
        x='lr',
        y=metric,
        hue='model_name',
        marker='o',
        dashes=False,
        ax=ax,
        palette=color_palette,
        errorbar='sd',
        markersize=9,
        markeredgecolor='white',
        markeredgewidth=1,
        linewidth=1.8,
        alpha=0.9
    )

    if not eisep_df.empty:
        sns.lineplot(
            data=eisep_df,
            x='lr',
            y=metric,
            hue='model_name',
            marker='o',
            dashes=False,
            ax=ax,
            palette=color_palette,
            errorbar='sd',
            markersize=9,
            markeredgecolor='white',
            markeredgewidth=1,
            linewidth=1.8,
            alpha=0.9
        )


    for collection in ax.collections:
        collection.set_alpha(0.1)

    ax.set_title(title, fontsize=20, pad=15)
    ax.set_xlabel('Learning Rate', fontsize=17)
    ax.set_ylabel(y_label, fontsize=14)
    ax.set_xscale('log')
    if is_log_y:
        ax.set_yscale('log')
    ax.tick_params(axis='both', which='major', labelsize=12)
    
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_edgecolor('black')
    
    handles, labels = ax.get_legend_handles_labels()
    ax.get_legend().remove() 
    return handles, labels

def visualize_combined_results():
    
    plt.rcParams.update({
        'font.family': 'Arial'
    })

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    handles, labels = plot_subplot(
        axes[0],
        'results/adding_problem1/benchmark_results1.pkl',
        'test_mse',
        'Adding Problem',
        'Test MSE (lower is better)',
        is_log_y=True
    )

    plot_subplot(
        axes[1],
        'results/sequential_mnist/benchmark_results.pkl',
        'test_accuracy',
        'Sequential MNIST',
        'Test Accuracy (higher is better)',
        is_log_y=False
    )

    plot_subplot(
        axes[2],
        'results/sort_problem/benchmark_results.pkl',
        'test_accuracy',
        'Sorting Problem',
        'Test Accuracy (higher is better)',
        is_log_y=False
    )

    if handles and labels:
        
        legend_dict = dict(zip(labels, handles))
        fig.legend(
            legend_dict.values(),
            legend_dict.keys(),
            loc='lower center',
            bbox_to_anchor=(0.5, -0.05),
            ncol=len(legend_dict),
            fontsize=15
            # title='Model',
            # title_fontsize=18
        )

    plt.tight_layout(rect=[0, 0.05, 1, 1])

    output_dir = 'results'
    os.makedirs(output_dir, exist_ok=True)
    fig_path = os.path.join(output_dir, 'combined_benchmark_performance_split_linear.png')
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"\nCombined visualization saved to {fig_path}")
    plt.show()


if __name__ == '__main__':
    visualize_combined_results()
