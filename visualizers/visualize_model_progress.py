import matplotlib.pyplot as plt
from results import Result, get_classification_results, get_regression_results_highest_y, get_regression_results_num_better_candidates, get_regression_results_num_over_200
from typing import Callable, Dict
from pathlib import Path

def get_ylabel(result_creator: Callable[..., Result]) -> str:
    if result_creator == get_classification_results:
        return 'Num Hits'
    elif result_creator == get_regression_results_num_better_candidates:
        return 'Num Better Candidates'
    elif result_creator == get_regression_results_highest_y:
        return 'Best Actual Y'
    elif result_creator == get_regression_results_num_over_200:
        return 'Num Over 200'
    else:
        raise ValueError("Unknown result creator")

def visualize_hits(optimization_results: Dict[str, Result], result_creator: Callable[..., Result], loader_name: str, model_name:str):
    fig, ax = plt.subplots()

    # Plot the number of hits for each acquisition function
    for name, results in optimization_results.items():
        x = [result.batch_number for result in results]
        y = [result.y_axis for result in results]
        ax.plot(x, y, label=name)

    # Set the labels for x-axis and y-axis
    ax.set_xlabel('Batch Number')
    ax.set_ylabel(get_ylabel(result_creator))
    ax.set_xticks(x)
    ax.set_xticklabels(x)

    title = f"{loader_name} dataset using {model_name}"
    ax.set_title(title)
    ax.legend()

    Path("out").mkdir(parents=True, exist_ok=True)
    plt.savefig(f"out/{title.replace(' ', '_')}.png")

    # Display the graph
    # plt.show()
