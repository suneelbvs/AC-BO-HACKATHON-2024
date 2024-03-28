import matplotlib.pyplot as plt
from results import Result, get_classification_results, get_regression_results_highest_y, get_regression_results_num_better_candidates
from typing import Callable, Dict
from pathlib import Path

def get_ylabel(result_creator: Callable[..., Result]) -> str:
    if result_creator == get_classification_results:
        return 'Num Hits'
    elif result_creator == get_regression_results_num_better_candidates:
        return 'Num Better Candidates'
    elif result_creator == get_regression_results_highest_y:
        return 'Best Actual Y'
    else:
        raise ValueError("Unknown result creator")

def visualize_hits(optimization_results: Dict[str, Result], result_creator: Callable[..., Result]):
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

    ax.set_title('Num Hits')
    ax.legend()

    Path("out").mkdir(parents=True, exist_ok=True)
    plt.savefig('out/hit_graph.png')

    # Display the graph
    # plt.show()
