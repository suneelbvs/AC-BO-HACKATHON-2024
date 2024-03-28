import matplotlib.pyplot as plt
from results import Result
from typing import Dict
from pathlib import Path

def visualize_hits(optimization_results: Dict[str, Result]):
    fig, ax = plt.subplots()

    # Plot the number of hits for each acquisition function
    for name, results in optimization_results.items():
        x = [result.batch_number for result in results]
        y = [result.num_hits for result in results]
        ax.plot(x, y, label=name)

    # Set the labels for x-axis and y-axis
    ax.set_xlabel('Batch Number')
    ax.set_ylabel('Num Hits')
    ax.set_xticks(x)
    ax.set_xticklabels(x)

    ax.set_title('Num Hits')
    ax.legend()

    Path("out").mkdir(parents=True, exist_ok=True)
    plt.savefig('out/hit_graph.png')

    # Display the graph
    # plt.show()
