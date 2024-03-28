import matplotlib.pyplot as plt
from results import ResultTracker
from pathlib import Path

def visualize_results(result_tracker: ResultTracker, loader_name: str, model_name:str):
    fig, ax = plt.subplots()

    # Plot the number of hits for each acquisition function
    for name, results in result_tracker.results.items():
        x = [result.batch_number for result in results]
        y = [result.y_axis for result in results]
        ax.plot(x, y, label=name)

    # Set the labels for x-axis and y-axis
    ax.set_xlabel('Batch Number')
    ax.set_ylabel(result_tracker.y_label)
    ax.set_xticks(x)
    ax.set_xticklabels(x)

    title = f"{loader_name} dataset using {model_name}"
    ax.set_title(title)
    ax.legend()

    Path("out").mkdir(parents=True, exist_ok=True)
    plt.savefig(f"out/{title.replace(' ', '_')}.png")

    # Display the graph
    # plt.show()
