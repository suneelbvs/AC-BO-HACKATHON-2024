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

    plt.clf()

    # if results file exists, backlog it and give it a new name
    if Path("out/results.csv").exists():
        # look for all backlog files
        backlog_files = list(Path("out").glob("results_backlog*.csv"))
        # get the highest number in the backlog files
        highest_backlog_number = max([int(file.stem.split("_")[-1]) for file in backlog_files] or [0])
        # rename the current results file to a new backlog file
        Path("out/results.csv").rename(f"out/results_backlog_{highest_backlog_number + 1}.csv")

    with open("out/results.csv", 'w') as f:
        for name, results in result_tracker.results.items():
            f.write(f"{name}, {','.join([str(result.y_axis) for result in results])}\n")