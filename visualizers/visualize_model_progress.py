import matplotlib.pyplot as plt
from results import Result

def visualize_hits(results: [Result]):
    x = [result.batch_number for result in results]
    y = [result.num_hits for result in results]

    # Create a new figure and axis
    fig, ax = plt.subplots()

    # Plot the line graph
    ax.plot(x, y)

    # Set the labels for x-axis and y-axis
    ax.set_xlabel('Batch Number')
    ax.set_ylabel('Num Hits')
    ax.set_xticks(x)
    ax.set_xticklabels(x)

    # Set the title of the graph
    ax.set_title('Num Hits')

    # Save the figure as a PNG file
    plt.savefig('out/hit_graph.png')

    # Display the graph
    # plt.show()
