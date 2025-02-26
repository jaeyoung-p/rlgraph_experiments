import matplotlib.pyplot as plt
from task4feedback.fastsim.interface import SimulatorHandler
from task4feedback.types import TaskID, TaskInfo
import networkx as nx
import matplotlib.pyplot as plt
import pydot


def save_boxplot(
    data: dict[str, list[float]],
    x_label: str,
    y_label: str,
    font_size: int,
    image_name: str,
) -> None:
    """
    Creates and saves a box-and-whisker plot.

    Parameters:
        data (dict[str, list[float]]): Dictionary where keys are the categories for the x-axis and
                                       values are the numerical data for the box plots.
        x_label (str): Label for the x-axis.
        y_label (str): Label for the y-axis.
        font_size (int): Font size for labels and tick marks.
        image_name (str): File name for saving the image (e.g., 'plot.png').
    """
    # Extract the keys (categories) and corresponding data lists
    categories = list(data.keys())
    data_values = list(data.values())

    # Create a new figure and axis object
    fig, ax = plt.subplots(figsize=(10, 6))

    # Create the boxplot with category labels on the x-axis
    ax.boxplot(data_values, labels=categories)

    # Set the x and y labels with the provided font size
    ax.set_xlabel(x_label, fontsize=font_size)
    ax.set_ylabel(y_label, fontsize=font_size)

    # Set tick parameters to control the font size of the tick labels
    ax.tick_params(axis="both", which="major", labelsize=font_size)

    # Save the figure to a file
    plt.savefig(image_name, bbox_inches="tight")

    # Close the plot to free up memory
    plt.close(fig)


def draw_graph(
    SimulatorHandler: SimulatorHandler,
):
    tasks = SimulatorHandler.tasks
    graph = pydot.Dot(graph_type="digraph")
    for name, task_info in tasks.items():
        node = pydot.Node(
            name=str(name),
            style="filled",
            fillcolor="white",
        )
        graph.add_node(node)
        for dep_id in task_info.dependencies:
            edge = pydot.Edge(str(dep_id), str(name))
            graph.add_edge(edge)
    graph.write_png("graph.png")
