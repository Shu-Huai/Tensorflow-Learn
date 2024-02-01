import matplotlib.pyplot as plt
from numpy import ndarray


class Param:
    def __init__(self, value: ndarray, label: str = "", marker: str = "", line_style: str = "-"):
        self.value = value
        self.label = label
        self.marker = marker
        self.line_style = line_style


def plot(save: bool = False, path: str = None, height: int = None, width: int = None,
         **kwargs: dict[str:any]) -> None:
    if height is not None:
        plt.rcParams['figure.figsize'] = (plt.rcParams['figure.figsize'][0], height)
    if width is not None:
        plt.rcParams['figure.figsize'] = (width, plt.rcParams['figure.figsize'][1])
    for key, value in kwargs.items():
        if isinstance(value, Param):
            plt.plot(value.value, label=value.label, marker=value.marker,
                     linestyle=value.line_style)
        else:
            plt.plot(value, label=str(key))
    plt.legend()
    if save:
        plt.savefig(path)
    plt.show()
