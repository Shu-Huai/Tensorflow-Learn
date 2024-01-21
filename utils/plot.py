import matplotlib.pyplot as plt


def plot(save: bool = False, path: str = None, height: int = None, width: int = None,
         **kwargs: dict[str:any]) -> None:
    if height is not None:
        plt.rcParams['figure.figsize'] = (plt.rcParams['figure.figsize'][0], height)
    if width is not None:
        plt.rcParams['figure.figsize'] = (width, plt.rcParams['figure.figsize'][1])
    for key, value in kwargs.items():
        plt.plot(value, label=str(key))
    plt.legend()
    if save:
        plt.savefig(path)
    plt.show()
