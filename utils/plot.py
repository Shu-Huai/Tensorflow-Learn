import matplotlib.pyplot as plt


def plot(save: bool = False, path: str = None, **kwargs: dict[str:any]) -> None:
    for key, value in kwargs.items():
        plt.plot(value, label=str(key))
    plt.legend()
    if save:
        plt.savefig(path)
    plt.show()
