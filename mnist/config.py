import os

batch_size: int = 60
epochs: int = 20
plot_dpi: int = 600
data_path: str = os.path.join("datasets", "mnist.npz")
plot_path: str = r"plots"
model_path: str = r".\models\mnist.h5"
draw_path: str = r".\my_draws"
