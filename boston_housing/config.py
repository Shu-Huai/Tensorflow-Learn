import os

batch_size: int = 1
epochs: int = 100
plot_dpi: int = 600
data_path: str = os.path.join("datasets", "boston_housing.npz")
plot_path: str = r"plots"
model_path: str = r".\models\boston_housing.h5"