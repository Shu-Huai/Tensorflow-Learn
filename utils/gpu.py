import tensorflow as tf


def check_device() -> None:
    """
    是否能用Gpu
    :return: 无
    """
    print(f"设备：{str(tf.config.list_physical_devices('GPU'))}")
