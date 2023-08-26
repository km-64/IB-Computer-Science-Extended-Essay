import train
from tqdm import tqdm

EPOCHS = 15
DATASET_SIZES = [int((i + 1) * 11280) for i in range(10)]

for size in tqdm(
    DATASET_SIZES, total=len(DATASET_SIZES), desc=f"Configurations", unit="configs"
):
    train.training(size, EPOCHS, "./checkpoints")
