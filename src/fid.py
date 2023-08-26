from cleanfid import fid
import torch, os, shutil, torch
import models

from torchvision import datasets, transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader

from tqdm import tqdm, trange

device = "cuda" if torch.cuda.is_available() else "cpu"


GEN_BATCH_SIZE = 11280
FID_BATCH_SIZE = 188


if not fid.test_stats_exists("emnist", mode="clean"):
    dataloader = DataLoader(
        dataset=datasets.EMNIST(
            download=True,
            root="./data",
            split="balanced",
            transform=transforms.Compose(
                [
                    transforms.Lambda(lambda x: transforms.functional.rotate(x, 270)),
                    transforms.Lambda(lambda x: transforms.functional.hflip(x)),
                    transforms.ToTensor(),
                ]
            ),
        ),
        batch_size=1000,
        shuffle=True,
        drop_last=True,
        num_workers=4,
        pin_memory=True,
    )

    img_id = 0
    if not os.path.exists("./temp"):
        os.makedirs("./temp")
    for batch, _ in tqdm(dataloader, desc=f"Processing Dataset", unit="img"):
        for img in batch:
            save_image(img, f"./temp/{img_id}.png")
            img_id += 1

    # precompute FID
    fid.make_custom_stats("emnist", "./temp", mode="clean", batch_size=FID_BATCH_SIZE)

    # Remove folder
    shutil.rmtree("./temp")


scores = []


paths = [
    [str(q.path) for q in sorted(os.scandir(str(p.path)), key=os.path.getmtime)]
    for p in sorted(os.scandir("./checkpoints"), key=os.path.getmtime)
]
ckpt_paths = [p[-1] for p in paths]


for path in tqdm(ckpt_paths, desc="Configs"):
    # generate in temp folder
    generator = models.Generator(z_dim=100, n_classes=47, embed_size=100).to(device)
    generator.load_state_dict(torch.load(path, map_location=device))
    generator.eval()

    if not os.path.exists("./temp"):
        os.makedirs("./temp")

    with torch.no_grad():
        img_id = 0
        for _ in trange(int(112800 / GEN_BATCH_SIZE), desc="Images", leave=False):
            z = torch.randn((GEN_BATCH_SIZE, 100, 1, 1), device=device)
            y = torch.randint(
                size=(GEN_BATCH_SIZE,), low=0, high=47, dtype=torch.int32, device=device
            )
            imgs = generator(z, y)
            del z, y

            for img in tqdm(imgs, total=len(imgs), desc="Saving", leave=False):
                save_image(img, f"./temp/{img_id}.png")
                img_id += 1
            del imgs

    # calculate fid from precompute
    score = fid.compute_fid(
        "./temp",
        dataset_name="emnist",
        mode="clean",
        dataset_split="custom",
        num_workers=8,
        batch_size=FID_BATCH_SIZE,
        device=device,
    )

    scores.append(score)
    print(score)

    shutil.rmtree("./temp")


print("SCORES:")
for idx, score in enumerate(scores):
    print(idx, score)
