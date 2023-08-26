import torch, torchvision, os
from models import Critic, Generator

from torch.utils.data import DataLoader
from torchvision import transforms, datasets

from tqdm.auto import tqdm, trange
from torch.utils.tensorboard import SummaryWriter

torch.backends.cudnn.benchmark = True
NUM_WORKERS = 8
device = "cuda" if torch.cuda.is_available() else "cpu"
print("USING: " + device)


# HYPER PARAMETERS
BATCH_SIZE = 64
BETAS = (0.0, 0.9)
LR = 1e-4

Z_DIM = 100
EMBED_SIZE = 100
LAMBDA = 10
N_CRITIC = 5

NUM_CLASSES = 47


def training(dataset_size: int, epochs: int, checkpoint_dir: str):

    writer = SummaryWriter(f"logs/runs/emnist_{dataset_size}")

    fixed_noise = torch.randn((NUM_CLASSES**2, Z_DIM, 1, 1), device=device)
    fixed_labels = torch.zeros(
        (NUM_CLASSES, NUM_CLASSES), dtype=torch.int32, device=device
    )
    for y in range(fixed_labels.shape[0]):
        for x in range(fixed_labels.shape[1]):
            fixed_labels[y][x] = y
    fixed_labels = fixed_labels.view(-1)

    dataset = datasets.EMNIST(
        download=True,
        root="./data",
        split="balanced",
        transform=transforms.Compose(
            [
                transforms.Lambda(lambda x: transforms.functional.rotate(x, 270)),
                transforms.Lambda(lambda x: transforms.functional.hflip(x)),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
            ]
        ),
    )

    dataset.data = dataset.data[:dataset_size]
    dataset.targets = dataset.targets[:dataset_size]

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    G = Generator(z_dim=Z_DIM, embed_size=EMBED_SIZE, n_classes=NUM_CLASSES).to(device)
    C = Critic(n_classes=NUM_CLASSES).to(device)

    G_opt = torch.optim.Adam(G.parameters(), lr=LR, betas=BETAS)
    C_opt = torch.optim.Adam(C.parameters(), lr=LR, betas=BETAS)

    step = 0
    for epoch in range(epochs):
        loop = tqdm(
            enumerate(dataloader),
            desc=f"Epoch [{epoch+1}/{epochs}]",
            total=len(dataloader),
            unit="batch",
            leave=False,
        )
        for idx, (x, y) in loop:
            x = x.to(device)
            y = y.to(device)

            for _ in range(N_CRITIC):
                z = torch.randn((BATCH_SIZE, Z_DIM, 1, 1), device=device)
                x_ = G(z, y)
                C_x = C(x, y).view(-1)
                C_x_ = C(x_, y).view(-1)

                eps = torch.randn((BATCH_SIZE, 1, 1, 1), device=device).repeat(
                    1, 1, 28, 28
                )
                x_int = eps * x + (1 - eps) * x_

                C_x_int = C(x_int, y)
                gradient = torch.autograd.grad(
                    inputs=x_int,
                    outputs=C_x_int,
                    grad_outputs=torch.ones_like(C_x_int),
                    create_graph=True,
                    retain_graph=True,
                )[0]
                gradient = gradient.view(gradient.shape[0], -1)
                norm = gradient.norm(2, dim=1)
                gp = torch.mean((norm - 1) ** 2) * LAMBDA

                w_dist = torch.mean(C_x_) - torch.mean(C_x)
                C_loss = w_dist + gp

                C_opt.zero_grad()
                C_loss.backward(retain_graph=True)
                C_opt.step()

            C_G_z = C(x_, y).view(-1)
            G_loss = -torch.mean(C_G_z)

            G_opt.zero_grad()
            G_loss.backward()
            G_opt.step()

            if step % 10 == 0:
                loop.set_postfix(
                    C_l=f"{C_loss.item():.4f}",
                    G_l=f"{G_loss.item():.4f}",
                    W_l=f"{w_dist.item():.4f}",
                )
                writer.add_scalar("Critic Loss", C_loss.item(), global_step=step)
                writer.add_scalar("Generator Loss", G_loss.item(), global_step=step)
                writer.add_scalar(
                    "Wasserstein Distance Estimate", w_dist.item(), global_step=step
                )

            if step % 250 == 0:
                with torch.no_grad():
                    img = G(fixed_noise, fixed_labels)
                    grid = torchvision.utils.make_grid(img, nrow=NUM_CLASSES)
                    writer.add_image("Generated Images", grid, global_step=step)

            step += 1

        path = os.path.join(checkpoint_dir, f"emnist_{dataset_size}")
        if not os.path.exists(path):
            os.makedirs(path)
        torch.save(G.state_dict(), os.path.join(path, f"epoch_{epoch}.pth"))
