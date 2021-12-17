"""
Generic training script for the vae_vmf
"""
from pathlib import Path

import torch
import numpy as np
from tqdm import tqdm

from vae_motion import VAE_Motion
from data_utils import load_bones_data

filepath = Path(__file__).parent.absolute()
models_path = filepath / "models"
models_path.mkdir(exist_ok=True)


def fit(model: VAE_Motion, data_loader, device, optimizer):
    model.train()
    running_loss = 0.0
    for pos in tqdm(data_loader):
        pos = pos[0]
        pos = pos.to(device)
        # pos = pos.unsqueeze(1)  # old model
        optimizer.zero_grad()
        q_z_given_x, p_x_given_z = model.forward(pos)
        loss = model.elbo_loss(pos, q_z_given_x, p_x_given_z)
        running_loss += loss.item()
        loss.backward()
        optimizer.step()


def validate(model, test_loader, device, test_dataset, epoch=0):
    model.eval()
    running_loss = 0.0
    print(f"Validation (for epoch {epoch}): ")
    with torch.no_grad():
        for pos in tqdm(test_loader):
            pos = pos[0]
            pos = pos.to(device)
            # pos = pos.unsqueeze(1)  # old model
            q_z_given_x, p_x_given_z = model.forward(pos)
            loss = model.elbo_loss(pos, q_z_given_x, p_x_given_z)
            running_loss += loss.item()

    print(f"Loss in validation: {running_loss / len(test_dataset)}")
    return running_loss


def run(
    model,
    train_dataset,
    test_dataset,
    name="vae_motion",
    max_epochs=500,
    batch_size=16,
    lr=1e-3,
    z_dim=2,
):
    # Loading the data.
    # train_dataset, test_dataset = load_walking_data()
    train_data_loader = torch.utils.data.DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=True
    )
    test_data_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    # Trains.
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = {}
    best_loss = np.inf
    n_without_improvement = 0
    for epoch in range(max_epochs):
        print(f"Epoch {epoch+1} of at most {max_epochs}.")
        fit(model, train_data_loader, device, optimizer)
        model_loss = validate(model, test_data_loader, device, test_dataset, epoch=epoch)
        results[epoch] = model_loss
        if model_loss < best_loss:
            # Save the model.
            print("Saving model.")
            torch.save(model.state_dict(), models_path / f"{name}.pt")
            n_without_improvement = 0
            best_loss = model_loss
        else:
            n_without_improvement += 1

        if n_without_improvement == 5:
            print("Stopping early.")
            break


if __name__ == "__main__":
    batch_size = 16
    n_hidden = 30

    # Loading the data.
    bones = torch.tensor([1, 2, 3, 4, 6, 7, 8, 9, 13, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29])
    train_dataset, test_dataset, radii = load_bones_data(bones=bones)
    n_bones = len(bones)

    # Loads up the model.
    model = VAE_Motion(n_bones=n_bones, n_hidden=n_hidden)
    print(model)

    # Trains
    run(model, train_dataset, test_dataset, name="motion")
