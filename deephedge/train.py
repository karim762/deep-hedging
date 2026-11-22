"""Training loop shared by all experiments: Adam with a step learning-rate schedule.

Seeding is left to the caller: call torch.manual_seed before building the model, so that the seed
fixes both the initial weights and the order of the mini-batches.
"""
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


def fit(model, tensors, loss_fn, n_epochs=30, batch_size=1000, lr=1e-3, milestones=(18, 25),
        desc=None, progress=True):
    """Minimise loss_fn(model, *batch) over mini-batches of `tensors`. Returns the model in eval
    mode and the average training loss of each epoch."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=list(milestones), gamma=0.1)
    loader = DataLoader(TensorDataset(*tensors), batch_size=batch_size, shuffle=True)
    history = []
    for _ in tqdm(range(n_epochs), desc=desc, disable=not progress):
        total = 0.0
        for batch in loader:
            loss = loss_fn(model, *batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total += loss.item() * len(batch[0])
        scheduler.step()
        history.append(total / len(loader.dataset))
    return model.eval(), history
