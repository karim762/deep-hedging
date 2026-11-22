"""Training loop shared by all experiments: Adam with a step learning-rate schedule.

Seeding is left to the caller: call torch.manual_seed before building the model, so that the seed
fixes both the initial weights and the order of the mini-batches.
"""
import copy

import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


def fit(model, tensors, loss_fn, n_epochs=30, batch_size=1000, lr=1e-3, milestones=(18, 25),
        desc=None, progress=True, validate=None, every=10):
    """Minimise loss_fn(model, *batch) over mini-batches of `tensors`. Returns the model in eval
    mode and the average training loss of each epoch.

    If `validate` is given (a function of the model returning a score, lower is better), it is called
    every `every` epochs and the best checkpoint is restored at the end.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=list(milestones), gamma=0.1)
    loader = DataLoader(TensorDataset(*tensors), batch_size=batch_size, shuffle=True)
    history = []
    best_score, best_state = float("inf"), None
    for epoch in tqdm(range(n_epochs), desc=desc, disable=not progress):
        total = 0.0
        for batch in loader:
            loss = loss_fn(model, *batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total += loss.item() * len(batch[0])
        scheduler.step()
        history.append(total / len(loader.dataset))
        if validate is not None and (epoch + 1) % every == 0:
            model.eval()
            with torch.no_grad():
                score = validate(model)
            model.train()
            if score < best_score:
                best_score, best_state = score, copy.deepcopy(model.state_dict())
    if validate is not None:
        model.load_state_dict(best_state)
    return model.eval(), history
