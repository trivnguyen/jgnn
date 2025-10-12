
import os
import torch
from tqdm import tqdm

@torch.no_grad()
def sample(model, data_loader, num_posteriors=100, transforms=None, norm_dict=None, to_numpy=True):

    model.eval()
    device = model.device
    posteriors = []
    truths = []
    for i, batch in tqdm(enumerate(data_loader)):
        if transforms is not None:
            batch = transforms(batch)
        batch = batch.to(device)
        cond = batch.cond if hasattr(batch, 'cond') else None

        flow_context = model(
            batch.x, batch.edge_index, batch=batch.batch,
            edge_attr=None, edge_weight=None, cond=cond
        )
        posterior = model.flows(flow_context).sample(
            (num_posteriors, ))
        posterior = posterior.transpose(0, 1).cpu() #(batch, num_posteriors, num_features)
        posteriors.append(posterior)
        truths.append(batch.theta.cpu())
    posteriors = torch.cat(posteriors, axis=0)
    truths = torch.cat(truths, axis=0)

    if to_numpy:
        posteriors = posteriors.numpy()
        truths = truths.numpy()

    if norm_dict is not None:
        posteriors = posteriors * norm_dict['theta_scale'] + norm_dict['theta_loc']
        truths = truths * norm_dict['theta_scale'] + norm_dict['theta_loc']

    return posteriors, truths