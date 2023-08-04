import torch


def compute_latent(model, loader, args):
    mu_set = []
    var_set = []
    latent_set = []
    with torch.no_grad():
        for x in loader:
            # Send to device
            x = x[0].to(args.device, non_blocking=True)
            # Encode into model
            latent, mu, var = model.encode(x)
            latent_set.append(latent.detach())
            if args.model != 'ae':
                mu_set.append(mu.detach())
                var_set.append(var.detach())
    # Concatenate into vector
    final_latent = torch.cat(latent_set, dim=0)
    if len(mu_set) > 1:
        mu_set = torch.cat(mu_set, dim=0)
        var_set = torch.cat(var_set, dim=0)
    return final_latent, mu_set, var_set
