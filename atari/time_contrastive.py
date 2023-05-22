import torch

from einops import rearrange, einsum


def sample_contrastive_batch(obs, p, batch_size=16):
    T, N, H, W = obs.shape
    i_step, i_env = torch.randint(0, T - 1, size=(batch_size,), device=obs.device), torch.randint(0, N, size=(batch_size,), device=obs.device)
    obs_anc = obs[i_step, i_env]

    dt_max = T - i_step
    dist = torch.distributions.Geometric(probs=torch.tensor(p, device=obs.device))
    dt = dist.sample((batch_size,)).long() + 1
    i_step = i_step + (dt % dt_max)  # modulo ensures geometric distribution within bounds
    obs_pos = obs[i_step, i_env]

    i_step, i_env = torch.randint(0, T, size=(batch_size,), device=obs.device), torch.randint(0, N, size=(batch_size,), device=obs.device)
    obs_neg = obs[i_step, i_env]

    return obs_anc, obs_pos, obs_neg


def calc_contrastive_loss(encoder, obs_anc, obs_pos, obs_neg):
    latent_anc = encoder.encode(rearrange(obs_anc, "bs h w -> bs 1 h w"))
    latent_pos = encoder.encode(rearrange(obs_pos, "bs h w -> bs 1 h w"))
    latent_neg = encoder.encode(rearrange(obs_neg, "bs h w -> bs 1 h w"))

    # pos = torch.cosine_similarity(latent_anchor, latent_positive, dim=-1)
    # neg = torch.cosine_similarity(latent_anchor, latent_negative, dim=-1)
    pos = einsum(latent_anc, latent_pos, "bs d, bs d -> bs")
    neg = einsum(latent_anc, latent_neg, "bs d, bs d -> bs")

    loss_pos = (pos.sigmoid()).log()
    loss_neg = (1.0 - neg.sigmoid()).log()
    loss = (loss_pos + loss_neg).mean()
    return -loss


if __name__ == "__main__":
    import numpy as np
    from env_atari import make_env
    from agent_atari import Encoder
    from tqdm.auto import tqdm

    env = make_env("MontezumaRevenge", n_envs=64)
    obsi, info = env.reset()
    obs = [obsi[:, -1]]

    for i in tqdm(range(1024)):
        obsi, rew, term, trunc, info = env.step(env.action_space.sample())
        obs.append(obsi[:, -1])

    obs = torch.as_tensor(np.stack(obs))

    n_steps = 100
    batch_size = 256
    lr = 3e-4
    device = "mps"

    encoder = Encoder((1, 84, 84), 64).to(device)
    opt = torch.optim.Adam(encoder.parameters(), lr=lr)

    losses = []
    for i_step in tqdm(range(n_steps)):
        obs_anc, obs_pos, obs_neg = sample_contrastive_batch(obs, p=0.1, batch_size=batch_size)
        obs_anc, obs_pos, obs_neg = obs_anc.to(device), obs_pos.to(device), obs_neg.to(device)

        loss = calc_contrastive_loss(encoder, obs_anc, obs_pos, obs_neg)

        opt.zero_grad()
        loss.backward()
        opt.step()

        print(loss.item())
        losses.append(loss.item())

    import matplotlib.pyplot as plt

    # plt.plot(losses)
    # plt.show()

    i_env = torch.randint(0, 8, (1024,))
    i_step = torch.randint(0, 128, (1024,))
    o = obs[i_step, i_env].to(device)

    latent = encoder.encode(o[:, None])
    X = latent.cpu().detach().numpy()
    import sklearn.manifold

    X2D = sklearn.manifold.TSNE().fit_transform(X)

    plt.figure(figsize=(20, 4))
    for i in range(5):
        plt.subplot(2, 5, i + 1)
        plt.imshow(o[i].cpu().detach().numpy())
        plt.subplot(2, 5, i + 1 + 5)
        # plt.scatter(X2D[[i], 0], X2D[[i], 1])
        plt.scatter(*X2D.T, s=1.0)
        plt.scatter(*X2D[[i], :].T, s=100.0, c="r", marker="x")
        # plt.xlim(-3, 3); plt.ylim(-3, 3)
    plt.tight_layout()
    plt.show()
