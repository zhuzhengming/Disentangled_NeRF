import torch
import torch.nn as nn

# position encoder
def Position_encoder(x, degree):
    y = torch.cat([2.**i * x for i in range(degree)], -1)
    w = 1
    return torch.cat([x] + [torch.sin(y) * w, torch.cos(y) * w], -1)

# conditional NeRF
class ConNeRF(nn.Module):
    def __init__(self, Zs_blocks=2, Zt_blocks=1, dim=256,
                 xyz_freq=10, dir_freq=4, latent_dim=256):
        super().__init__()
        self.Zs_blocks = Zs_blocks
        self.Zt_blocks = Zt_blocks
        self.xyz_freq = xyz_freq
        self.dir_freq = dir_freq
        self.latent_dim = latent_dim

        # dimension: sin(nx) + cos(nx) + x
        d_xyz = 3 + 3 * self.xyz_freq + 3 * self.xyz_freq
        d_dir = 3 + 6 * self.dir_freq + 3 * self.dir_freq

        self.encoding_xyz = nn.Sequential(nn.Linear(d_xyz, dim), nn.ReLU())

        for i in range(self.Zs_blocks):
            layer = nn.Sequential(nn.Linear(self.latent_dim, dim), nn.ReLU())
            setattr(self, f"shape_latent_layer_{i+1}", layer)

            layer = nn.Sequential(nn.Linear(dim, dim), nn.ReLU())
            setattr(self, f"shape_layer_{i+1}", layer)

        self.encoding_shape = nn.Linear(dim, dim)  # no activate function
        self.sigma = nn.Sequential(nn.Linear(dim, 1), nn.Softplus())
        self.encoding_dir = nn.Sequential(nn.Linear(dim+d_dir, dim), nn.ReLU())
        for j in range(self.Zs_blocks):
            layer = nn.Sequential(nn.Linear(latent_dim, dim), nn.ReLU())
            setattr(self, f"texture_latent_layer_{j+1}", layer)
            layer = nn.Sequential(nn.Linear(dim, dim), nn.ReLU())
            setattr(self, f"texture_layer_{j + 1}", layer)
        self.rgb = nn.Sequential(nn.Linear(dim, dim//2), nn.ReLU(), nn.Linear(dim//2, 3))

    def forward(self, xyz, dir, Zs, Zt):
        # position encoding
        xyz = Position_encoder(xyz, self.xyz_freq)
        dir = Position_encoder(dir, self.dir_freq)

        # pipeline
        y = self.encoding_xyz(xyz)

        for i in range(self.Zs_blocks):
            z = getattr(self, f"shape_latent_layer_{i+1}")(Zs)
            y = y + z
            y = getattr(self, f"shape_layer_{i+1}")(y)
        sigma = self.sigma(y)

        y = torch.cat([self.encoding_shape(y), dir], -1)
        y = self.encoding_dir(y)

        for j in range(self.Zt_blocks):
            z = getattr(self, f"texture_latent_layer_{j+1}")(Zt)
            y = y + z
            y = getattr(self, f"texture_layer_{j+1}")(y)

        rgb = self.rgb(y)
        return sigma, rgb
