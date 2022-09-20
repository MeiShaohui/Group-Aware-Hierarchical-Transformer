import math
import torch
import torch.nn as nn


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        """
        input: (B, N, C)
        B = Batch size, N = patch_size * patch_size, C = dimension hidden_features and out_features
        output: (B, N, C)
        """
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=16, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads

        self.q = nn.Linear(dim, dim, bias=True)
        self.kv = nn.Linear(dim, dim * 2, bias=True)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        """
        input: (B, N, C)
        B = Batch size, N = patch_size * patch_size, C = dimension for attention
        output: (B, N, C)
        """
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1))
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class GroupedPixelEmbedding(nn.Module):
    def __init__(self, in_feature_map_size=7, in_chans=3, embed_dim=128, n_groups=1):
        super().__init__()
        self.ifm_size = in_feature_map_size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=3, stride=1, padding=1, groups=n_groups)
        self.batch_norm = nn.BatchNorm2d(embed_dim)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        """
        input: (B, in_chans, in_feature_map_size, in_feature_map_size)
        output: (B, (after_feature_map_size x after_feature_map_size-2), embed_dim = C)
        """
        x = self.proj(x)
        x = self.relu(self.batch_norm(x))
        
        x = x.flatten(2).transpose(1, 2)
        
        after_feature_map_size = self.ifm_size  
        
        return x, after_feature_map_size


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4, drop=0., attn_drop=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads=num_heads, attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class MyTransformer(nn.Module):
    def __init__(self, img_size=224, in_chans=3, num_classes=1000, num_stages=3, 
                n_groups=[32, 32, 32], embed_dims=[256, 128, 64], num_heads=[8, 4, 2], mlp_ratios=[1, 1, 1], depths=[2, 2, 2]):
        super().__init__()

        self.num_stages = num_stages
        
        new_bands = math.ceil(in_chans / n_groups[0]) * n_groups[0]
        self.pad = nn.ReplicationPad3d((0, 0, 0, 0, 0, new_bands - in_chans))

        for i in range(num_stages):
            patch_embed = GroupedPixelEmbedding(
                in_feature_map_size=img_size,
                in_chans=new_bands if i == 0 else embed_dims[i - 1],
                embed_dim=embed_dims[i],
                n_groups=n_groups[i]
            )

            block = nn.ModuleList([Block(
                dim=embed_dims[i], 
                num_heads=num_heads[i],
                mlp_ratio=mlp_ratios[i], 
                drop=0., 
                attn_drop=0.) for j in range(depths[i])])
            
            norm = nn.LayerNorm(embed_dims[i])

            setattr(self, f"patch_embed{i + 1}", patch_embed)
            setattr(self, f"block{i + 1}", block)
            setattr(self, f"norm{i + 1}", norm)
        
        self.head = nn.Linear(embed_dims[-1], num_classes)  # 只有pvt时的Head

    def forward_features(self, x):
        # (bs, 1, n_bands, patch size (ps, of HSI), ps)
        x = self.pad(x).squeeze(dim=1)
        B = x.shape[0]

        for i in range(self.num_stages):
            patch_embed = getattr(self, f"patch_embed{i + 1}")
            block = getattr(self, f"block{i + 1}")
            norm = getattr(self, f"norm{i + 1}")
            
            x, s = patch_embed(x)  # s = feature map size after patch embedding
            for blk in block:
                x = blk(x)
            
            x = norm(x)
            
            if i != self.num_stages - 1: 
                x = x.reshape(B, s, s, -1).permute(0, 3, 1, 2).contiguous()
        
        return x
    
    def forward(self, x):
        x = self.forward_features(x)
        x = x.mean(dim=1)
        x = self.head(x)
        return x


def proposed(dataset, patch_size):
    model = None
    if dataset == 'sa':
        model = MyTransformer(img_size=patch_size, in_chans=204, num_classes=16, n_groups=[16, 16, 16], depths=[2, 1, 1])
    elif dataset == 'pu':
        model = MyTransformer(img_size=patch_size, in_chans=103, num_classes=9, n_groups=[2, 2, 2], depths=[1, 2, 1])
    elif dataset == 'whulk':
        model = MyTransformer(img_size=patch_size, in_chans=270, num_classes=9, n_groups=[2, 2, 2], depths=[2, 2, 1])
    elif dataset == 'hrl':
        model = MyTransformer(img_size=patch_size, in_chans=176, num_classes=14, n_groups=[4, 4, 4], depths=[1, 2, 1])
    return model

if __name__ == "__main__":
    t = torch.randn(size=(3, 1, 204, 7, 7))
    print("input shape:", t.shape)
    net = proposed(dataset='sa', patch_size=7)
    print("output shape:", net(t).shape)

