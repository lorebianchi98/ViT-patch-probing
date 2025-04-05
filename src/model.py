import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import timm

class LinearProber(nn.Module):
    def __init__(
            self, model_name, n_classes, resize_dim=518):
        super().__init__()
        self.model_name = model_name
        # loading the model
        
        if 'dinov2' in model_name:
            self.model_family = 'facebookresearch/dinov2' if 'dinov2' in model_name else 'facebookresearch/dino:main'
            self.visual_backbone = torch.hub.load(self.model_family, model_name)
            
            
        elif 'mae' in model_name or 'clip' in model_name or 'dino' in model_name:
            self.visual_backbone = timm.create_model(
                model_name,
                pretrained=True,
                num_classes=0,  # remove classifier nn.Linear
                img_size=resize_dim
            )

        else:
            raise Exception("Unknown ViT model")
        # self.model.eval()
        mean = (0.485, 0.456, 0.406) if not 'clip' in model_name else (0.4815, 0.4578, 0.4082)
        std = (0.229, 0.224, 0.225) if not 'clip' in model_name else (0.2686, 0.2613, 0.2758)
        self.image_transforms = T.Compose([
            T.Resize((resize_dim, resize_dim), interpolation=T.InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=mean,
                        std=std),
        ])
        
        self.resize_dim = resize_dim
        self.num_global_tokens = 1 if "reg" not in model_name else 5
        self.patch_dim = 14 if '14' in model_name else 16
        self.num_patch_tokens = resize_dim // self.patch_dim * resize_dim // self.patch_dim
        self.num_tokens = self.num_global_tokens + self.num_patch_tokens
        if 'vitl' in model_name or 'vit_large' in model_name or 'ViT-L' in model_name:
            self.embed_dim = 1024
        elif 'vitb' in model_name or 'vit_base' in model_name or 'ViT-B' in model_name:
            self.embed_dim = 768
        elif 'vits' in model_name or 'vit_small' in model_name:
            self.embed_dim = 384
        
        self.visual_backbone.requires_grad_(False)
        
        self.linear_probing_layer = nn.Linear(self.embed_dim, n_classes)
    
    def forward(self, imgs):
        if 'dinov2' in self.model_name:
            outs = self.visual_backbone(imgs, is_training=True)
        elif 'mae' in self.model_name or 'clip' in self.model_name or 'dino' in self.model_name:
            output = self.visual_backbone.forward_features(imgs)
            # reporting output in DINOv2 format
            outs = {
                'x_norm_clstoken': output[:, 0, :],
                'x_norm_patchtokens': output[:, 1:, :],
            }
        x = outs['x_norm_patchtokens']
        x = self.linear_probing_layer(x)
        return x