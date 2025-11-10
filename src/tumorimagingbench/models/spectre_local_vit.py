from spectre import SpectreImageFeatureExtractor

from .utils import get_transforms
from .base import BaseModel


class SpectreLocalViTExtractor(BaseModel):
    def __init__(self):
        super().__init__()
        
        config = {
            "backbone": "vit_large_patch16_128",
            "backbone_checkpoint_path_or_url": r"/gpfs/work4/0/tese0618/Projects/SPECTRE/weights/spectre_backbone_vit_large_patch16_128.pt",
            "backbone_kwargs": {
                "num_classes": 0,
                "global_pool": '',
                "pos_embed": "rope",
                "rope_kwargs": {"base": 1000.0},
                "init_values": 1.0,
            },
        }
        self.model = SpectreImageFeatureExtractor.from_config(config)
        self.transforms = get_transforms(
            orient="RAS",
            scale_range=(-1000, 1000),
            spatial_size=(128, 128, 64),
            spacing=(0.5, 0.5, 1.0),
        )

    def load(self, weights_path: str = None):
        self.model.eval()

    def preprocess(self, x):
        return self.transforms(x)

    def forward(self, x):
        x = x.unsqueeze(1)  # Add `patch` dimension
        print(x.shape)
        return self.model(x)