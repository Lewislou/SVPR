import torch
import torch.nn as nn
import timm
from transformers import CLIPVisionModel, CLIPImageProcessor, CLIPVisionConfig
from torchvision import transforms
import os
import torch.nn.functional as F
class PROVisionTower(nn.Module):
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__()

        self.is_loaded = False

        self.vision_tower_name = vision_tower
        self.select_layer = args.mm_vision_select_layer
        self.select_feature = getattr(args, 'mm_vision_select_feature', 'patch')

        if not delay_load:
            self.load_model()
        elif getattr(args, 'unfreeze_mm_vision_tower', False):
            self.load_model()
        else:
            self.cfg_only = {}#CLIPVisionConfig.from_pretrained(self.vision_tower_name)

    def load_model(self, device_map=None):
        if self.is_loaded:
            print('{} is already loaded, `load_model` called again, skipping.'.format(self.vision_tower_name))
            return

        #self.image_processor = CLIPImageProcessor.from_pretrained(self.vision_tower_name)
        #self.vision_tower = CLIPVisionModel.from_pretrained(self.vision_tower_name, device_map=device_map)
        self.vision_tower = timm.create_model("hf_hub:prov-gigapath/prov-gigapath", pretrained=True)
        #self.vision_tower = timm.create_model("vit_large_patch16_224", img_size=224,init_values=1e-5, num_classes=0)
        #self.vision_tower.load_state_dict(torch.load(os.path.join('/mntcephfs/lab_data/louwei/visual_language_models/uni_model/', "pytorch_model.bin"), map_location="cpu"), strict=True)
        #self.vision_tower = timm.create_model("hf-hub:MahmoodLab/uni", pretrained=True, init_values=1e-5, dynamic_img_size=True)
        #self.image_processor = transforms.Compose(
                            #[
                                #transforms.Resize(224),
                                #transforms.ToTensor(),
                                #transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                            #]
                        #)
        
        #print(self.vision_tower,device_map)
        #self.image_processor = transforms.Compose(
            #[
                #transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
                #transforms.CenterCrop(224),
                #transforms.ToTensor(),
                #transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            #]
        #)
        self.image_processor = CLIPImageProcessor.from_pretrained(
                            "openai/clip-vit-base-patch32"
                        )         
        self.vision_tower.requires_grad_(False)
        self.vision_tower.eval()
        
        self.is_loaded = True

    def feature_select(self, image_forward_outs):
        image_features = image_forward_outs.hidden_states[self.select_layer]
        if self.select_feature == 'patch':
            image_features = image_features[:, 1:]
        elif self.select_feature == 'cls_patch':
            image_features = image_features
        else:
            raise ValueError(f'Unexpected select feature: {self.select_feature}')
        return image_features
    # 进行池化操作
    def pool_features(self,input_features):
        batch_size, num_patches, feature_dim = input_features.shape
        height = width = int(num_patches ** 0.5)  # 假设输入特征是一个正方形的平铺
        assert height * width == num_patches, "输入特征的形状必须是可平方的"

        # 重新调整特征形状为 (batch_size, feature_dim, height, width)
        input_features = input_features.view(batch_size, height, width, feature_dim).permute(0, 3, 1, 2)

        # 使用自适应平均池化将其池化到 (batch_size, feature_dim, 7, 7)
        pooled_features = F.adaptive_avg_pool2d(input_features, (7, 7))

        # 调整池化后的特征形状为 (batch_size, 49, feature_dim)
        pooled_features = pooled_features.permute(0, 2, 3, 1).view(batch_size, -1, feature_dim)

        return pooled_features
    @torch.no_grad()
    def forward(self, images):
        #print(type(images))
        #print(self.vision_tower.patch_embed.proj.weight.dtype)
        dtype = self.vision_tower.patch_embed.proj.weight.dtype
        if type(images) is list:
            image_features = []
            for image in images:
                image_forward_out = self.vision_tower(image.to(device=self.device, dtype=dtype).unsqueeze(0))
                image_feature = self.feature_select(image_forward_out).to(image.dtype)
                image_features.append(image_feature)
        else:
            #print(self.dtype)
            self.vision_tower = self.vision_tower.to(device=self.device)
            image_forward_outs = self.vision_tower.forward_features(images.to(device=self.device, dtype=dtype))
            image_forward_outs = image_forward_outs[:, 1:]
            #print(image_forward_outs.shape)
            #if images.shape[0] > 50:
            #image_forward_outs = self.pool_features(image_forward_outs)
                
            #print(image_forward_outs.shape)
            #image_forward_outs = image_forward_outs.reshape(image_forward_outs.shape[0]*image_forward_outs.shape[1],image_forward_outs.shape[2])
            #image_features = self.feature_select(image_forward_outs).to(images.dtype)
            image_features = image_forward_outs.to(images.dtype)
            #image_features = image_features.unsqueeze(1)
            #print(image_features.shape)

        return image_features

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    #@property
    #def dtype(self):
        #return self.vision_tower.patch_embedding.weight.dtype

    @property
    def device(self):
        return 'cuda'

    @property
    def config(self):
        if self.is_loaded:
            return self.vision_tower.config
        else:
            return self.cfg_only

    @property
    def hidden_size(self):
        return 1536

    @property
    def num_patches_per_side(self):
        return self.config.image_size // self.config.patch_size

    @property
    def num_patches(self):
        return (self.config.image_size // self.config.patch_size) ** 2



