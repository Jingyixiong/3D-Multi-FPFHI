import os 
import timm
import torch

import matplotlib.pyplot as plt

from torchvision import transforms
from PIL import Image

class ModelINet(torch.nn.Module):
    # hrnet_w32, wide_resnet50_2
    def __init__(self, device, backbone_name='wide_resnet50_2', out_indices=(1, 2, 3), checkpoint_path='',
                 pool_last=False):
        super().__init__()
        # Determine if to output features.
        kwargs = {'features_only': True if out_indices else False}
        if out_indices:
            kwargs.update({'out_indices': out_indices})
        print(backbone_name)
        self.backbone = timm.create_model(model_name=backbone_name, pretrained=True, checkpoint_path=checkpoint_path,
                                          **kwargs)
        self.backbone.eval()

        self.device = device
        self.avg_pool = torch.nn.AdaptiveAvgPool2d((1, 1)) if pool_last else None

    def forward(self, x):
        x = x.to(self.device)

        # Backbone forward pass.
        features = self.backbone(x)

        # Adaptive average pool over the last layer.
        if self.avg_pool:
            fmap = features[-1]
            fmap = self.avg_pool(fmap)
            fmap = torch.flatten(fmap, 1)
            features.append(fmap)

        # 最后返回的特征默认是 第2，3尺度的特征
        return features

if __name__ == '__main__':
    # load images
    img_n = 'img.png'
    # root_p = '/media/jing/Storage/anomaly_dataset/al_general_infra/real_arch/Narch_131123/vs_0.02_rs_512/mult_view_img/view_0'
    root_p = '/media/jing/Storage/anomaly_dataset/al_general_infra/real_arch/memory_bank/Narch_130305/vs_0.02_rs_512/mult_view_img/view_0'
    img_path = os.path.join(root_p, img_n)
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]
    rgb_transform = transforms.Compose(
            #  [transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
              [transforms.ToTensor(),
              transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)]  # it does not require the input to be (0, 1)
             )
    
    test_img = Image.open(img_path).convert('RGB')
    plt.imshow(test_img)
    plt.show()
    test_img = rgb_transform(test_img)
    # test_img = test_img.numpy()
    # print(test_img.shape)
    # plt.imshow(test_img)
    # plt.show()
    # test_img = np.asarray(test_img)

    # load model
    out_indices=(1, 2, 3)
    kwargs = {'features_only': True if out_indices else False}
    if out_indices:
        kwargs.update({'out_indices': out_indices})
    print(kwargs)
    device = 'cuda'
    
    model = timm.create_model(
        model_name='resnet50d',
        in_chans=3,
        pretrained=True, checkpoint_path='',
        **kwargs
        ).to(device)
    model.eval()

    # print(model.default_cfg)
    # print(model.feature_info.module_name())
    # print(model.feature_info.reduction())
    # print(model.feature_info.channels())

    # the img size for training the network is 224*224
    # img = torch.rand([1, 3, 512, 512]).to(device)
    print(test_img.unsqueeze(dim=0).shape)
    outputs = model(test_img.unsqueeze(dim=0).to(device))

    for o in outputs:
        plt.imshow(o[0].transpose(0, 2).transpose(1, 0).sum(-1).detach().to('cpu').numpy())
        plt.show()
        print(o.shape)