import torch
from torch import nn
from torchvision.models.vgg import vgg19
import torch.nn.functional as F
import torchvision.models as models
import pickle
import torchvision.transforms as transforms


# https://buildersbox.corp-sansan.com/entry/2019/04/29/110000
class VGGLoss(nn.Module):
    def __init__(self, loss_type):
        super(VGGLoss, self).__init__()
        vgg = vgg19(pretrained=True)
        if loss_type == 'vgg22':
            vgg_net = nn.Sequential(*list(vgg.features[:9]))
        elif loss_type == 'vgg54':
            vgg_net = nn.Sequential(*list(vgg.features[:36]))

        for param in vgg_net.parameters():
            param.requires_grad = False

        self.vgg_net = vgg_net.eval()
        self.loss = nn.L1Loss()

        self.register_buffer('vgg_mean', torch.tensor([0.485, 0.456, 0.406], requires_grad=False))
        self.register_buffer('vgg_std', torch.tensor([0.229, 0.224, 0.225], requires_grad=False))

    def forward(self, real_img, fake_img):
        # Resize images
        real_img = F.interpolate(real_img, (224, 224))
        fake_img = F.interpolate(fake_img, (224, 224))

        real_img = real_img.sub(self.vgg_mean[:, None, None]).div(self.vgg_std[:, None, None])
        fake_img = fake_img.sub(self.vgg_mean[:, None, None]).div(self.vgg_std[:, None, None])
        feature_real = self.vgg_net(real_img)
        feature_fake = self.vgg_net(fake_img)
        return self.loss(feature_real, feature_fake)


# https://github.com/cydonia999/VGGFace2-pytorch/issues/4#issuecomment-610903491
class VGGFace(nn.Module):
    def __init__(self, model_path):
        super(VGGFace, self).__init__()
        model = models.resnet50(num_classes=8631, pretrained=False)
        with open(model_path, 'rb') as f:
            obj = f.read()

        weights = {key: torch.from_numpy(arr) for key, arr in pickle.loads(obj, encoding='latin1').items()}
        model.load_state_dict(weights)
        model.eval()

        self.register_buffer('vgg_mean', torch.tensor([0.485, 0.456, 0.406], requires_grad=False))
        self.register_buffer('vgg_std', torch.tensor([0.229, 0.224, 0.225], requires_grad=False))
        self.tf_last_layer_chopped = nn.Sequential(*list(model.children())[:-1])
        self.loss = nn.L1Loss()

    def forward(self, real_img, fake_img):
        # Resize images
        real_img = F.interpolate(real_img, (224, 224))
        fake_img = F.interpolate(fake_img, (224, 224))

        real_img = real_img.sub(self.vgg_mean[:, None, None]).div(self.vgg_std[:, None, None])
        fake_img = fake_img.sub(self.vgg_mean[:, None, None]).div(self.vgg_std[:, None, None])
        feature_real = self.tf_last_layer_chopped(real_img)
        feature_fake = self.tf_last_layer_chopped(fake_img)
        return self.loss(feature_real, feature_fake)
