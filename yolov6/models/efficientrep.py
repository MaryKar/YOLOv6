from pickle import FALSE
from torch import nn
from yolov6.layers.common import BottleRep, RepVGGBlock, RepBlock, BepC3, SimSPPF, SPPF, SimCSPSPPF, CSPSPPF, ConvBNSiLU, \
                                MBLABlock, ConvBNHS, Lite_EffiBlockS2, Lite_EffiBlockS1
import torchvision
from torch.nn import Conv2d
from torch import  no_grad
from torch.hub import load
import torch
class EfficientRep(nn.Module):
    '''EfficientRep Backbone
    EfficientRep is handcrafted by hardware-aware neural network design.
    With rep-style struct, EfficientRep is friendly to high-computation hardware(e.g. GPU).
    '''

    def __init__(
        self,
        in_channels=3,
        channels_list=None,
        num_repeats=None,
        block=RepVGGBlock,
        fuse_P2=False,
        cspsppf=False
    ):
        super().__init__()

        assert channels_list is not None
        assert num_repeats is not None
        self.fuse_P2 = fuse_P2

        self.stem = block(
            in_channels=in_channels,
            out_channels=channels_list[0],
            kernel_size=3,
            stride=2
        )

        self.ERBlock_2 = nn.Sequential(
            block(
                in_channels=channels_list[0],
                out_channels=channels_list[1],
                kernel_size=3,
                stride=2
            ),
            RepBlock(
                in_channels=channels_list[1],
                out_channels=channels_list[1],
                n=num_repeats[1],
                block=block,
            )
        )

        self.ERBlock_3 = nn.Sequential(
            block(
                in_channels=channels_list[1],
                out_channels=channels_list[2],
                kernel_size=3,
                stride=2
            ),
            RepBlock(
                in_channels=channels_list[2],
                out_channels=channels_list[2],
                n=num_repeats[2],
                block=block,
            )
        )

        self.ERBlock_4 = nn.Sequential(
            block(
                in_channels=channels_list[2],
                out_channels=channels_list[3],
                kernel_size=3,
                stride=2
            ),
            RepBlock(
                in_channels=channels_list[3],
                out_channels=channels_list[3],
                n=num_repeats[3],
                block=block,
            )
        )

        channel_merge_layer = SPPF if block == ConvBNSiLU else SimSPPF
        if cspsppf:
            channel_merge_layer = CSPSPPF if block == ConvBNSiLU else SimCSPSPPF

        self.ERBlock_5 = nn.Sequential(
            block(
                in_channels=channels_list[3],
                out_channels=channels_list[4],
                kernel_size=3,
                stride=2,
            ),
            RepBlock(
                in_channels=channels_list[4],
                out_channels=channels_list[4],
                n=num_repeats[4],
                block=block,
            ),
            channel_merge_layer(
                in_channels=channels_list[4],
                out_channels=channels_list[4],
                kernel_size=5
            )
        )

    def forward(self, x):

        outputs = []
        x = self.stem(x)
        x = self.ERBlock_2(x)
        if self.fuse_P2:
            print("x1" , x.shape)
            outputs.append(x)
            
        x = self.ERBlock_3(x)
        print("x2" , x.shape)
        outputs.append(x)
        x = self.ERBlock_4(x)
        print("x3" , x.shape)
        outputs.append(x)
        x = self.ERBlock_5(x)
        print("x4" , x.shape)
        outputs.append(x)
        print(g)
        return tuple(outputs)


class EfficientRep6(nn.Module):
    '''EfficientRep+P6 Backbone
    EfficientRep is handcrafted by hardware-aware neural network design.
    With rep-style struct, EfficientRep is friendly to high-computation hardware(e.g. GPU).
    '''

    def __init__(
        self,
        in_channels=3,
        channels_list=None,
        num_repeats=None,
        block=RepVGGBlock,
        fuse_P2=False,
        cspsppf=False
    ):
        super().__init__()

        assert channels_list is not None
        assert num_repeats is not None
        self.fuse_P2 = fuse_P2

        self.stem = block(
            in_channels=in_channels,
            out_channels=channels_list[0],
            kernel_size=3,
            stride=2
        )

        self.ERBlock_2 = nn.Sequential(
            block(
                in_channels=channels_list[0],
                out_channels=channels_list[1],
                kernel_size=3,
                stride=2
            ),
            RepBlock(
                in_channels=channels_list[1],
                out_channels=channels_list[1],
                n=num_repeats[1],
                block=block,
            )
        )

        self.ERBlock_3 = nn.Sequential(
            block(
                in_channels=channels_list[1],
                out_channels=channels_list[2],
                kernel_size=3,
                stride=2
            ),
            RepBlock(
                in_channels=channels_list[2],
                out_channels=channels_list[2],
                n=num_repeats[2],
                block=block,
            )
        )

        self.ERBlock_4 = nn.Sequential(
            block(
                in_channels=channels_list[2],
                out_channels=channels_list[3],
                kernel_size=3,
                stride=2
            ),
            RepBlock(
                in_channels=channels_list[3],
                out_channels=channels_list[3],
                n=num_repeats[3],
                block=block,
            )
        )

        self.ERBlock_5 = nn.Sequential(
            block(
                in_channels=channels_list[3],
                out_channels=channels_list[4],
                kernel_size=3,
                stride=2,
            ),
            RepBlock(
                in_channels=channels_list[4],
                out_channels=channels_list[4],
                n=num_repeats[4],
                block=block,
            )
        )

        channel_merge_layer = SimSPPF if not cspsppf else SimCSPSPPF

        self.ERBlock_6 = nn.Sequential(
            block(
                in_channels=channels_list[4],
                out_channels=channels_list[5],
                kernel_size=3,
                stride=2,
            ),
            RepBlock(
                in_channels=channels_list[5],
                out_channels=channels_list[5],
                n=num_repeats[5],
                block=block,
            ),
            channel_merge_layer(
                in_channels=channels_list[5],
                out_channels=channels_list[5],
                kernel_size=5
            )
        )

    def forward(self, x):

        outputs = []
        x = self.stem(x)
        x = self.ERBlock_2(x)
        if self.fuse_P2:
            outputs.append(x)
        x = self.ERBlock_3(x)
        outputs.append(x)
        x = self.ERBlock_4(x)
        outputs.append(x)
        x = self.ERBlock_5(x)
        outputs.append(x)
        x = self.ERBlock_6(x)
        outputs.append(x)

        return tuple(outputs)


class CSPBepBackbone(nn.Module):
    """
    CSPBepBackbone module.
    """

    def __init__(
        self,
        in_channels=3,
        channels_list=None,
        num_repeats=None,
        block=RepVGGBlock,
        csp_e=float(1)/2,
        fuse_P2=False,
        cspsppf=False,
        stage_block_type="BepC3"
    ):
        super().__init__()

        assert channels_list is not None
        assert num_repeats is not None

        if stage_block_type == "BepC3":
            stage_block = BepC3
        elif stage_block_type == "MBLABlock":
            stage_block = MBLABlock
        else:
            raise NotImplementedError

        self.fuse_P2 = fuse_P2

        self.stem = block(
            in_channels=in_channels,
            out_channels=channels_list[0],
            kernel_size=3,
            stride=2
        )

        self.ERBlock_2 = nn.Sequential(
            block(
                in_channels=channels_list[0],
                out_channels=channels_list[1],
                kernel_size=3,
                stride=2
            ),
            stage_block(
                in_channels=channels_list[1],
                out_channels=channels_list[1],
                n=num_repeats[1],
                e=csp_e,
                block=block,
            )
        )

        self.ERBlock_3 = nn.Sequential(
            block(
                in_channels=channels_list[1],
                out_channels=channels_list[2],
                kernel_size=3,
                stride=2
            ),
            stage_block(
                in_channels=channels_list[2],
                out_channels=channels_list[2],
                n=num_repeats[2],
                e=csp_e,
                block=block,
            )
        )

        self.ERBlock_4 = nn.Sequential(
            block(
                in_channels=channels_list[2],
                out_channels=channels_list[3],
                kernel_size=3,
                stride=2
            ),
            stage_block(
                in_channels=channels_list[3],
                out_channels=channels_list[3],
                n=num_repeats[3],
                e=csp_e,
                block=block,
            )
        )

        channel_merge_layer = SPPF if block == ConvBNSiLU else SimSPPF
        if cspsppf:
            channel_merge_layer = CSPSPPF if block == ConvBNSiLU else SimCSPSPPF

        self.ERBlock_5 = nn.Sequential(
            block(
                in_channels=channels_list[3],
                out_channels=channels_list[4],
                kernel_size=3,
                stride=2,
            ),
            stage_block(
                in_channels=channels_list[4],
                out_channels=channels_list[4],
                n=num_repeats[4],
                e=csp_e,
                block=block,
            ),
            channel_merge_layer(
                in_channels=channels_list[4],
                out_channels=channels_list[4],
                kernel_size=5
            )
        )

    def forward(self, x):

        outputs = []
        x = self.stem(x)
        x = self.ERBlock_2(x)
        if self.fuse_P2:
            outputs.append(x)
        x = self.ERBlock_3(x)
        outputs.append(x)
        x = self.ERBlock_4(x)
        outputs.append(x)
        x = self.ERBlock_5(x)
        outputs.append(x)

        return tuple(outputs)


class CSPBepBackbone_P6(nn.Module):
    """
    CSPBepBackbone+P6 module.
    """

    def __init__(
        self,
        in_channels=3,
        channels_list=None,
        num_repeats=None,
        block=RepVGGBlock,
        csp_e=float(1)/2,
        fuse_P2=False,
        cspsppf=False,
        stage_block_type="BepC3"
    ):
        super().__init__()
        assert channels_list is not None
        assert num_repeats is not None

        if stage_block_type == "BepC3":
            stage_block = BepC3
        elif stage_block_type == "MBLABlock":
            stage_block = MBLABlock
        else:
            raise NotImplementedError
        
        self.fuse_P2 = fuse_P2

        self.stem = block(
            in_channels=in_channels,
            out_channels=channels_list[0],
            kernel_size=3,
            stride=2
        )

        self.ERBlock_2 = nn.Sequential(
            block(
                in_channels=channels_list[0],
                out_channels=channels_list[1],
                kernel_size=3,
                stride=2
            ),
            stage_block(
                in_channels=channels_list[1],
                out_channels=channels_list[1],
                n=num_repeats[1],
                e=csp_e,
                block=block,
            )
        )

        self.ERBlock_3 = nn.Sequential(
            block(
                in_channels=channels_list[1],
                out_channels=channels_list[2],
                kernel_size=3,
                stride=2
            ),
            stage_block(
                in_channels=channels_list[2],
                out_channels=channels_list[2],
                n=num_repeats[2],
                e=csp_e,
                block=block,
            )
        )

        self.ERBlock_4 = nn.Sequential(
            block(
                in_channels=channels_list[2],
                out_channels=channels_list[3],
                kernel_size=3,
                stride=2
            ),
            stage_block(
                in_channels=channels_list[3],
                out_channels=channels_list[3],
                n=num_repeats[3],
                e=csp_e,
                block=block,
            )
        )

        channel_merge_layer = SPPF if block == ConvBNSiLU else SimSPPF
        if cspsppf:
            channel_merge_layer = CSPSPPF if block == ConvBNSiLU else SimCSPSPPF

        self.ERBlock_5 = nn.Sequential(
            block(
                in_channels=channels_list[3],
                out_channels=channels_list[4],
                kernel_size=3,
                stride=2,
            ),
            stage_block(
                in_channels=channels_list[4],
                out_channels=channels_list[4],
                n=num_repeats[4],
                e=csp_e,
                block=block,
            ),
        )
        self.ERBlock_6 = nn.Sequential(
            block(
                in_channels=channels_list[4],
                out_channels=channels_list[5],
                kernel_size=3,
                stride=2,
            ),
            stage_block(
                in_channels=channels_list[5],
                out_channels=channels_list[5],
                n=num_repeats[5],
                e=csp_e,
                block=block,
            ),
            channel_merge_layer(
                in_channels=channels_list[5],
                out_channels=channels_list[5],
                kernel_size=5
            )
        )

    def forward(self, x):

        outputs = []
        x = self.stem(x)
        x = self.ERBlock_2(x)
        outputs.append(x)
        x = self.ERBlock_3(x)
        outputs.append(x)
        x = self.ERBlock_4(x)
        outputs.append(x)
        x = self.ERBlock_5(x)
        outputs.append(x)
        x = self.ERBlock_6(x)
        outputs.append(x)

        return tuple(outputs)

class Lite_EffiBackbone(nn.Module):
    def __init__(self,
                 in_channels,
                 mid_channels,
                 out_channels,
                 num_repeat=[1, 3, 7, 3]
    ):
        super().__init__()
        out_channels[0]=24
        self.conv_0 = ConvBNHS(in_channels=in_channels,
                             out_channels=out_channels[0],
                             kernel_size=3,
                             stride=2,
                             padding=1)

        self.lite_effiblock_1 = self.build_block(num_repeat[0],
                                                 out_channels[0],
                                                 mid_channels[1],
                                                 out_channels[1])

        self.lite_effiblock_2 = self.build_block(num_repeat[1],
                                                 out_channels[1],
                                                 mid_channels[2],
                                                 out_channels[2])

        self.lite_effiblock_3 = self.build_block(num_repeat[2],
                                                 out_channels[2],
                                                 mid_channels[3],
                                                 out_channels[3])

        self.lite_effiblock_4 = self.build_block(num_repeat[3],
                                                 out_channels[3],
                                                 mid_channels[4],
                                                 out_channels[4])

    def forward(self, x):
        outputs = []
        print(x.shape) ## 3 , 640 , 640 
        x = self.conv_0(x)
        x = self.lite_effiblock_1(x)
        x = self.lite_effiblock_2(x)
        print("x1" , x.shape) ## batch , 96 , 80 , 80
        outputs.append(x)
        x = self.lite_effiblock_3(x)
        print("x2" , x.shape) ## batch , 192 , 40 , 40
        outputs.append(x)
        x = self.lite_effiblock_4(x)
        print("x3" , x.shape) ## batch , 384 , 20 , 20
        outputs.append(x)
        # print(g)
        return tuple(outputs)

    @staticmethod
    def build_block(num_repeat, in_channels, mid_channels, out_channels):
        block_list = nn.Sequential()
        for i in range(num_repeat):
            if i == 0:
                block = Lite_EffiBlockS2(
                            in_channels=in_channels,
                            mid_channels=mid_channels,
                            out_channels=out_channels,
                            stride=2)
            else:
                block = Lite_EffiBlockS1(
                            in_channels=out_channels,
                            mid_channels=mid_channels,
                            out_channels=out_channels,
                            stride=1)
            block_list.add_module(str(i), block)
        return block_list

def get_features(features ,name):
    def hook(model, input, output):
        res = output.detach()
        if name == "c3":
            target = torch.zeros(res.shape[0] , res.shape[1] , 80 , 80)
            target[:, :, :79,:79] = res
            features[name] = target
        
        else : features[name] = res
    return hook

class GPUNet_EffiBackbone(nn.Module):
    def __init__(self,
                #  in_channels,
                #  mid_channels,
                #  out_channels,
                #  num_repeat=[1, 3, 7, 3]

    ):
        super().__init__()
        model_type = "GPUNet-0" # select one from above
        precision = "fp32"
        # self.gpunet = load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_gpunet', pretrained=True, model_type=model_type, model_math=precision)

        # self.gpunet = load('./Models/gpunet.pth','gpunet')
        self.model = torchvision.models.resnet50(pretrained=True)
        # mds = list(self.gpunet.modules())
        # c3 = mds[79]
        # c4 = mds[93]
        c3 = self.model.layer2
        c4 = self.model.layer3
        c5 = self.model.layer4
        self.features = {}
        c3.register_forward_hook(get_features(self.features ,"c3"))
        c4.register_forward_hook(get_features(self.features ,"c4"))
        c5.register_forward_hook(get_features(self.features ,"c5"))


    def forward(self, x):
        outputs = []
        c6 = self.model(x)
        outputs.append(self.features["c3"])
        ## 256, 256, 1, 1
        c4_ = Conv2d(in_channels = 1024, out_channels = 256, kernel_size = 20, stride=20)(self.features["c4"])
        outputs.append(c4_)
        
        ## 256, 512, 1, 1 
        c5_ = Conv2d(in_channels = 2048, out_channels = 512, kernel_size = 10, stride=10)(self.features["c5"])
        outputs.append(c5_)

        ## 256, 1024, 1, 1
        c6_ = Conv2d(in_channels = 2048, out_channels = 1024, kernel_size = 20, stride=20)(self.features["c5"])
        outputs.append(c6_)

        return tuple(outputs)
############################ test
# import pickle
# import io
# import torch
# class CPU_Unpickler(pickle.Unpickler):
#     def find_class(self, module, name):
#         if module == 'torch.storage' and name == '_load_from_bytes':
#             return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
#         else:
#             return super().find_class(module, name)

################################ test
class Lite_GPUNet_EffiBackbone(nn.Module):
    def __init__(self,
                #  in_channels,
                #  mid_channels,
                #  out_channels,
                #  num_repeat=[1, 3, 7, 3]

    ):
        super().__init__()
        model_type = "GPUNet-0" # select one from above
        precision = "fp32"
        self.gpunet = load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_gpunet', pretrained=True, model_type=model_type, model_math=precision)

        # self.gpunet = load('./Models/gpunet.pth')
        # self.model = torchvision.models.resnet50(pretrained=True)
        mds = list(self.gpunet.modules())
        c3 = mds[34]
        c4 = mds[68]
        c5 = mds[107]
        # # c3 = self.model.layer2
        # # c4 = self.model.layer3
        # # c5 = self.model.layer4
        self.features = {}

        hc3 = c3.register_forward_hook(get_features(self.features ,"c3"))
        hc4 = c4.register_forward_hook(get_features(self.features ,"c4"))
        hc5 = c5.register_forward_hook(get_features(self.features ,"c5"))
        ###################### test
        # self.features = CPU_Unpickler(open('./GPU_NET/features.pkl','rb')).load()
        # self.features = load(open('./GPU_NET/features.pkl', 'rb'), map_location=torch.device('cpu'))
        ########################### test

    def forward(self, x):
        outputs = []
        with no_grad():
            c5 = self.gpunet(x)
        # print("sfd" , self.features["c3"].shape , self.features["c4"].shape , c5.shape)

        ## 96 , 80 , 80
        # print("c3" ,self.features["c3"].shape ) ## 64, 80, 80
        in_channel_c3 = self.features["c3"].shape[1]
        c3_ = Conv2d(in_channels = in_channel_c3, out_channels = 96, kernel_size = 1, stride=1)(self.features["c3"])
        outputs.append(c3_)
        
        ## 192 , 40 , 40
        ## c4 256, 40, 40
        in_channel_c4 = self.features["c4"].shape[1]
        c4_ = Conv2d(in_channels = in_channel_c4, out_channels = 192, kernel_size = 1, stride=1)(self.features["c4"])
        outputs.append(c4_)
        
        ## 384 , 20 , 20
        ## c5 704 20 20
        in_channel_c5 = self.features["c5"].shape[1]
        c5_ = Conv2d(in_channels = in_channel_c5, out_channels = 384, kernel_size = 1, stride=1)(self.features["c5"])#self.features["c5"]
        outputs.append(c5_)

        # print("---->" , c3_.shape , c4_.shape , c5_.shape)
        # c6_ = Conv2d(in_channels = 2048, out_channels = 1024, kernel_size = 20, stride=20)(self.features["c5"])
        # outputs.append(c6_)

        return tuple(outputs)

