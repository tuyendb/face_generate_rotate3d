from TMTChatbot import BaseServiceSingleton

from config.config import Config
from model.detect_swap.Deep3DFaceRecon_pytorch.models import create_model
from model.detect_swap.PRNet.api import PRN
from process.option import Option


class LoadModels(BaseServiceSingleton):
    def __init__(self, opt, device, config: Config = None):
        super(LoadModels, self).__init__(config=config)
        self.config = config
        self.opt = opt
        self.device = device

    def init_models(self):
        #Deep3dpytorch model for face reconstruction
        recon_model = create_model(self.opt)
        recon_model.setup(self.opt)
        recon_model.device = self.device
        recon_model.parallelize()
        recon_model.eval()
        #PRNet model for face swapping
        swap_model = PRN(is_dlib=True, prefix='./model')
        
        return recon_model, swap_model
