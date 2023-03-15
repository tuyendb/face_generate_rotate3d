import os
import torch

from TMTChatbot import BaseDataModel, BaseServiceSingleton

from config.config import Config
os.environ['CUDA_VISIBLE_DEVICES'] = Config().gpu_id
from objects.image_list import ImageList
from pipeline.pipeline import Pipeline
from process.init_model import LoadModels
from process.option import Option
from common.queue_name import QName
from services.buffer_manager import BufferManager


class Processor(BaseServiceSingleton):
    def __init__(self, config: Config = None):
        super(Processor, self).__init__(config=config)
        self.device = torch.device(0)
        torch.cuda.set_device(self.device)
        self.option = Option().parse()
        self.recon_model, self.swap_model = LoadModels(self.option, self.device, config=self.config).init_models()
        self.buffer_manager = BufferManager(config=config)
        self.pipeline = Pipeline(option=self.option, recon_model=self.recon_model, swap_model=self.swap_model)
        self.pipeline.start()
        self.pipeline.join()

    def process(self, data: BaseDataModel):
        image_list = ImageList.from_json(data.data)
        self.buffer_manager.put_data(queue_name=QName.IMAGE_LIST_Q, data=image_list)
        image_list: ImageList = self.buffer_manager.get_data(queue_name=QName.OUTPUT_Q)
        print("GET DONE")
        if image_list.compulsory_flag:
            data.data = image_list.all_data(0)
        else:
            data.data = image_list.all_data(1)
        return data
