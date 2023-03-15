from queue import Queue

from config.config import Config
from common.queue_name import QName

from TMTChatbot import BaseServiceSingleton


class BufferManager(BaseServiceSingleton):
    def __init__(self, config: Config = None):
        super(BufferManager, self).__init__(config=config)
        self.qs = {
            QName.IMAGE_LIST_Q: Queue(maxsize=1),
            QName.MASK_TEXTURE_Q: Queue(maxsize=1),
            QName.RENDER_IMAGE_Q: Queue(maxsize=1),
            QName.OUTPUT_Q: Queue(maxsize=1)
        }

    def get_data(self, queue_name: QName):
        if queue_name in self.qs:
            return self.qs[queue_name].get()
        else:
            raise ValueError(f"{queue_name} not in Queue list")

    def put_data(self, queue_name: QName, data):
        if queue_name in self.qs:
            self.qs[queue_name].put(data)
        else:
            raise ValueError(f"{queue_name} not in Queue list")
            