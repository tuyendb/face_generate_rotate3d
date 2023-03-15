from TMTChatbot.Common.utils.logging_utils import setup_logging
from TMTChatbot.ServiceWrapper import BaseApp

from config.config import Config
from process.processor import Processor


class CustomApp(BaseApp):
    def __init__(self, config: Config = None):
        super(CustomApp, self).__init__(config=config, with_kafka_app=False, with_default_pipeline=False)
        self.processor = Processor(config=config)
        self.api_app.add_endpoint(endpoint="/",
                                  func=self.processor.process,
                                  methods=["POST"],
                                  use_thread=False,
                                  use_async=False,
                                  description="FaceGenerate")


def create_app(multiprocess: bool = False):
    _config = Config()
    setup_logging(logging_folder=_config.logging_folder, log_name=_config.log_name)
    if multiprocess:
        raise NotImplementedError("Multiprocessing app not created")
    else:
        _app = CustomApp(config=_config)
    return _app


main_app = create_app(False)
app = main_app.api_app.app