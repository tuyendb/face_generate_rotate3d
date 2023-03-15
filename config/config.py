import os

from TMTChatbot.Common.config.config import Config as BaseConfig
from TMTChatbot.Common.common_keys import *

from common.common_keys import *
from common.file_server import *


class Config(BaseConfig):
    def __init__(self):
        super(Config, self).__init__()

        self.gpu_id = os.getenv(GPU_ID, '1')
        self.bfm_dir = os.getenv(BFM_DIR, 'model/BFM')
        self.rec_checkpoint_path = os.getenv(REC_CHECKPOINT_PATH, 'model/checkpoints/deep3d_face_recon/epoch_20.pth')
        self.api_port = int(os.getenv(API_PORT, 35558))
        self.dlib_model_path = os.getenv(DLIB_MODEL_PATH, 'model/dlib_model/shape_predictor_68_face_landmarks.dat')
        self.rec_checkpoint_dir = os.getenv(REC_CHECKPOINT_DIR, 'model/checkpoints/deep3d_face_recon')
        #FILESERVER
        self.fileserver_url = os.getenv(FILESERVER_URL, "https://files.tmtco.dev/api/v2/Media/upload-files")
        self.secret_key = os.getenv(SECRET_KEY, "cjgMuAM3f755NqnVs4WE9CvkLyliKfX8")
        self.max_retries = int(os.getenv(MAX_RETRIES, 3))
        self.time_out = int(os.getenv(TIME_OUT, 5))

        self.file_server_url = os.getenv(FILE_SERVER_URL, "http://172.29.13.23:35432/api/file/upload-file-local")
        self.file_server_get_url = os.getenv(FILE_SERVER_GET_URL, "http://172.29.13.23:35432/api/file/Get-File-Local?guid=")
