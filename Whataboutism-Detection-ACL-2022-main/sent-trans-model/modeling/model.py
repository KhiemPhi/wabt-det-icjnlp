import pytorch_lightning as pl 
from transformers import AutoModel

class BaselineTransformer(pl.LightningModule):

    def __init__(self):
        super().__init__()         
        self.sent_model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L12-v2")    
    
  