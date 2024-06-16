import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.config import cfg
import lib.utils as utils
from torch.autograd import Variable
from models.encoder_decoder.PureT_encoder import Encoder
from models.encoder_decoder.NAIC_decoder import Decoder_NA

class NAIC(nn.Module):
    def __init__(self):
        super(NAIC, self).__init__()
        self.vocab_size = cfg.MODEL.VOCAB_SIZE + 1

        # raw Dimension to Model Dimension
        if cfg.MODEL.ATT_FEATS_DIM == cfg.MODEL.ATT_FEATS_EMBED_DIM:
            self.att_embed = nn.Identity()
        else:
            self.att_embed = nn.Sequential(
                nn.Linear(cfg.MODEL.ATT_FEATS_DIM, cfg.MODEL.ATT_FEATS_EMBED_DIM),
                utils.activation(cfg.MODEL.ATT_FEATS_EMBED_ACT),
                nn.LayerNorm(cfg.MODEL.ATT_FEATS_EMBED_DIM) if cfg.MODEL.ATT_FEATS_NORM == True else nn.Identity(),
                nn.Dropout(cfg.MODEL.DROPOUT_ATT_EMBED)
            )

        use_gx = False
        self.encoder = Encoder(
            embed_dim=cfg.MODEL.ATT_FEATS_EMBED_DIM, 
            input_resolution=(12, 12), 
            depth=cfg.MODEL.BILINEAR.ENCODE_LAYERS, 
            num_heads=cfg.MODEL.BILINEAR.HEAD, 
            window_size=6,
            shift_size=3,
            mlp_ratio=4,
            dropout=0.1,
            use_gx = use_gx
        )
        
        self.decoder = Decoder_NA(
            vocab_size = self.vocab_size, 
            embed_dim = cfg.MODEL.BILINEAR.DIM, 
            depth = cfg.MODEL.BILINEAR.DECODE_LAYERS,
            num_heads = cfg.MODEL.BILINEAR.HEAD, 
            dropout = cfg.MODEL.BILINEAR.DECODE_DROPOUT, 
            ff_dropout = cfg.MODEL.BILINEAR.DECODE_FF_DROPOUT
        )
        
    def forward(self, att_feats):
        if cfg.DATA_LOADER.SEQ_PER_IMG>1:
            att_feats = utils.expand_tensor(att_feats, cfg.DATA_LOADER.SEQ_PER_IMG)

        att_feats = self.att_embed(att_feats)
        _, encoder_out = self.encoder(att_feats)
        decoder_out = self.decoder(encoder_out)
        return F.log_softmax(decoder_out, dim=-1)

    def decode(self, att_feats):
        att_feats = self.att_embed(att_feats)
        _, encoder_out = self.encoder(att_feats)
        decoder_out = self.decoder(encoder_out)
        return F.log_softmax(decoder_out, dim=-1)
