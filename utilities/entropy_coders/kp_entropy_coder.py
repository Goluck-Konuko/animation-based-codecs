'''
An implementation of simple entropy coding using The standard Arithmetic encoder and 
PPM MODEL from:
https://github.com/nayuki/Reference-arithmetic-coding

The entropy coder here is optimized to re-use the frequency table for efficient
keypoint encoding.
TO-DO:
    :: Optimize the frequency table update procedure and PpmModel order selection
    :: Possibly introduce a NN likelihood estimation and use NeuralFrequencyTable table in
    :: the PPM model in place of the SimpleFrequencyTable
'''


import os
import torch
import contextlib
import numpy as np
from tempfile import mkstemp
from .io_utils import read_bitstring, filesize, to_cuda
from .arithmetic_coder import FlatFrequencyTable, SimpleFrequencyTable,\
    ArithmeticEncoder,ArithmeticDecoder, BitOutputStream, BitInputStream
from .ppm_model import PpmModel


class BasicEntropyCoder:
    '''A simple wrapper aroung the standard arithmetic codec'''
    def __init__(self, q_step=64, num_kp=10) -> None:
        self.q_step = q_step
        self.num_kp = num_kp
        self.kp_reference = None

    def quantize(self, tgt):
        return torch.round((tgt+1)*self.q_step/1)

    def dequantize(self, tgt):
        return tgt/self.q_step - 1

    def encode_kp(self,kp_source: dict,kp_driving: dict):
        #input shape
        if self.kp_reference is None:
            self.kp_reference = kp_source

        #compute kp residual
        sr = torch.round((self.kp_reference['value']+1)*self.q_step/1)
        dr = torch.round((kp_driving['value']+1)*self.q_step/1)
        kp_res = sr-dr #keypoint location residual
        #create a compressed bitstring
        info_out  = self.compress(kp_res)
        
        #derive quantized jacobian values
        #Coputing the difference between the jacobian matrices does not work
        #may infact increase the signal entropy
        #encode directly for each frame or propose a strategy to re-use jacobians for 
        #frames that are suffiently close i.e. skip modes on encoding the jacobians
        # if 'jacobian' in kp_driving:
        #     jc = self.quantize(kp_driving['jacobian'])
        #     #compress directly
        #     jc_dec_info, self.freqs  = self.compress(jc)
        #     kp_dec_info['bistring_size'] += jc_dec_info['bistring_size']
        #     kp
            
        #decode the keypoint information
        kp_decoded = (sr - info_out['res'])/self.q_step - 1
        kp_dec = {'value': to_cuda(kp_decoded)}
        # if 'jacobian' in kp_driving:
        #     jc_dec = self.dequantize(jc_dec)
        #     kp_dec.update({'jacobian': jc_dec})
        
        self.kp_reference = kp_dec
        info_out.update({'kp_dec': kp_dec})
        return info_out


    def compress(self, kp: torch.tensor):
        shape = kp.shape
        kp = kp.flatten().numpy().astype(np.int8)
        #create a temporary path
        tmp, tmp_path = mkstemp("inp_temp.bin")
        #convert into a bitstring
        raw_bitstring = np.array(kp).tobytes()
        with open(tmp_path, "wb") as raw:
            raw.write(raw_bitstring)

        #initialize arithmetic coding
        initfreqs = FlatFrequencyTable(257)
        freqs = SimpleFrequencyTable(initfreqs)

        #create an output path
        tmp_out, tmp_out_path = mkstemp("out_temp.bin")
        with open(tmp_path, 'rb') as inp, contextlib.closing(BitOutputStream(open(tmp_out_path, "wb"))) as bitout:
            enc = ArithmeticEncoder(32,bitout)
            while True:
                # Read and encode one byte
                symbol = inp.read(1)
                if len(symbol) == 0:
                    break
                enc.write(freqs, symbol[0])
                freqs.increment(symbol[0])
            enc.write(freqs, 256)  # EOF
            enc.finish()	
        
        bit_size = filesize(tmp_out_path)
        bitstring = read_bitstring(tmp_out_path)
        #get the decoding
        kp_res= self.decompress(tmp_out_path)
        kp_res = np.reshape(kp_res, shape)

        os.close(tmp)
        os.remove(tmp_path)

        os.close(tmp_out)
        os.remove(tmp_out_path)	
        return {'bitstring': bitstring, 'bitstring_size': bit_size, 'res': kp_res}

    def decompress(self, in_path: str):
        dec_p, dec_path = mkstemp("decoding.bin")
        initfreqs = FlatFrequencyTable(257)
        freqs = SimpleFrequencyTable(initfreqs)

        with open(in_path, "rb") as inp, open(dec_path, "wb") as out:
            bitin = BitInputStream(inp)

            dec = ArithmeticDecoder(32, bitin)
            while True:
                # Decode and write one byte
                symbol = dec.read(freqs)
                if symbol == 256:  # EOF symbol
                    break
                out.write(bytes((symbol,)))
                freqs.increment(symbol)
        
        #read decoded_bytes
        with open(dec_path, 'rb') as dec_out:
            decoded_bytes = dec_out.read()

        kp_res = np.frombuffer(decoded_bytes, dtype=np.int8)

        os.close(dec_p)
        os.remove(dec_path)
        return kp_res

import time

def to_tensor(x):
     return torch.tensor(x, dtype=torch.float32)

class KpEntropyCoder(BasicEntropyCoder):
    '''Using PPM context model and an arithmetic codec with persistent frequency tables'''
    def __init__(self, q_step=512, num_kp=10, model_order=0, device='cpu') -> None:
        super().__init__(q_step, num_kp)
        self.history, self.dec_history = [], []
        self.ppm_model = PpmModel(model_order, 257, 256)
        self.dec_ppm_model = PpmModel(model_order, 257, 256)
        
    def encode_kp(self,kp_target: dict, device='cpu'):
        #compute kp residual
        sr = torch.round((self.kp_reference['value']+1.0)*self.q_step)
        dr = torch.round((kp_target['value']+1.0)*self.q_step)
        kp_res = sr-dr #keypoint location residual
        #create a compressed bitstring
        info_out = self.compress(kp_res)
        # decode the keypoint information
        res_hat = info_out['res']
        if torch.cuda.is_available():
            res_hat = res_hat.cuda()
        
        dr_hat = sr - res_hat
        kp_decoded = (dr_hat/self.q_step) - 1.0
        kp_dec = {'value': kp_decoded}
        self.kp_reference = {'value': kp_dec['value'].detach().clone()}
        info_out.update({'kp_hat': kp_dec})
        return info_out

    def compress(self, kp):
        enc_start = time.time()
        shape = kp.shape
        if kp.requires_grad:
             kp = kp.detach()
        kp = kp.cpu().flatten().numpy().astype(np.int8)
        #create a temporary path
        tmp, tmp_path = mkstemp("inp_temp.bin")
        #convert into a bitstring
        raw_bitstring = np.array(kp).tobytes()
        with open(tmp_path, "wb") as raw:
            raw.write(raw_bitstring)

        # Set up encoder and model. In this PPM model, symbol 256 represents EOF;
        # its frequency is 1 in the order -1 context but its frequency
        # is 0 in all other contexts (which have non-negative order).
        #create an output path
        tmp_out, tmp_out_path = mkstemp("out_temp.bin")

        # Perform file compression
        with open(tmp_path, "rb") as inp, \
            contextlib.closing(BitOutputStream(open(tmp_out_path, "wb"))) as bitout:
            enc = ArithmeticEncoder(32, bitout)
            while True:
                # Read and encode one byte
                symbol = inp.read(1)
                if len(symbol) == 0:
                    break
                symbol = symbol[0]
                self.encode_symbol(self.ppm_model, self.history, symbol, enc)
                self.ppm_model.increment_contexts(self.history, symbol)
                
                if self.ppm_model.model_order >= 1:
                    # Prepend current symbol, dropping oldest symbol if necessary
                    if len(self.history) == self.ppm_model.model_order:
                        self.history.pop()
                    self.history.insert(0, symbol)
            
            self.encode_symbol(self.ppm_model, self.history, 256, enc)  # EOF
            enc.finish()  # Flush remaining code bits
        
        bit_size= filesize(tmp_out_path)
        bitstring = read_bitstring(tmp_out_path)
        enc_time = time.time()-enc_start
        #get the decoding

        dec_start = time.time()
        kp_res = self.decompress(tmp_out_path)
        kp_res = np.reshape(kp_res, shape)
        dec_time = time.time()-dec_start
        
        os.close(tmp)
        os.remove(tmp_path)

        os.close(tmp_out)
        os.remove(tmp_out_path)
        return {'bitstring_size': bit_size, 
                'bitstring': bitstring, 'res': to_tensor(kp_res),
                'time':{'enc_time':enc_time,'dec_time':dec_time}}

    def decompress(self, in_path):
        dec_p, dec_path = mkstemp("decoding.bin")
        with open(in_path, "rb") as inp, open(dec_path, "wb") as out:
            bitin = BitInputStream(inp)
            dec = ArithmeticDecoder(32, bitin)
            while True:
                # Decode and write one byte
                symbol = self.decode_symbol(dec, self.dec_ppm_model, self.dec_history)
                if symbol == 256:  # EOF symbol
                    break
                out.write(bytes((symbol,)))
                self.dec_ppm_model.increment_contexts(self.dec_history, symbol)
                
                if self.dec_ppm_model.model_order >= 1:
                    # Prepend current symbol, dropping oldest symbol if necessary
                    if len(self.dec_history) == self.dec_ppm_model.model_order:
                        self.dec_history.pop()
                    self.dec_history.insert(0, symbol)
        #read decoded_bytes
        with open(dec_path, 'rb') as dec_out:
            decoded_bytes = dec_out.read()

        kp_res = np.frombuffer(decoded_bytes, dtype=np.int8)

        os.close(dec_p)
        os.remove(dec_path)
        return kp_res

    def encode_symbol(self, model, history, symbol, enc):
        # Try to use highest order context that exists based on the history suffix, such
        # that the next symbol has non-zero frequency. When symbol 256 is produced at a context
        # at any non-negative order, it means "escape to the next lower order with non-empty
        # context". When symbol 256 is produced at the order -1 context, it means "EOF".
        for order in reversed(range(len(history) + 1)):
            ctx = model.root_context
            for sym in history[ : order]:
                assert ctx.subcontexts is not None
                ctx = ctx.subcontexts[sym]
                if ctx is None:
                    break
            else:  # ctx is not None
                if symbol != 256 and ctx.frequencies.get(symbol) > 0:
                    enc.write(ctx.frequencies, symbol)
                    return
                # Else write context escape symbol and continue decrementing the order
                enc.write(ctx.frequencies, 256)
        # Logic for order = -1
        enc.write(model.order_minus1_freqs, symbol)


    def decode_symbol(self, dec, model, history):
        # Try to use highest order context that exists based on the history suffix. When symbol 256
        # is consumed at a context at any non-negative order, it means "escape to the next lower order
        # with non-empty context". When symbol 256 is consumed at the order -1 context, it means "EOF".
        for order in reversed(range(len(history) + 1)):
            ctx = model.root_context
            for sym in history[ : order]:
                assert ctx.subcontexts is not None
                ctx = ctx.subcontexts[sym]
                if ctx is None:
                    break
            else:  # ctx is not None
                symbol = dec.read(ctx.frequencies)
                if symbol < 256:
                    return symbol
                # Else we read the context escape symbol, so continue decrementing the order
        # Logic for order = -1
        return dec.read(model.order_minus1_freqs)
