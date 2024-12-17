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
    def __init__(self):
        self.previous_res = None

    def quantize(self, tgt):
        return torch.round((tgt)).detach().cpu()

    def dequantize(self, tgt):
        return tgt
    
    def mid_rise_quantizer(self, arr,levels=256):
        arr = np.array(arr)
        # print(arr)
        max_val = np.max(arr)
        min_val = np.min(arr)
        range_val = max_val - min_val
        step_size = range_val / levels

        quantized_arr = np.round(np.floor((arr - min_val) / step_size) * step_size + min_val).astype(np.int8)
        quantization_error = np.mean(np.abs(arr - quantized_arr))
        return quantized_arr, quantization_error
    
    def mid_rise_dequantizer(self, quantized_arr, levels=256):
        max_val = np.max(quantized_arr)
        min_val = np.min(quantized_arr)
        range_val = max_val - min_val
        step_size = range_val / levels

        dequantized_arr = (quantized_arr / step_size) * step_size + min_val
        return dequantized_arr
    

    def compress_residual(self,residual: torch.tensor, temporal: bool = True, levels=128):
        shape = residual.shape
        r_flat = residual.cpu().flatten().numpy().astype(np.int16)
        r_flat, err = self.mid_rise_quantizer(r_flat,levels)
        # #create a compressed bitstring
        if temporal and self.previous_res is not None:
            r_delta = r_flat - self.previous_res
            info_out  = self.compress(r_delta)
            r_hat = info_out['res']+self.previous_res
        else:
            info_out  = self.compress(r_flat)
            r_hat = info_out['res']
        self.previous_res = r_hat
        res_hat =  self.mid_rise_dequantizer(r_hat,levels)
        res_hat = np.reshape(r_hat, shape)
        res_hat = torch.tensor(res_hat, dtype=torch.float32) 
        info_out['res_hat'] = res_hat
        return info_out


    def compress(self, kp: torch.tensor):
        # shape = kp.shape
        # kp = kp.cpu().flatten().numpy().astype(np.int8)
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
        res_hat= self.decompress(tmp_out_path)
        # res_hat = np.reshape(res_hat, shape)

        os.close(tmp)
        os.remove(tmp_path)

        os.close(tmp_out)
        os.remove(tmp_out_path)	
        return {'bitstring': bitstring, 'bitstring_size': bit_size, 'res': res_hat}

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

        res_hat = np.frombuffer(decoded_bytes, dtype=np.int8)

        os.close(dec_p)
        os.remove(dec_path)
        return res_hat

        
import time
class ResEntropyCoder(BasicEntropyCoder):
    '''Using PPM context model and an arithmetic codec with persistent frequency tables'''
    def __init__(self, model_order=0, eof=256) -> None:
        super().__init__()
        self.history, self.dec_history = [], []
        self.eof = eof
        self.ppm_model = PpmModel(model_order, self.eof+1, self.eof)
        self.dec_ppm_model = PpmModel(model_order, self.eof+1, self.eof)

        self.inputs = []
        self.encoded = []

    def compress(self, kp):
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
        enc_start = time.time()
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
            
            self.encode_symbol(self.ppm_model, self.history, self.eof, enc)  # EOF
            enc.finish()  # Flush remaining code bits
        enc_time = time.time()-enc_start
        bit_size= filesize(tmp_out_path)
        bitstring = read_bitstring(tmp_out_path)
        #get the decoding
        dec_start = time.time()
        kp_res = self.decompress(tmp_out_path)
        dec_time = time.time()-dec_start
        # kp_res = np.reshape(kp_res, shape)
        
        os.close(tmp)
        os.remove(tmp_path)

        os.close(tmp_out)
        os.remove(tmp_out_path)
        return {'bitstring_size': bit_size, 
                'bitstring': bitstring, 'res': kp_res,
                'time':{'enc_time':enc_time,'dec_time':dec_time}}

    def decompress(self, in_path):
        dec_p, dec_path = mkstemp("decoding.bin")
        with open(in_path, "rb") as inp, open(dec_path, "wb") as out:
            bitin = BitInputStream(inp)
            dec = ArithmeticDecoder(32, bitin)
            while True:
                # Decode and write one byte
                symbol = self.decode_symbol(dec, self.dec_ppm_model, self.dec_history)
                if symbol == self.eof:  # EOF symbol
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
                if symbol != self.eof and ctx.frequencies.get(symbol) > 0:
                    enc.write(ctx.frequencies, symbol)
                    return
                # Else write context escape symbol and continue decrementing the order
                enc.write(ctx.frequencies, self.eof)
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
                if symbol < self.eof:
                    return symbol
                # Else we read the context escape symbol, so continue decrementing the order
        # Logic for order = -1
        return dec.read(model.order_minus1_freqs)


