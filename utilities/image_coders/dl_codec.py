from compressai.zoo import cheng2020_attn
import torch
import time


def count_bytes(strings):
    total_bytes = 0
    for s in strings:
        total_bytes += len(s[-1])
    return total_bytes

class AImageCodec:
    def __init__(self, qp=1, device='cpu') -> None:
        self.codec = cheng2020_attn(quality=qp, pretrained=True).to(device)
    
    def run(self, img):
        enc_start = time.time()
        info = self.codec.compress(img)
        enc_time = time.time()-enc_start
        total_bits = count_bytes(info['strings'])*8

        dec_start = time.time()
        dec_info = self.codec.decompress(**info)
        dec_time = time.time()-dec_start
        rec = dec_info['x_hat']
        out = {
            'bitstring_size': total_bits,
            'decoded': rec,
            'time':{'enc_time': enc_time, 'dec_time': dec_time}
        }
        return out

if __name__ == "__main__":
    img = torch.randn((1,3,256,256))

    codec = AImageCodec()
    codec.run(img)