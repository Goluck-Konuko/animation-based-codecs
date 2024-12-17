'''
This code is adapted from the benchmark scripts used in the compressai library for 
learning based image codecs 
(https://github.com/InterDigitalInc/CompressAI/blob/master/compressai/utils/bench/codecs.py)
[Thus relevant license and permissions are transferred here].

Simplified and optimized to use for reference frame coding in the animation-based
video codecs by Goluck Konuko [https://github.com/Goluck-Konuko]
'''

import os,sys
import imageio
import subprocess
import numpy as np
from .utils import read_bitstring, filesize
from tempfile import mkstemp
import time

#TO-DO :
#Implement a decoder that takes a bpg bitstring and decodeds the frame

class BPG:
    """BPG from Fabrice Bellard."""
    def __init__(self, color_mode="rgb", encoder="x265",
                        subsampling_mode="420", bit_depth='8', 
                        encoder_path='bpgenc', decoder_path='bpgdec'):
        self.fmt = ".bpg"
        self.color_mode = color_mode
        self.encoder = encoder
        self.subsampling_mode = subsampling_mode
        self.bitdepth = bit_depth
        self.encoder_path = encoder_path
        self.decoder_path = decoder_path

    def _load_img(self, img):
        return read_image(os.path.abspath(img))

    def _run_impl(self, in_filepath, quality):
        fd0, png_filepath = mkstemp(suffix=".png")
        fd1, out_filepath = mkstemp(suffix=self.fmt)

        # Encode
        enc_start = time.time()
        run_command(self._get_encode_cmd(in_filepath, quality, out_filepath))
        enc_time = time.time()-enc_start
        size = filesize(out_filepath)
        bitstring = read_bitstring(out_filepath)
        # Decode
        dec_start = time.time()
        run_command(self._get_decode_cmd(out_filepath, png_filepath))
        dec_time = time.time()-dec_start
        # Read image
        rec = read_image(png_filepath)
        os.close(fd0)
        os.remove(png_filepath)
        os.close(fd1)
        os.remove(out_filepath)
        out = {
            'bitstring': bitstring,
            'bitstring_size': size,
            'decoded': np.array(rec),
            'time':{'enc_time': enc_time, 'dec_time': dec_time}
        }
        return out

    def run(self,in_filepath,quality: int):
        assert isinstance(in_filepath, np.ndarray)
        
        #create a temporary file for the input image
        fd_in, png_in_filepath = mkstemp(suffix=".png")
        imageio.imsave(png_in_filepath, in_filepath)
        in_file = png_in_filepath

        #compression
        info = self._run_impl(in_file, quality)
        os.close(fd_in)
        os.remove(png_in_filepath)
        return info

    @property
    def name(self):
        return (
            f"BPG {self.bitdepth}b {self.subsampling_mode} {self.encoder} "
            f"{self.color_mode}"
        )

    @property
    def description(self):
        return f"BPG. BPG version {_get_bpg_version(self.encoder_path)}"



    def _get_encode_cmd(self, in_filepath, quality, out_filepath):
        if not 0 <= quality <= 51:
            raise ValueError(f"Invalid quality value: {quality} (0,51)")
        cmd = [
            self.encoder_path,
            "-o",
            out_filepath,
            "-q",
            str(quality),
            "-f",
            self.subsampling_mode,
            "-e",
            self.encoder,
            "-c",
            self.color_mode,
            "-b",
            self.bitdepth,
            in_filepath,
        ]
        return cmd

    def _get_decode_cmd(self, out_filepath, rec_filepath):
        cmd = [self.decoder_path, "-o", rec_filepath, out_filepath]
        return cmd

def read_image(filepath: str, mode: str = "RGB") -> np.array:
    """Return PIL image in the specified `mode` format."""
    if not os.path.isfile(filepath):
        raise ValueError(f'Invalid file "{filepath}".')
    return imageio.imread(filepath)

def run_command(cmd, ignore_returncodes=None):
    cmd = [str(c) for c in cmd]
    try:
        rv = subprocess.check_output(cmd)
        return rv.decode("ascii")
    except subprocess.CalledProcessError as err:
        if ignore_returncodes is not None and err.returncode in ignore_returncodes:
            return err.output
        print(err.output.decode("utf-8"))
        sys.exit(1)


def _get_bpg_version(encoder_path):
    rv = run_command([encoder_path, "-h"], ignore_returncodes=[1])
    return rv.split()[4] 


if __name__ == "__main__":
    img_n = 8
    img = f"imgs/{img_n}.png"
    img_arr = imageio.imread(img)
    qp = 30
    bpg = BPG()

    
    #pass img as np.ndarray :: #Images must be uint8
    out  = bpg.run(img_arr,qp)
    imageio.imsave(f"{img_n}_{qp}_decoded.png", out['decoded'])