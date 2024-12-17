from skimage import img_as_ubyte, img_as_float32
import numpy as np
import subprocess
import imageio
import shutil
import time
import torch
import os
from .io_utils import filesize, read_bitstring


def compute_bitrate(bits, fps,frames):
    return ((bits*8*fps)/(1000*frames))

class VvenC:
    '''
        VvenC CODEC WRAPPER
    '''
    def __init__(self, qp = 50,fps=30,frame_dim='256x256',gop_size=10, sequence = None,out_path='vvc_logs/'):
        self.qp = qp
        self.fps = fps
        self.n_frames = len(sequence)
        self.frame_dim = frame_dim
        self.skip_frames = 0
        self.intra_period = (gop_size-gop_size%4)

        self.input = sequence
        self.out_path = out_path

        #inputs
        self.in_mp4_path = self.out_path+'in_video_'+str(self.qp)+'.mp4'
        self.in_yuv_path = self.out_path+'in_video_'+str(self.qp)+'.yuv'

        #outputs
        self.ostream_path = self.out_path+'out_'+str(self.qp)+'.266'
        self.dec_yuv_path = self.out_path+'out_'+str(self.qp)+'.yuv'
        self.dec_mp4_path = self.out_path+'out_'+str(self.qp)+'.mp4'
        
        #logging file
        self.log_path =  self.out_path+'out_'+str(self.qp)+'.log'

        if not os.path.exists(self.out_path):
            os.makedirs(self.out_path)

        #create yuv video
        self._create_mp4()
        self._mp4_2_yuv()

		
    def _create_config(self):
        '''
            Creates a configuration file for HEVC encoder
        '''
        with open(self.config_path, 'r') as file:
            template = file.read()
        #print(template)
        template = template.replace('inputYUV', str(self.in_yuv_path))
        template = template.replace('outStream', str(self.ostream_path))
        template = template.replace('outYUV', str(self.dec_yuv_path))
        template = template.replace('inputW', str(self.frame_dim[0]))
        template = template.replace('inputH', str(self.frame_dim[1]))
        template = template.replace('inputNrFrames', str(self.n_frames))
        template = template.replace('intraPeriod', str(self.intra_period))
        template = template.replace('inputSkip', str(self.skip_frames))
        template = template.replace('inputFPS', str(self.fps))
        template = template.replace('setQP', str(self.qp))
        with open(self.config_out_path, 'w+') as cfg_file:
            cfg_file.write(template)


    def _create_mp4(self):
        frames = [img_as_ubyte(frame) for frame in self.input]

        writer = imageio.get_writer(self.in_mp4_path, format='FFMPEG', mode='I',fps=self.fps, codec='libx264',pixelformat='yuv420p', quality=10)
        for frame in frames:
            writer.append_data(frame)
        writer.close()	
		
    def _mp4_2_yuv(self):
        #check for yuv video in target directory
        subprocess.call(['ffmpeg','-nostats','-loglevel','error','-i',self.in_mp4_path,self.in_yuv_path, '-r',str(self.fps)])
		
    def _yuv_2_mp4(self):
        cmd = ['ffmpeg','-nostats','-loglevel','error', '-f', 'rawvideo', '-pix_fmt','yuv420p10le','-s:v', self.frame_dim, '-r', str(self.fps), '-i', self.dec_yuv_path,  self.dec_mp4_path]
        subprocess.call(cmd)
	
    def _load_sequences(self):
        original = imageio.mimread(self.input_path, memtest=False)
        decoded = imageio.mimread(self.dec_mp4_path, memtest=False)
        return original, decoded        
		
    def _get_decoded_frames(self):
        #convert yuv to mp4
        self._yuv_2_mp4()
        frames = imageio.mimread(self.dec_mp4_path, memtest=False)
        # hevc_frames = torch.tensor(np.array([np.array([img_as_float32(frame) for frame in frames]).transpose((3, 0, 1, 2))]), dtype=torch.float32)
        return frames

		
    def run(self):
        ## Encoding
        cmd = ['vvencapp', '--preset', 'fast', '-i', self.in_yuv_path, '-s', self.frame_dim,'-q', str(self.qp),   '-f', str(self.n_frames),'-ip',str(self.intra_period), '-o', self.ostream_path]
        # cmd = ["models/utils/hevc_hm/hm_16_15_regular/bin/TAppEncoderStatic", "-c", self.config_out_path,"-i", self.in_yuv_path]
        # with open(self.log_path, 'w+') as out:
        #     subprocess.call(cmd, stdout=out)
        enc_start = time.time()
        subprocess.call(cmd, stdout=subprocess.DEVNULL,stderr=subprocess.STDOUT)
        enc_time = time.time()-enc_start

        bit_size = filesize(self.ostream_path)
        bitstring = read_bitstring(self.ostream_path)


        ## Decoding
        # dec_cmd = ['vvdecapp', '-b', self.ostream_path, '-o', self.dec_yuv_path]
        # with open(self.log_path, 'w+') as out:
        #     subprocess.call(dec_cmd, stdout=out)
        # bit_size = filesize(self.ostream_path)
        # bitrate = compute_bitrate(bytes,self.fps,self.n_frames)
        # dec_frames = self._get_decoded_frames()

        dec_start = time.time()
        dec_cmd = ['vvdecapp', '-b', self.ostream_path, '-o', self.dec_yuv_path]
        subprocess.call(dec_cmd, stdout=subprocess.DEVNULL,stderr=subprocess.STDOUT)
        dec_time = time.time() - dec_start

        dec_frames = self._get_decoded_frames()
        shutil.rmtree(self.out_path)
        return {'dec_frames':dec_frames,
                'bitstring_size': bit_size,
                'bitstring':bitstring,
                'time':{'enc_time': enc_time,'dec_time':dec_time}}


				
if __name__ == "__main__":
    in_path = "sample_vid/7.mp4"
    video = imageio.mimread(in_path, memtest=False)
    fps = imageio.get_reader(in_path).get_meta_data()['fps']

    H,W,C = video[0].shape
    params = {'qp': 50,
             'fps':fps,
             'frame_dim': f"{H}x{W}" ,
             'gop_size': 10, 
             'sequence': video[:100],
             'out_path': 'vvc_logs/'}
    codec = VvenC(**params)
    dec_frames, bitrate, nm_bits = codec.run()
    print(dec_frames.shape, bitrate)