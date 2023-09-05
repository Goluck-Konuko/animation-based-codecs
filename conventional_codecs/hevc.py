from skimage import img_as_ubyte, img_as_float32
from .io_utils import filesize, read_bitstring
import numpy as np
import subprocess
import imageio
import shutil
import os
import time


class HEVC:
    '''
        HEVC HM CODEC WRAPPER
    '''
    def __init__(self,seq_name='0',bits=8, qp = 50,fps=30,frame_dim=(256,256),gop_size=10, config='conventional_codecs/hevc_hm/config_template.cfg', sequence = None):
        self.qp = qp
        self.fps = fps
        self.bits= bits
        self.n_frames = len(sequence)
        self.frame_dim = frame_dim
        self.skip_frames = 0
        self.intra_period = (gop_size-gop_size%4)

        self.input = sequence
        self.out_path = f"hevc_logs/{seq_name}/"
        self.config_name = 'hevc_'+str(qp)+'.cfg'
        self.config_path = config

        #inputs
        self.in_mp4_path = self.out_path+'in_video_'+str(self.qp)+'.mp4'
        self.in_yuv_path = self.out_path+'in_video_'+str(self.qp)+'.yuv'

        #outputs
        self.ostream_path = self.out_path+'out_'+str(self.qp)+'.bin'
        self.dec_yuv_path = self.out_path+'out_'+str(self.qp)+'.yuv'
        self.dec_mp4_path = self.out_path+'out_'+str(self.qp)+'.mp4'
        
        #logging file
        self.log_path =  self.out_path+'out_'+str(self.qp)+'.log'

        if not os.path.exists(self.out_path):
            os.makedirs(self.out_path)
        
        self.config_out_path = self.out_path+self.config_name
        self._create_config()

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
        template = template.replace('inputBit', str(self.bits))
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
        cmd = ['ffmpeg','-nostats','-loglevel','error', '-f', 'rawvideo', '-pix_fmt','yuv420p','-s:v', f'{self.frame_dim[0]}x{self.frame_dim[1]}', '-r', str(self.fps), '-i', self.dec_yuv_path,  self.dec_mp4_path]
        subprocess.call(cmd)      
		
    def _get_rec_frames(self):
        #convert yuv to mp4
        self._yuv_2_mp4()
        frames = imageio.mimread(self.dec_mp4_path, memtest=False)
        # frames = torch.tensor(np.array([img_as_float32(frame) for frame in frames]).transpose((3, 0, 1, 2)), dtype=torch.float32)
        return np.array(frames)

    def __str__(self) -> str:
        return "HEVC"
		
    def run(self):
        cmd = ["conventional_codecs/hevc_hm/hm_16_15_regular/bin/TAppEncoderStatic", "-c", self.config_out_path,"-i", self.in_yuv_path]
        # cmd = ["hevc_hm/hm_16_15_regular/bin/TAppEncoderStatic", "-c", self.config_out_path,"-i", self.in_yuv_path]
        
        enc_start = time.time()
        # with open(self.log_path, 'w') as out:
        #     subprocess.call(cmd, stdout=out)
        subprocess.call(cmd, stdout=subprocess.DEVNULL,stderr=subprocess.STDOUT)
        enc_time = time.time()-enc_start
        bit_size = filesize(self.ostream_path)
        bitstring = read_bitstring(self.ostream_path)
        

        dec_cmd = ["conventional_codecs/hevc_hm/hm_16_15_regular/bin/TAppDecoderStatic", "-b", self.ostream_path,'-o',self.dec_yuv_path]
        # dec_cmd = ["hevc_hm/hm_16_15_regular/bin/TAppDecoderStatic", "-b", self.ostream_path,'-o',self.dec_yuv_path]
        dec_start = time.time()
        subprocess.call(dec_cmd, stdout=subprocess.DEVNULL,stderr=subprocess.STDOUT)
        dec_time = time.time() - dec_start
        hevc_frames = self._get_rec_frames()
        shutil.rmtree(self.out_path)
        # out = 
        return {'dec_frames':hevc_frames,
                'bitstring_size': bit_size,
                'bitstring':bitstring,
                'time':{'enc_time': enc_time,'dec_time':dec_time}}

				
