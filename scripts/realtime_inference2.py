import argparse
import os
from omegaconf import OmegaConf
import numpy as np
import cv2
import torch
import glob
import pickle
import sys
from tqdm import tqdm
import copy
import json
from transformers import WhisperModel

from musetalk.utils.face_parsing import FaceParsing
from musetalk.utils.utils import datagen
from musetalk.utils.preprocessing import get_landmark_and_bbox, read_imgs
from musetalk.utils.blending import get_image_prepare_material, get_image_blending
from musetalk.utils.utils import load_all_model
from musetalk.utils.audio_processor import AudioProcessor

import shutil
import threading
import queue
import time
import subprocess


def fast_check_ffmpeg():
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        return True
    except:
        return False


def video2imgs(vid_path, save_path, ext='.png', cut_frame=10000000):
    cap = cv2.VideoCapture(vid_path)
    count = 0
    while True:
        if count > cut_frame:
            break
        ret, frame = cap.read()
        if ret:
            cv2.imwrite(f"{save_path}/{count:08d}.png", frame)
            count += 1
        else:
            break


def osmakedirs(path_list):
    for path in path_list:
        os.makedirs(path) if not os.path.exists(path) else None


@torch.no_grad()
class Avatar:
    def __init__(self, avatar_id, video_path, bbox_shift, batch_size, preparation):
        print("Initializing avatar...")
        self.avatar_id = avatar_id
        self.video_path = video_path
        self.bbox_shift = bbox_shift
        self.version = "v15"
        self.extra_margin = 10
        self.parsing_mode = "jaw"
        self.audio_padding_length_left = 2
        self.audio_padding_length_right = 2
        # 根据版本设置不同的基础路径
        # if args.version == "v15":
        self.base_path = f"./results/{self.version}/avatars/{avatar_id}"
        # else:  # v1
        #     self.base_path = f"./results/avatars/{avatar_id}"
            
        self.avatar_path = self.base_path
        self.full_imgs_path = f"{self.avatar_path}/full_imgs"
        self.coords_path = f"{self.avatar_path}/coords.pkl"
        self.latents_out_path = f"{self.avatar_path}/latents.pt"
        self.video_out_path = f"{self.avatar_path}/vid_output/"
        self.mask_out_path = f"{self.avatar_path}/mask"
        self.mask_coords_path = f"{self.avatar_path}/mask_coords.pkl"
        self.avatar_info_path = f"{self.avatar_path}/avator_info.json"
        self.avatar_info = {
            "avatar_id": avatar_id,
            "video_path": video_path,
            "bbox_shift": bbox_shift,
            "version": self.version
        }
        self.preparation = preparation
        self.batch_size = batch_size
        self.idx = 0
        # self.skip_save_images = False
        self.skip_save_images = True
        self.init_model()
        self.init()
        print("Avatar initialized.")

    def init(self):
        if self.preparation:
            print("preparation...")
            if os.path.exists(self.avatar_path):
                response = input(f"{self.avatar_id} exists, Do you want to re-create it ? (y/n)")
                if response.lower() == "y":
                    shutil.rmtree(self.avatar_path)
                    print("*********************************")
                    print(f"  creating avator: {self.avatar_id}")
                    print("*********************************")
                    osmakedirs([self.avatar_path, self.full_imgs_path, self.video_out_path, self.mask_out_path])
                    self.prepare_material()
                else:
                    self.input_latent_list_cycle = torch.load(self.latents_out_path)
                    with open(self.coords_path, 'rb') as f:
                        self.coord_list_cycle = pickle.load(f)
                    input_img_list = glob.glob(os.path.join(self.full_imgs_path, '*.[jpJP][pnPN]*[gG]'))
                    input_img_list = sorted(input_img_list, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
                    self.frame_list_cycle = read_imgs(input_img_list)
                    with open(self.mask_coords_path, 'rb') as f:
                        self.mask_coords_list_cycle = pickle.load(f)
                    input_mask_list = glob.glob(os.path.join(self.mask_out_path, '*.[jpJP][pnPN]*[gG]'))
                    input_mask_list = sorted(input_mask_list, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
                    self.mask_list_cycle = read_imgs(input_mask_list)
            else:
                print("*********************************")
                print(f"  creating avator: {self.avatar_id}")
                print("*********************************")
                osmakedirs([self.avatar_path, self.full_imgs_path, self.video_out_path, self.mask_out_path])
                self.prepare_material()
        else:
            print("not preparation...")
            if not os.path.exists(self.avatar_path):
                print(f"{self.avatar_id} does not exist, you should set preparation to True")
                sys.exit()

            with open(self.avatar_info_path, "r") as f:
                avatar_info = json.load(f)

            if avatar_info['bbox_shift'] != self.avatar_info['bbox_shift']:
                response = input(f" 【bbox_shift】 is changed, you need to re-create it ! (c/continue)")
                if response.lower() == "c":
                    shutil.rmtree(self.avatar_path)
                    print("*********************************")
                    print(f"  creating avator: {self.avatar_id}")
                    print("*********************************")
                    osmakedirs([self.avatar_path, self.full_imgs_path, self.video_out_path, self.mask_out_path])
                    self.prepare_material()
                else:
                    sys.exit()
            else:
                self.input_latent_list_cycle = torch.load(self.latents_out_path)
                with open(self.coords_path, 'rb') as f:
                    self.coord_list_cycle = pickle.load(f)
                input_img_list = glob.glob(os.path.join(self.full_imgs_path, '*.[jpJP][pnPN]*[gG]'))
                input_img_list = sorted(input_img_list, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
                self.frame_list_cycle = read_imgs(input_img_list)
                with open(self.mask_coords_path, 'rb') as f:
                    self.mask_coords_list_cycle = pickle.load(f)
                input_mask_list = glob.glob(os.path.join(self.mask_out_path, '*.[jpJP][pnPN]*[gG]'))
                input_mask_list = sorted(input_mask_list, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
                self.mask_list_cycle = read_imgs(input_mask_list)

    def init_model(self):
        print("Initializing model...")
        ffmpeg_path = "./ffmpeg-4.4-amd64-static/"
        gpu_id = 1 # 
        vae_type = "sd-vae"
        unet_config = "./models/musetalk/musetalk.json"
        unet_model_path = "./models/musetalk/pytorch_model.bin"
        whisper_dir = "./models/whisper"
        left_cheek_width = 90
        right_cheek_width = 90
        
        # Configure ffmpeg path
        if not fast_check_ffmpeg():
            print("Adding ffmpeg to PATH")
            # Choose path separator based on operating system
            path_separator = ';' if sys.platform == 'win32' else ':'
            os.environ["PATH"] = f"{ffmpeg_path}{path_separator}{os.environ['PATH']}"
            if not fast_check_ffmpeg():
                print("Warning: Unable to find ffmpeg, please ensure ffmpeg is properly installed")

        # Set computing device
        self.device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")

        # Load model weights
        self.vae, self.unet, self.pe = load_all_model(
            unet_model_path=unet_model_path,
            vae_type=vae_type,
            unet_config=unet_config,
            device=self.device
        )
        self.timesteps = torch.tensor([0], device=self.device)

        self.pe = self.pe.half().to(self.device)
        self.vae.vae = self.vae.vae.half().to(self.device)
        self.unet.model = self.unet.model.half().to(self.device)

        # Initialize audio processor and Whisper model
        self.audio_processor = AudioProcessor(feature_extractor_path=whisper_dir)
        self.weight_dtype = self.unet.model.dtype
        self.whisper = WhisperModel.from_pretrained(whisper_dir)
        self.whisper = self.whisper.to(device=self.device, dtype=self.weight_dtype).eval()
        self.whisper.requires_grad_(False)

        # Initialize face parser with configurable parameters based on version
        self.fp = FaceParsing(
            left_cheek_width=left_cheek_width,
            right_cheek_width=right_cheek_width
        )
        print("Model initialized.")

    def prepare_material(self):
        print("preparing data materials ... ...")
        with open(self.avatar_info_path, "w") as f:
            json.dump(self.avatar_info, f)

        if os.path.isfile(self.video_path):
            video2imgs(self.video_path, self.full_imgs_path, ext='png')
        else:
            print(f"copy files in {self.video_path}")
            files = os.listdir(self.video_path)
            files.sort()
            files = [file for file in files if file.split(".")[-1] == "png"]
            for filename in files:
                shutil.copyfile(f"{self.video_path}/{filename}", f"{self.full_imgs_path}/{filename}")
        input_img_list = sorted(glob.glob(os.path.join(self.full_imgs_path, '*.[jpJP][pnPN]*[gG]')))

        print("extracting landmarks...")
        coord_list, frame_list = get_landmark_and_bbox(input_img_list, self.bbox_shift)
        input_latent_list = []
        idx = -1
        # maker if the bbox is not sufficient
        coord_placeholder = (0.0, 0.0, 0.0, 0.0)
        for bbox, frame in zip(coord_list, frame_list):
            idx = idx + 1
            if bbox == coord_placeholder:
                continue
            x1, y1, x2, y2 = bbox
            if self.version == "v15":
                y2 = y2 + self.extra_margin
                y2 = min(y2, frame.shape[0])
                coord_list[idx] = [x1, y1, x2, y2]  # 更新coord_list中的bbox
            crop_frame = frame[y1:y2, x1:x2]
            resized_crop_frame = cv2.resize(crop_frame, (256, 256), interpolation=cv2.INTER_LANCZOS4)
            
            latents = self.vae.get_latents_for_unet(resized_crop_frame)
            input_latent_list.append(latents)

        self.frame_list_cycle = frame_list + frame_list[::-1]
        self.coord_list_cycle = coord_list + coord_list[::-1]
        self.input_latent_list_cycle = input_latent_list + input_latent_list[::-1]
        self.mask_coords_list_cycle = []
        self.mask_list_cycle = []

        for i, frame in enumerate(tqdm(self.frame_list_cycle)):
            cv2.imwrite(f"{self.full_imgs_path}/{str(i).zfill(8)}.png", frame)

            x1, y1, x2, y2 = self.coord_list_cycle[i]
            if self.version == "v15":
                mode = self.parsing_mode
            else:
                mode = "raw"
            mask, crop_box = get_image_prepare_material(frame, [x1, y1, x2, y2], fp=self.fp, mode=mode)

            cv2.imwrite(f"{self.mask_out_path}/{str(i).zfill(8)}.png", mask)
            self.mask_coords_list_cycle += [crop_box]
            self.mask_list_cycle.append(mask)

        with open(self.mask_coords_path, 'wb') as f:
            pickle.dump(self.mask_coords_list_cycle, f)

        with open(self.coords_path, 'wb') as f:
            pickle.dump(self.coord_list_cycle, f)

        torch.save(self.input_latent_list_cycle, os.path.join(self.latents_out_path))
        print("preparing data materials done.")

    def process_frames(self, res_frame_queue, video_len, skip_save_images):
        print(video_len)
        while True:
            if self.idx >= video_len - 1:
                break
            try:
                start = time.time()
                res_frame = res_frame_queue.get(block=True, timeout=1)
            except queue.Empty:
                continue

            bbox = self.coord_list_cycle[self.idx % (len(self.coord_list_cycle))]
            ori_frame = copy.deepcopy(self.frame_list_cycle[self.idx % (len(self.frame_list_cycle))])
            x1, y1, x2, y2 = bbox
            try:
                res_frame = cv2.resize(res_frame.astype(np.uint8), (x2 - x1, y2 - y1))
            except:
                continue
            mask = self.mask_list_cycle[self.idx % (len(self.mask_list_cycle))]
            mask_crop_box = self.mask_coords_list_cycle[self.idx % (len(self.mask_coords_list_cycle))]
            combine_frame = get_image_blending(ori_frame,res_frame,bbox,mask,mask_crop_box)

            if skip_save_images is False:
                cv2.imwrite(f"{self.avatar_path}/tmp/{str(self.idx).zfill(8)}.png", combine_frame)
            self.idx = self.idx + 1

    @torch.no_grad()
    def inference(self, audio_path, out_vid_name, fps, skip_save_images):
        os.makedirs(self.avatar_path + '/tmp', exist_ok=True)
        print("start inference")
        ############################################## extract audio feature ##############################################
        start_time = time.time()
        # Extract audio features
        whisper_input_features, librosa_length = self.audio_processor.get_audio_feature(audio_path, weight_dtype=self.weight_dtype)
        whisper_chunks = self.audio_processor.get_whisper_chunk(
            whisper_input_features,
            self.device,
            self.weight_dtype,
            self.whisper,
            librosa_length,
            fps=fps,
            audio_padding_length_left=self.audio_padding_length_left,
            audio_padding_length_right=self.audio_padding_length_right,
        )
        print(f"processing audio:{audio_path} costs {(time.time() - start_time) * 1000}ms")
        ############################################## inference batch by batch ##############################################
        video_num = len(whisper_chunks)
        res_frame_queue = queue.Queue()
        self.idx = 0
        # Create a sub-thread and start it
        process_thread = threading.Thread(target=self.process_frames, args=(res_frame_queue, video_num, skip_save_images))
        process_thread.start()

        gen = datagen(whisper_chunks,
                     self.input_latent_list_cycle,
                     self.batch_size)
        start_time = time.time()
        res_frame_list = []

        for i, (whisper_batch, latent_batch) in enumerate(tqdm(gen, total=int(np.ceil(float(video_num) / self.batch_size)))):
            audio_feature_batch = self.pe(whisper_batch.to(self.device))
            latent_batch = latent_batch.to(device=self.device, dtype=self.unet.model.dtype)

            pred_latents = self.unet.model(latent_batch,
                                    self.timesteps,
                                    encoder_hidden_states=audio_feature_batch).sample
            pred_latents = pred_latents.to(device=self.device, dtype=self.vae.vae.dtype)
            recon = self.vae.decode_latents(pred_latents)
            for res_frame in recon:
                res_frame_queue.put(res_frame)
        # Close the queue and sub-thread after all tasks are completed
        process_thread.join()

        if self.skip_save_images is True:
            print('Total process time of {} frames without saving images = {}s'.format(
                video_num,
                time.time() - start_time))
            
            print(f"streaming... avatar_path={self.avatar_path}  audio_path={audio_path}")
            # stream = f"ffmpeg -re -framerate 25 -f image2 -stream_loop -1 -i {self.avatar_path}/tmp/%08d.png -c:v libx264 -preset ultrafast -tune zerolatency -profile:v baseline -level 3.0 -pix_fmt yuv420p -g 30 -b:v 2048k -f flv -flvflags no_duration_filesize rtmp://43.139.227.110:1935/live/livestream"
            # stream = f"ffmpeg -re -framerate 25 -f image2 -stream_loop -1 -i {audio_path} -i {self.avatar_path}/tmp/%08d.png -c:v libx264 -preset ultrafast -tune zerolatency -profile:v baseline -level 3.0 -pix_fmt yuv420p -g 30 -b:v 2048k -f flv -flvflags no_duration_filesize rtmp://43.139.227.110:1935/live/livestream"
            stream = f"ffmpeg -re -framerate 25 -f image2 -i {self.avatar_path}/tmp/%08d.png -c:v libx264 -preset ultrafast -tune zerolatency -profile:v baseline -level 3.0 -pix_fmt yuv420p -g 30 -b:v 2048k -f flv -flvflags no_duration_filesize rtmp://43.139.227.110:1935/live/livestream"
            # stream = f"ffmpeg -re -framerate 25 -f image2 -i {self.avatar_path}/tmp/%08d.png -i {audio_path} -c:v libx264 -preset ultrafast -tune zerolatency -profile:v baseline -level 3.0 -pix_fmt yuv420p -g 30 -b:v 2048k -f flv -flvflags no_duration_filesize rtmp://43.139.227.110:1935/live/livestream"
            # stream = f"ffmpeg -re -framerate 25 -f image2 -i {self.avatar_path}/tmp/%08d.png -i {audio_path} -c:v libx264 -preset ultrafast -tune zerolatency -profile:v baseline -level 3.0 -pix_fmt yuv420p -g 30 -b:v 2048k -c:a aac -ar 44100 -ac 2 -map 0:v -map 1:a -shortest -f flv -flvflags no_duration_filesize rtmp://43.139.227.110:1935/live/livestream"
            os.system(stream)
            # self.start_background_stream(audio_path)
            print("streaming end")
            
        else:
            print('Total process time of {} frames including saving images = {}s'.format(
                video_num,
                time.time() - start_time))

        if out_vid_name is not None and self.skip_save_images is False:
            # optional
            cmd_img2video = f"ffmpeg -y -v warning -r {fps} -f image2 -i {self.avatar_path}/tmp/%08d.png -vcodec libx264 -vf format=yuv420p -crf 18 {self.avatar_path}/temp.mp4"
            print(cmd_img2video)
            os.system(cmd_img2video)

            output_vid = os.path.join(self.video_out_path, out_vid_name + ".mp4")  # on
            cmd_combine_audio = f"ffmpeg -y -v warning -i {audio_path} -i {self.avatar_path}/temp.mp4 {output_vid}"
            print(cmd_combine_audio)
            os.system(cmd_combine_audio)

            os.remove(f"{self.avatar_path}/temp.mp4")
            shutil.rmtree(f"{self.avatar_path}/tmp")
            print(f"result is save to {output_vid}")
            # stream = f"ffmpeg -re -stream_loop -1 -i {output_vid} -c:v libx264 -preset ultrafast -c:a aac -ar 44100 -ac 2 -f flv rtmps://rtmp.icommu.cn/live/livestream"
            # Input #0, mov,mp4,m4a,3gp,3g2,mj2, from './results/v15/avatars/avator_1/vid_output/audio_0.mp4':
            # rtmps://rtmp.icommu.cn/live/livestream: Input/output error
            stream = f"ffmpeg -re -stream_loop -1 -i {output_vid} -c copy -f flv rtmps://rtmp.icommu.cn/live/livestream"
            # ffmpeg -re -stream_loop -1 -i ./results/v15/avatars/avator_1/vid_output/audio_0.mp4 -c copy -f flv rtmps://rtmp.icommu.cn/live/livestream
            # ffmpeg -re -stream_loop -1 -i ./results/v15/avatars/avator_1/vid_output/audio_0.mp4 -c copy -f flv rtmp://43.139.227.110:1935/live/livestream
            os.system(stream)
        print("\n")

    def start_background_stream(self, audio_path=None):
        """
        启动后台推流
        """
        def stream_worker():
            while True:
                try:
                    # if audio_path and os.path.exists(audio_path):
                    #     cmd = f"ffmpeg -re -framerate 25 -f image2 -stream_loop -1 -i {self.avatar_path}/tmp/%08d.png -i {audio_path} -c:v libx264 -preset ultrafast -tune zerolatency -profile:v baseline -level 3.0 -pix_fmt yuv420p -g 30 -b:v 2048k -c:a aac -ar 44100 -ac 2 -f flv -flvflags no_duration_filesize rtmp://43.139.227.110:1935/live/livestream"
                    # else:
                    cmd = f"ffmpeg -re -framerate 25 -f image2 -stream_loop -1 -i {self.avatar_path}/tmp/%08d.png -c:v libx264 -preset ultrafast -tune zerolatency -profile:v baseline -level 3.0 -pix_fmt yuv420p -g 30 -b:v 2048k -f flv -flvflags no_duration_filesize rtmp://43.139.227.110:1935/live/livestream"
                    
                    print("开始推流...")
                    subprocess.run(cmd, shell=True, check=True)
                except subprocess.CalledProcessError:
                    print("推流中断，5秒后重试...")
                    time.sleep(5)
                except Exception as e:
                    print(f"推流错误: {e}")
                    break
        
        # 启动推流线程
        stream_thread = threading.Thread(target=stream_worker, daemon=True)
        stream_thread.start()
        print("后台推流已启动")

if __name__ == "__main__":
    '''
    This script is used to simulate online chatting and applies necessary pre-processing such as face detection and face parsing in advance. During online chatting, only UNet and the VAE decoder are involved, which makes MuseTalk real-time.
    '''

    # parser = argparse.ArgumentParser()
    # parser.add_argument("--version", type=str, default="v15", choices=["v1", "v15"], help="Version of MuseTalk: v1 or v15")
    # parser.add_argument("--ffmpeg_path", type=str, default="./ffmpeg-4.4-amd64-static/", help="Path to ffmpeg executable")
    # parser.add_argument("--gpu_id", type=int, default=0, help="GPU ID to use")
    # parser.add_argument("--vae_type", type=str, default="sd-vae", help="Type of VAE model")
    # parser.add_argument("--unet_config", type=str, default="./models/musetalk/musetalk.json", help="Path to UNet configuration file")
    # parser.add_argument("--unet_model_path", type=str, default="./models/musetalk/pytorch_model.bin", help="Path to UNet model weights")
    # parser.add_argument("--whisper_dir", type=str, default="./models/whisper", help="Directory containing Whisper model")
    # parser.add_argument("--inference_config", type=str, default="configs/inference/realtime.yaml")
    # parser.add_argument("--bbox_shift", type=int, default=0, help="Bounding box shift value")
    # parser.add_argument("--result_dir", default='./results', help="Directory for output results")
    # parser.add_argument("--extra_margin", type=int, default=10, help="Extra margin for face cropping")
    # parser.add_argument("--fps", type=int, default=25, help="Video frames per second")
    # parser.add_argument("--audio_padding_length_left", type=int, default=2, help="Left padding length for audio")
    # parser.add_argument("--audio_padding_length_right", type=int, default=2, help="Right padding length for audio")
    # parser.add_argument("--batch_size", type=int, default=20, help="Batch size for inference")
    # parser.add_argument("--output_vid_name", type=str, default=None, help="Name of output video file")
    # parser.add_argument("--use_saved_coord", action="store_true", help='Use saved coordinates to save time')
    # parser.add_argument("--saved_coord", action="store_true", help='Save coordinates for future use')
    # parser.add_argument("--parsing_mode", default='jaw', help="Face blending parsing mode")
    # parser.add_argument("--left_cheek_width", type=int, default=90, help="Width of left cheek region")
    # parser.add_argument("--right_cheek_width", type=int, default=90, help="Width of right cheek region")
    # parser.add_argument("--skip_save_images",
    #                    action="store_true",
    #                    help="Whether skip saving images for better generation speed calculation",
    #                    )

    # args = parser.parse_args()
    ffmpeg_path = "./ffmpeg-4.4-amd64-static/"
    gpu_id = 1
    vae_type = "sd-vae"
    unet_config = "./models/musetalk/musetalk.json"
    unet_model_path = "./models/musetalk/pytorch_model.bin"
    whisper_dir = "./models/whisper"
    left_cheek_width = 90
    right_cheek_width = 90
    

    # Configure ffmpeg path
    if not fast_check_ffmpeg():
        print("Adding ffmpeg to PATH")
        # Choose path separator based on operating system
        path_separator = ';' if sys.platform == 'win32' else ':'
        os.environ["PATH"] = f"{ffmpeg_path}{path_separator}{os.environ['PATH']}"
        if not fast_check_ffmpeg():
            print("Warning: Unable to find ffmpeg, please ensure ffmpeg is properly installed")

    # Set computing device
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")

    # Load model weights
    vae, unet, pe = load_all_model(
        unet_model_path=unet_model_path,
        vae_type=vae_type,
        unet_config=unet_config,
        device=device
    )
    timesteps = torch.tensor([0], device=device)

    pe = pe.half().to(device)
    vae.vae = vae.vae.half().to(device)
    unet.model = unet.model.half().to(device)

    # Initialize audio processor and Whisper model
    audio_processor = AudioProcessor(feature_extractor_path=whisper_dir)
    weight_dtype = unet.model.dtype
    whisper = WhisperModel.from_pretrained(whisper_dir)
    whisper = whisper.to(device=device, dtype=weight_dtype).eval()
    whisper.requires_grad_(False)

    # Initialize face parser with configurable parameters based on version
    fp = FaceParsing(
        left_cheek_width=left_cheek_width,
        right_cheek_width=right_cheek_width
    )

    # inference_config = OmegaConf.load(args.inference_config)
    # print(inference_config)

    # for avatar_id in inference_config:
    #     data_preparation = inference_config[avatar_id]["preparation"]
    #     video_path = inference_config[avatar_id]["video_path"]
    #     if args.version == "v15":
    #         bbox_shift = 0
    #     else:
    #         bbox_shift = inference_config[avatar_id]["bbox_shift"]
    #     avatar = Avatar(
    #         avatar_id=avatar_id,
    #         video_path=video_path,
    #         bbox_shift=bbox_shift,
    #         batch_size=args.batch_size,
    #         preparation=data_preparation)

    #     audio_clips = inference_config[avatar_id]["audio_clips"]
    #     for audio_num, audio_path in audio_clips.items():
    #         print("Inferring using:", audio_path)
    #         avatar.inference(audio_path,
    #                        audio_num,
    #                        args.fps,
    #                        args.skip_save_images)
