"""
Demo code

To run our method, you need a bounding box around the person. The person needs to be centered inside the bounding box and the bounding box should be relatively tight. You can either supply the bounding box directly or provide an [OpenPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose) detection file. In the latter case we infer the bounding box from the detections.

In summary, we provide 3 different ways to use our demo code and models:
1. Provide only an input image (using ```--img```), in which case it is assumed that it is already cropped with the person centered in the image.
2. Provide an input image as before, together with the OpenPose detection .json (using ```--openpose```). Our code will use the detections to compute the bounding box and crop the image.
3. Provide an image and a bounding box (using ```--bbox```). The expected format for the json file can be seen in ```examples/im1010_bbox.json```.

Example with OpenPose detection .json
```
python demo.py --checkpoint=data/models/model_checkpoint_h36m_up3d_extra2d.pt --img=examples/im1010.png --openpose=examples/im1010_openpose.json
```
Example with predefined Bounding Box
```
python demo.py --checkpoint=data/models/model_checkpoint_h36m_up3d_extra2d.pt --img=examples/im1010.png --bbox=examples/im1010_bbox.json
```
Example with cropped and centered image
```
python demo.py --checkpoint=data/models/model_checkpoint_h36m_up3d_extra2d.pt --img=examples/im1010.png
```

Running the previous command will save the results in ```examples/im1010_{gcnn,smpl,gcnn_side,smpl_side}.png```. The files ```im1010_gcnn``` and ```im1010_smpl``` show the overlayed reconstructions of the non-parametric and parametric shapes respectively. We also render side views, saved in ```im1010_gcnn_side.png``` and ```im1010_smpl_side.png```.
"""


import torch
from torchvision.transforms import Normalize
import numpy as np
import cv2
import argparse
import os
import re
import json
import trimesh
import scipy
import subprocess
from models.dmr import DMR
from utils.imutils import crop, uncrop
from utils.renderer import Renderer
import utils.config as cfg
from collections import namedtuple
from os.path import join, exists
from utils.objfile import read_obj

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', default=None, help='Path to pretrained checkpoint')
parser.add_argument('--config', default=None, help='Path to config file containing model architecture etc.')
parser.add_argument('--img', type=str, required=True, help='Path to input image')
parser.add_argument('--bbox', type=str, default=None, help='Path to .json file containing bounding box coordinates')
parser.add_argument('--openpose', type=str, default=None, help='Path to .json containing openpose detections')
parser.add_argument('--outfile', type=str, default=None, help='Filename of output images. If not set use input filename.')
parser.add_argument('--no_render', action='store_true', help="Don't render the output")

parser.add_argument('--image_folder', type=str, default=None, help='Path to the dir containing input images.')
parser.add_argument('--output_folder', type=str, default=None, help='Path to the desired output folder.')
parser.add_argument('--openpose_video', type=str, default=None, help='Path to folder containing all openpose detections.') # or I can run openpose here

def images_to_video(img_folder, output_vid_file):
    os.makedirs(img_folder, exist_ok=True)
    command = ['ffmpeg', '-i', f'{img_folder}/%04d.png', '-vcodec', 'copy', output_vid_file]
    print(f'Running \"{" ".join(command)}\"')
    subprocess.call(command)

def bbox_from_openpose(openpose_file, rescale=1.2, detection_thresh=0.2):
    """Get center and scale for bounding box from openpose detections."""
    with open(openpose_file, 'r') as f:
        keypoints = json.load(f)['people'][0]['pose_keypoints_2d']
    keypoints = np.reshape(np.array(keypoints), (-1,3))
    valid = keypoints[:,-1] > detection_thresh
    valid_keypoints = keypoints[valid][:,:-1]
    center = valid_keypoints.mean(axis=0)
    bbox_size = (valid_keypoints.max(axis=0) - valid_keypoints.min(axis=0)).max()
    # adjust bounding box tightness
    scale = bbox_size / 200.0
    scale *= rescale
    return center, scale

def bbox_from_json(bbox_file):
    """Get center and scale of bounding box from bounding box annotations.
    The expected format is [top_left(x), top_left(y), width, height].
    """
    with open(bbox_file, 'r') as f:
        bbox = np.array(json.load(f)['bbox']).astype(np.float32)
    ul_corner = bbox[:2]
    center = ul_corner + 0.5 * bbox[2:]
    width = max(bbox[2], bbox[3])
    scale = width / 200.0
    # make sure the bounding box is rectangular
    return center, scale

def process_image(img_file, bbox_file, openpose_file, input_res=224):
    """Read image, do preprocessing and possibly crop it according to the bounding box.
    If there are bounding box annotations, use them to crop the image.
    If no bounding box is specified but openpose detections are available, use them to get the bounding box.
    """
    normalize_img = Normalize(mean=cfg.IMG_NORM_MEAN, std=cfg.IMG_NORM_STD)
    # tmp = cv2.imread(img_file)
    img = cv2.imread(img_file)[:,:,::-1].copy()  # PyTorch does not support negative stride at the moment
    if bbox_file is None and openpose_file is None:
        # Assume that the person is centerered in the image
        height = img.shape[0]
        width = img.shape[1]
        center = np.array([width // 2, height // 2])
        scale = max(height, width) / 200
    else:
        if bbox_file is not None:
            center, scale = bbox_from_json(bbox_file)
        elif openpose_file is not None:
            center, scale = bbox_from_openpose(openpose_file)
            print("Center=", center, 'scale=', scale)
    img, ul, orig_shape = crop(img, center, scale, (input_res, input_res))

    img = img.astype(np.float32) / 255.
    img = torch.from_numpy(img).permute(2,0,1)
    norm_img = normalize_img(img.clone())[None]
    return img, norm_img, ul, center, scale, orig_shape

if __name__ == '__main__':
    args = parser.parse_args()

    # Make the output folder
    outfolder = os.path.join(args.output_folder, args.image_folder.split("/")[-2])
    print("OUTPUTFOLDER=", outfolder)
    os.makedirs(outfolder, exist_ok=True)
    os.makedirs(os.path.join(outfolder, 'front'), exist_ok=True)
    os.makedirs(os.path.join(outfolder, 'side'), exist_ok=True)
    os.makedirs(os.path.join(outfolder, 'cams'), exist_ok=True)
    os.makedirs(os.path.join(outfolder, 'meshes'), exist_ok=True)
    
    # Load model
    if args.config is None:
        tmp = args.checkpoint.split('/')[:-2]
        tmp.append('config.json')
        args.config = '/' + join(*tmp)

    with open(args.config, 'r') as f:
        options = json.load(f)
        options = namedtuple('options', options.keys())(**options)

    model = DMR(options, args.checkpoint)
    model.eval()

    # Setup renderer for visualization
    _, faces = read_obj('data/reference_mesh.obj')
    renderer = Renderer(faces=np.array(faces) - 1)



    ################ VIDEO ################
    if args.image_folder:
        # vidcap = cv2.VideoCapture(args.input_video)
        # success,image = vidcap.read()
        # count = 0
        # images = []
        # while success:
        #     #cv2.imwrite("frame%d.jpg" % count, image)     # save frame as JPEG file      
        #     success,image = vidcap.read()
        #     images.append(image)
        #     print('Read a new frame: ', success)
        #     count += 1
        # print("I loaded {} images from {}".format(len(images), args.input_video))

        # Get all openpose keypoint paths
        openpose_files = os.listdir(args.openpose_video)
        image_files = os.listdir(args.image_folder)

        print("BEFORE ORDERING")
        print(openpose_files)
        image_files.sort(key=lambda f: int(re.sub('\D', '', f)))
        openpose_files.sort(key=lambda f: int(re.sub('\D', '', f)))

        print("\n\nAFTER ORDERING:")
        print(openpose_files)

        # Loop over the images and run the model
        for idx, im in enumerate(image_files):
            
            # Get current file
            openpose_file = os.path.join(args.openpose_video, openpose_files[idx])
            image_file = os.path.join(args.image_folder, im)

            print("Index={}".format(idx))
            print("Image path={}".format(image_file))

            # Preprocess input image and generate predictions
            img, norm_img, ul, center, scale, orig_shape = process_image(image_file, None, openpose_file, input_res=cfg.INPUT_RES)
            with torch.no_grad():
                out_dict = model(norm_img.to(model.device))
                pred_vertices = out_dict['pred_vertices']
                pred_camera = out_dict['camera']

            # Calculate camera parameters for rendering
            camera_translation = torch.stack([pred_camera[:,1], pred_camera[:,2], 2*cfg.FOCAL_LENGTH/(cfg.INPUT_RES * pred_camera[:,0] +1e-9)],dim=-1)
            camera_translation = camera_translation[0].cpu().numpy()
            pred_vertices = pred_vertices[0].cpu().numpy()
            img = img.permute(1,2,0).cpu().numpy()

            # Render non-parametric shape
            if not args.no_render:
                img_render = renderer.render(pred_vertices,
                                        camera_t=camera_translation,
                                        img=img, use_bg=True, body_color='pink')
            # NOTE: save meshes and cams here
            mesh = trimesh.Trimesh(vertices=pred_vertices, 
                                   faces=np.array(faces) - 1, 
                                   process=False)
            mesh_filename = os.path.join(outfolder, 'meshes', '{:04d}.obj'.format(idx))
            mesh.export(mesh_filename)
            np.save(os.path.join(outfolder, 'cams', '{:04d}.npy'.format(idx)), [ orig_shape, ul, camera_translation])
            
            #img_render = uncrop(img_render, center, scale, orig_shape)
            #img_render = scipy.misc.imresize(img_render, orig_shape, interp='bilinear')
            
            if not args.no_render:
                ##############
                # --- NOTE: to get the full image. (Opencv has height, width format)
                # get the original full image
                original_img = cv2.imread(image_file)
                
                # get the uncropped render (i.e. resize to original image shape)
                uncropped_img_render = 255 * cv2.resize(img_render[:,:,::-1], dsize=(orig_shape[1],orig_shape[0]), interpolation=cv2.INTER_CUBIC)
                uncropped_img_render = np.clip(uncropped_img_render, 0, 255).astype(np.uint8) # back to uint8

                # now we replace the corresponding patch of the original image with the uncropped render 
                
                # DEBUGGING WEIRD CROP SIZES
                print("\n\noriginal image.shape:", original_img.shape)
                print("uncropped_img_render:", uncropped_img_render.shape)
                print("We will replace the image patch with the crop at location:", ul[1], ul[0])
                print("Does this go over the original image size?:", ul[1]+uncropped_img_render.shape[0], ul[0]+uncropped_img_render.shape[1])

                temp = 0
                extra = 0 

                # final crop pixel indices are below zero: impossible
                if ul[1] < 0:
                    temp = -1*ul[1]
                    print("We will cut off extra from top:", temp)
                    ul[1] = 0
                    original_img[ul[1]:ul[1]+uncropped_img_render.shape[0]-temp, ul[0]:ul[0]+uncropped_img_render.shape[1],:] = uncropped_img_render[temp:,:,:]

                # final crop pixel indices go over original image size
                elif ul[1]+uncropped_img_render.shape[0] > original_img.shape[0]:
                    extra = ul[1]+uncropped_img_render.shape[0] - original_img.shape[0]
                    print("We will cut off extra from bottom:", extra)
                    original_img[ul[1]:ul[1]+uncropped_img_render.shape[0]-extra, ul[0]:ul[0]+uncropped_img_render.shape[1],:] = uncropped_img_render[:-extra,:,:]

                else:
                    original_img[ul[1]:ul[1]+uncropped_img_render.shape[0], ul[0]:ul[0]+uncropped_img_render.shape[1],:] = uncropped_img_render

                print("\n")

                # change names
                img_render = original_img
                ##################

                # Render side views
                aroundy = cv2.Rodrigues(np.array([0, np.radians(270.), 0]))[0]
                center = pred_vertices.mean(axis=0)
                center_smpl = pred_vertices.mean(axis=0)
                rot_vertices = np.dot((pred_vertices - center), aroundy) + center

                # Render non-parametric shape
                img_render_side = renderer.render(rot_vertices,
                                        camera_t=camera_translation,
                                        img=np.ones_like(img), use_bg=True, body_color='pink')
                
                white_image = np.ones(original_img.shape)*255
                uncropped_img_render_side = 255 * cv2.resize(img_render_side[:,:,::-1], dsize=(orig_shape[1],orig_shape[0]), interpolation=cv2.INTER_CUBIC)
                uncropped_img_render_side = np.clip(uncropped_img_render_side, 0, 255).astype(np.uint8)

                print("white_image shape", white_image.shape)
                print("ul[1]:ul[1]+orig_shape[1]", ul[1],ul[1]+uncropped_img_render.shape[1])
                print("ul[0]:ul[0]+orig_shape[0]", ul[0],ul[0]+uncropped_img_render.shape[0])
                print("extra here", extra)

                if extra != 0:
                    white_image[ul[1]:ul[1]+uncropped_img_render.shape[0]-extra-temp, ul[0]:ul[0]+uncropped_img_render.shape[1],:] = uncropped_img_render_side[temp:-extra,:,:]
                else:
                    white_image[ul[1]:ul[1]+uncropped_img_render.shape[0]-extra-temp, ul[0]:ul[0]+uncropped_img_render.shape[1],:] = uncropped_img_render_side[temp:,:,:]

                img_render_side = white_image

                # Render parametric shape
                #outfile = im.split('.')[0] if args.outfile is None else args.outfile

                # Save reconstructions
                cv2.imwrite(os.path.join(outfolder, 'front', '{:04d}.png'.format(idx)), img_render)
                cv2.imwrite(os.path.join(outfolder, 'side', '{:04d}.png'.format(idx)), img_render_side)
                print("Image saved.\n\n\n\n")

        # Convert to video
        if not args.no_render:
            images_to_video(os.path.join(outfolder, 'front'), os.path.join(outfolder, 'front.mp4'))
            images_to_video(os.path.join(outfolder, 'side'), os.path.join(outfolder, 'side.mp4'))

    ############ SINGLE IMAGE ##############
    else:

        # Preprocess input image and generate predictions
        img, norm_img = process_image(args.img, args.bbox, args.openpose, input_res=cfg.INPUT_RES)
        with torch.no_grad():
            out_dict = model(norm_img.to(model.device))
            pred_vertices = out_dict['pred_vertices']
            pred_camera = out_dict['camera']

        # Calculate camera parameters for rendering
        camera_translation = torch.stack([pred_camera[:,1], pred_camera[:,2], 2*cfg.FOCAL_LENGTH/(cfg.INPUT_RES * pred_camera[:,0] +1e-9)],dim=-1)
        camera_translation = camera_translation[0].cpu().numpy()
        pred_vertices = pred_vertices[0].cpu().numpy()
        img = img.permute(1,2,0).cpu().numpy()

        # Render non-parametric shape
        img_render = renderer.render(pred_vertices,
                                camera_t=camera_translation,
                                img=img, use_bg=True, body_color='pink')

        # Render side views
        aroundy = cv2.Rodrigues(np.array([0, np.radians(270.), 0]))[0]
        center = pred_vertices.mean(axis=0)
        center_smpl = pred_vertices.mean(axis=0)
        rot_vertices = np.dot((pred_vertices - center), aroundy) + center

        # Render non-parametric shape
        img_render_side = renderer.render(rot_vertices,
                                camera_t=camera_translation,
                                img=np.ones_like(img), use_bg=True, body_color='pink')
        
        # Render parametric shape
        outfile = args.img.split('.')[0] if args.outfile is None else args.outfile

        # Save reconstructions
        cv2.imwrite(outfile + '_render.png', 255 * img_render[:,:,::-1])
        cv2.imwrite(outfile + '_render_side.png', 255 * img_render_side[:,:,::-1])
        print("Image saved.")
