
import modules.scripts as scripts
import gradio as gr
import os
import cv2
from PIL import Image
import numpy as np
from modules import images
from modules.processing import process_images, Processed
from modules.processing import Processed
from modules.shared import opts, cmd_opts, state
from skimage import measure
from modules import scripts_postprocessing
from modules.ui_components import FormRow
import os
import datetime
from modules import scripts_postprocessing, devices, scripts
import gradio as gr

from modules.ui_components import FormRow

import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

from pixelization.models.networks import define_G
import pixelization.models.c2pGen
import re


pixelize_code = [
    233356.8125, -27387.5918, -32866.8008, 126575.0312, -181590.0156,
    -31543.1289, 50374.1289, 99631.4062, -188897.3750, 138322.7031,
    -107266.2266, 125778.5781, 42416.1836, 139710.8594, -39614.6250,
    -69972.6875, -21886.4141, 86938.4766, 31457.6270, -98892.2344,
    -1191.5887, -61662.1719, -180121.9062, -32931.0859, 43109.0391,
    21490.1328, -153485.3281, 94259.1797, 43103.1992, -231953.8125,
    52496.7422, 142697.4062, -34882.7852, -98740.0625, 34458.5078,
    -135436.3438, 11420.5488, -18895.8984, -71195.4141, 176947.2344,
    -52747.5742, 109054.6562, -28124.9473, -17736.6152, -41327.1562,
    69853.3906, 79046.2656, -3923.7344, -5644.5229, 96586.7578,
    -89315.2656, -146578.0156, -61862.1484, -83956.4375, 87574.5703,
    -75055.0469, 19571.8203, 79358.7891, -16501.5000, -147169.2188,
    -97861.6797, 60442.1797, 40156.9023, 223136.3906, -81118.0547,
    -221443.6406, 54911.6914, 54735.9258, -58805.7305, -168884.4844,
    40865.9609, -28627.9043, -18604.7227, 120274.6172, 49712.2383,
    164402.7031, -53165.0820, -60664.0469, -97956.1484, -121468.4062,
    -69926.1484, -4889.0151, 127367.7344, 200241.0781, -85817.7578,
    -143190.0625, -74049.5312, 137980.5781, -150788.7656, -115719.6719,
    -189250.1250, -153069.7344, -127429.7891, -187588.2500, 125264.7422,
    -79082.3438, -114144.5781, 36033.5039, -57502.2188, 80488.1562,
    36501.4570, -138817.5938, -22189.6523, -222146.9688, -73292.3984,
    127717.2422, -183836.3750, -105907.0859, 145422.8750, 66981.2031,
    -9596.6699, 78099.4922, 70226.3359, 35841.8789, -116117.6016,
    -150986.0156, 81622.4922, 113575.0625, 154419.4844, 53586.4141,
    118494.8750, 131625.4375, -19763.1094, 75581.1172, -42750.5039,
    97934.8281, 6706.7949, -101179.0078, 83519.6172, -83054.8359,
    -56749.2578, -30683.6992, 54615.9492, 84061.1406, -229136.7188,
    -60554.0000, 8120.2622, -106468.7891, -28316.3418, -166351.3125,
    47797.3984, 96013.4141, 71482.9453, -101429.9297, 209063.3594,
    -3033.6882, -38952.5352, -84920.6719, -5895.1543, -18641.8105,
    47884.3633, -14620.0273, -132898.6719, -40903.5859, 197217.3750,
    -128599.1328, -115397.8906, -22670.7676, -78569.9688, -54559.7070,
    -106855.2031, 40703.1484, 55568.3164, 60202.9844, -64757.9375,
    -32068.8652, 160663.3438, 72187.0703, -148519.5469, 162952.8906,
    -128048.2031, -136153.8906, -15270.3730, -52766.3281, -52517.4531,
    18652.1992, 195354.2188, -136657.3750, -8034.2622, -92699.6016,
    -129169.1406, 188479.9844, 46003.7500, -93383.0781, -67831.6484,
    -66710.5469, 104338.5234, 85878.8438, -73165.2031, 95857.3203,
    71213.1250, 94603.1094, -30359.8125, -107989.2578, 99822.1719,
    184626.3594, 79238.4531, -272978.9375, -137948.5781, -145245.8125,
    75359.2031, 26652.7930, 50421.4141, 60784.4102, -18286.3398,
    -182851.9531, -87178.7969, -13131.7539, 195674.8906, 59951.7852,
    124353.7422, -36709.1758, -54575.4766, 77822.6953, 43697.4102,
    -64394.3438, 113281.1797, -93987.0703, 221989.7188, 132902.5000,
    -9538.8574, -14594.1338, 65084.9453, -12501.7227, 130330.6875,
    -115123.4766, 20823.0898, 75512.4922, -75255.7422, -41936.7656,
    -186678.8281, -166799.9375, 138770.6250, -78969.9531, 124516.8047,
    -85558.5781, -69272.4375, -115539.1094, 228774.4844, -76529.3281,
    -107735.8906, -76798.8906, -194335.2812, 56530.5742, -9397.7529,
    132985.8281, 163929.8438, -188517.7969, -141155.6406, 45071.0391,
    207788.3125, -125826.1172, 8965.3320, -159584.8438, 95842.4609,
    -76929.4688
]

path_checkpoints = os.path.join(scripts.basedir(), "checkpoints")
path_pixelart_vgg19 = os.path.join(path_checkpoints, "pixelart_vgg19.pth")
path_160_net_G_A = os.path.join(path_checkpoints, "160_net_G_A.pth")
path_alias_net = os.path.join(path_checkpoints, "alias_net.pth")

def extract_number_from_filename(filename):
    pattern = r"\d+"  # Regular expression pattern to match one or more digits
    numbers = re.findall(pattern, filename)
    if numbers:
        return int(numbers[-1])  # Extract the last number
    else:
        return None

class TorchHijackForC2pGen:
    def __getattr__(self, item):
        if item == 'load':
            return self.load

        if hasattr(torch, item):
            return getattr(torch, item)

        raise AttributeError("'{}' object has no attribute '{}'".format(type(self).__name__, item))

    def load(self, filename, *args, **kwargs):
        if filename == "./pixelart_vgg19.pth":
            filename = path_pixelart_vgg19

        return torch.load(filename, *args, **kwargs)


pixelization.models.c2pGen.torch = TorchHijackForC2pGen()


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.G_A_net = None
        self.alias_net = None

    def load(self):
        os.makedirs(path_checkpoints, exist_ok=True)

        missing = False

        if not os.path.exists(path_pixelart_vgg19):
            print(f"Missing {path_pixelart_vgg19} - download it from https://drive.google.com/uc?id=1VRYKQOsNlE1w1LXje3yTRU5THN2MGdMM")
            missing = True

        if not os.path.exists(path_160_net_G_A):
            print(f"Missing {path_160_net_G_A} - download it from https://drive.google.com/uc?id=1i_8xL3stbLWNF4kdQJ50ZhnRFhSDh3Az")
            missing = True

        if not os.path.exists(path_alias_net):
            print(f"Missing {path_alias_net} - download it from https://drive.google.com/uc?id=17f2rKnZOpnO9ATwRXgqLz5u5AZsyDvq_")
            missing = True

        assert not missing, 'Missing checkpoints for pixelization - see console for doqwnload links.'

        with torch.no_grad():
            self.G_A_net = define_G(3, 3, 64, "c2pGen", "instance", False, "normal", 0.02, [0])
            self.alias_net = define_G(3, 3, 64, "antialias", "instance", False, "normal", 0.02, [0])

            G_A_state = torch.load(path_160_net_G_A)
            for p in list(G_A_state.keys()):
                G_A_state["module." + str(p)] = G_A_state.pop(p)
            self.G_A_net.load_state_dict(G_A_state)

            alias_state = torch.load(path_alias_net)
            for p in list(alias_state.keys()):
                alias_state["module." + str(p)] = alias_state.pop(p)
            self.alias_net.load_state_dict(alias_state)


def process(img):
    ow, oh = img.size

    nw = int(round(ow / 4) * 4)
    nh = int(round(oh / 4) * 4)

    left = (ow - nw) // 2
    top = (oh - nh) // 2
    right = left + nw
    bottom = top + nh

    img = img.crop((left, top, right, bottom))

    trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    return img #trans(img)[None, :, :, :]


def to_image(tensor, pixel_size, upscale_after):
    img = tensor.data[0].cpu().float().numpy()
    img = (np.transpose(img, (1, 2, 0)) + 1) / 2.0 * 255.0
    img = img.astype(np.uint8)
    img = Image.fromarray(img)
    img = img.resize((img.size[0]//4, img.size[1]//4), resample=Image.Resampling.NEAREST)
    if upscale_after:
        img = img.resize((img.size[0]*pixel_size, img.size[1]*pixel_size), resample=Image.Resampling.NEAREST)

    return img




def create_gif(atlas_image, frame_height):
    frame_width = atlas_image.width

    # Calculate the number of frames in each column
    num_frames = atlas_image.height // frame_height

    # Calculate the number of columns
    num_columns = atlas_image.width // frame_width

    # Create a new list of lists to hold vertically looped images for each column
    looped_images = []
    for col in range(num_columns):
        column_images = []
        for frame in range(num_frames):
            # Crop the current frame from the atlas image
            left = col * frame_width
            upper = frame * frame_height
            right = left + frame_width
            lower = upper + frame_height
            cropped_image = atlas_image.crop((left, upper, right, lower))

            # Append the cropped image to the column images list
            column_images.append(cropped_image)

        # Append the column images list to the looped images list
        looped_images.append(column_images)

    return looped_images

def remove_bg(pil_image):
    # convert the PIL image to OpenCV format (numpy array)
    img = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

    # Create a grayscale version of the image
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Threshold the image: this will create a binary image where
    # white pixels are those that were greater than 254 and black pixels the rest.
    _, thresh = cv2.threshold(gray, 230, 255, cv2.THRESH_BINARY_INV)

    # Find contours in the thresholded image
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create an empty mask to fill in the found contours
    mask = np.zeros_like(thresh)

    # Draw white filled contours on the black mask
    cv2.drawContours(mask, contours, -1, (255), thickness=cv2.FILLED)

    # Create a 4-channel image (RGBA) from the original and the mask
    rgba = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    rgba[:, :, 3] = mask

    # Convert the image back to PIL format
    pil_image = Image.fromarray(cv2.cvtColor(rgba, cv2.COLOR_BGRA2RGBA))

    return pil_image

def concatenate_images(images):
    # Determine output dimensions
    output_width = max(image.width for image in images)
    output_height = sum(image.height for image in images)

    # Create a new blank image with transparent background
    output_image = Image.new('RGBA', (output_width, output_height), (0, 0, 0, 0))

    # Paste the images vertically
    y_offset = 0
    for image in images:
        output_image.paste(image, (0, y_offset))
        y_offset += image.height

    return output_image


class ScriptPostprocessingUpscale(scripts_postprocessing.ScriptPostprocessing):
    name = "Pixel Trans"
    order = 10000
    model = None
    gif_id = None

    def ui(self):
        with FormRow():
            with gr.Column():
                with FormRow():
                    enable = gr.Checkbox(True, label="Enable pixelization")
                    save_original = gr.Checkbox(True, label="Save Original")
                    save_transparent = gr.Checkbox(True, label="Save Stransparent")
                    save_pixelization = gr.Checkbox(False, label="Save Pixelized")
                    save_pixel_transparent = gr.Checkbox(True, label="Save Pixel Transparent Pixelized")
                    upscale_after = gr.Checkbox(False, label="Keep resolution")

            with gr.Column():
                pixel_size = gr.Slider(minimum=1, maximum=16, step=1, label="Pixel size", value=4, elem_id="pixelization_pixel_size")

        return {
            "enable": enable,
            "save_original": save_original,
            "save_transparent": save_transparent,
            "save_pixelization": save_pixelization,
            "save_pixel_transparent": save_pixel_transparent,
            "upscale_after": upscale_after,
            "pixel_size": pixel_size,
        }

    def process(self, pp: scripts_postprocessing.PostprocessedImage, enable, save_original,save_pixelization,save_transparent,save_pixel_transparent,upscale_after, pixel_size):
        if not enable:
            return


        image=pp.image
        pixel_image=None
        trans_image=None
        file_id=0

        if (save_original):
            file_name, _=images.save_image(pp.image,basename= "original" ,path=opts.outdir_img2img_samples,  extension=opts.samples_format, info= pp.info)
        if (save_atlas):
            file_name, _=images.save_image(remove_bg(pp.image),basename= "atlas" ,path=opts.outdir_img2img_samples,  extension=opts.samples_format, info= pp.info) 


        if self.model is None:
            model = Model()
            model.load()

            self.model = model

        self.model.to(devices.device)


        image = image.resize((image.width * 4 // pixel_size, image.height * 4 // pixel_size))

        with torch.no_grad():
            in_t = process(image).to(devices.device)

            feature = self.model.G_A_net.module.RGBEnc(in_t)
            code = torch.asarray(pixelize_code, device=devices.device).reshape((1, 256, 1, 1))
            adain_params = self.model.G_A_net.module.MLP(code)
            my_images = self.model.G_A_net.module.RGBDec(feature, adain_params)
            out_t = self.model.alias_net(my_images)
            pixel_imagery = to_image(out_t, pixel_size=pixel_size, upscale_after=upscale_after)
            pixel_image=pixel_imagery
            trans_image=remove_bg(pixel_imagery)

  
        self.model.to(devices.cpu)

        pixel_output=pixel_image
        trans_output=trans_image
        
        if (save_pixelization):
            file_name, _=images.save_image(pixel_output,basename= "pixel"  ,path=opts.outdir_img2img_samples,  extension=opts.samples_format, info= pp.info) 


        if (save_pixel_transparent):
            file_name, _=images.save_image(trans_output,basename= "trans_pixel"  ,path=opts.outdir_img2img_samples,  extension=opts.samples_format, info= pp.info) 
        

        pp.image=pixel_output
        pp.info["Pixelization pixel size"] = pixel_size

        file_id=extract_number_from_filename(str(file_name)) 
        current_time = datetime.datetime.now()
        print("{} > {}".format(current_time,file_id))

