#imports
import argparse
import torch
from models import Darknet
from torch.autograd import Variable
import onnx
from tool import load_image
from torchvision import transforms

#load the model from the darknet weights
model = Darknet("cfg/yolov3-spp.cfg")
state_dict =  torch.load("yolov3-spp.pt",map_location = "cpu")
print(model.load_state_dict(state_dict["model"]))

#dictates the the program should use the cpu for the calulations
device = torch.device("cpu")

#parses the arguments passed into the program from the command line declare arguments as such $ python [SCRIPT] eval --content-image [IMAGE]
main_arg_parser = argparse.ArgumentParser(description="parser for fast-neural-style")
subparsers = main_arg_parser.add_subparsers(title="subcommands", dest="subcommand")
eval_arg_parser = subparsers.add_parser("eval", help="parser for evaluation/stylizing arguments")
eval_arg_parser.add_argument("--content-image", type=str, required=True, help="path to content image you want to stylize")
args = main_arg_parser.parse_args()

#load the image which will be processed by the model so that a trace of the model's functionality can be made (this means that an image is run thhrought the model and any calculations that the model makes are recorded and placed into the converted model
content_image = load_image(args.content_image, scale=None)
content_transform = transforms.Compose([transforms.ToTensor(),transforms.Lambda(lambda x: x.mul(255))])
content_image = content_transform(content_image)
content_image = content_image.unsqueeze(0).to(device)

#export the model to .onnx format by performing a trace of the model processing an image and building a new file from the calclations performed by the original model
torch.onnx.export(model, content_image, 'yolov3.onnx', export_params=True)
model = onnx.load('yolov3.onnx')
