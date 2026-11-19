import torch
import os
import time
import torch.nn as nn
from timm.models import create_model
from torchvision import transforms
from PIL import Image


def get_model(model_name, num_class):
    if model_name == 'swin_v2':
        model = swinv2_base_window8_256(num_class)
        return model
    elif model_name == 'vit_base':
        model = vit_base_window16_224(num_class)
        return model
    else:
        print("we do not support this model: {}" .format(model_name))

class vit_base_window16_224(nn.Module):
    def __init__(self, num_class=2):
        """Select conv1_1 ~ conv5_1 activation maps."""
        super(vit_base_window16_224, self).__init__()
        # self.model = create_model('vit_base_patch16_224', pretrained=False, num_classes=num_class, global_pool="avg")
        self.model = create_model('vit_base_patch16_224', pretrained=True, num_classes=num_class)

    def forward(self, x):
        out = self.model(x)
        return out
class swinv2_base_window8_256(nn.Module):
    def __init__(self, num_class=2):
        """Select conv1_1 ~ conv5_1 activation maps."""
        super(swinv2_base_window8_256, self).__init__()
        # Load the Swin Transformer model -- swinv2_base_window12to16_192to256_22kft1k
        # self.model = create_model('swinv2_base_window8_256', pretrained=True)
        self.model = create_model('swinv2_base_window12to16_192to256_22kft1k', pretrained=True)
        self.model.head = nn.Linear(self.model.head.in_features, num_class)  # Replace the model head

    def forward(self, x):
        out = self.model(x)
        return out

# def init_iqa_model(
#         MODEL_NAME="./swin_v2_iqa_class_random/iqa_fold1_class_model4swin_v2_lr8e-06-AUC9889.pth",
#         device=torch.device('cpu')):
#     model_classify = None
#     start = time.time()
#
#     if not os.path.isfile(MODEL_NAME):
#         print("Warning: No such model file {} !!!" .format(MODEL_NAME))
#         return model_classify
#
#     try:
#         model_classify = torch.load(MODEL_NAME)
#     except:
#         print("init_iqa_model: load classification model {} error!!!".format(MODEL_NAME))
#
#     end_time = time.time()
#     print('loading IQA model takes time %.2f' % (end_time - start))
#     return model_classify

def init_iqa_model(MODEL_NAME="vit_base_eyeqc_class_4nus/eyeqc_class_model4vit_base_lr1e-06-AUC9988-3DB-statedict.pth",
                   model_backbone='vit_base', num_class=2, device=torch.device('cpu')):
    # model list: 1)iqa_fold1_class_model4swin_v2_lr8e-06-AUC9889-statedict.pth 2) eyeqc_class_model4vit_base_lr1e-06-AUC9988-3DB-statedict.pth

    model_classify = None
    start = time.time()

    # 检查原始路径是否存在
    if os.path.isfile(MODEL_NAME):
        print(f"find the model: {MODEL_NAME}")
    else:
        #  check the current folder
        model_file = os.path.basename(MODEL_NAME)
        current_dir_file = os.path.join("./", model_file)

        if os.path.isfile(current_dir_file):
            print(f"find the model in current folder: {current_dir_file}")
        else:
            print(f"cannot find the model file: {model_file}")
            return None

    try:
        model_classify = get_model(model_backbone, num_class=num_class)
        model_classify.load_state_dict(torch.load(MODEL_NAME, map_location=device))
        model_classify.eval()
    except Exception as e:
        print("init_iqa_model: load classification model {} error: {}!!!".format(MODEL_NAME, e))

    end_time = time.time()
    print('loading IQA model takes time %.2f' % (end_time - start))
    return model_classify

def get_ai_iqa_prediction(file_location, ai_model, threshold=0.67, device=torch.device('cpu'), image_size=224):
    ai_model.eval()

    data_transforms_test = transforms.Compose([
        transforms.Resize(size=(image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    try:
        # Load data and get label
        imgs = Image.open(file_location)
        imgs = imgs.convert('RGB')
        transform = data_transforms_test
        if transform is not None:
            imgs = transform(imgs)

        imgs = torch.unsqueeze(imgs, 0)
        imgs = imgs.to(device)

        model_ft2 = ai_model.to(device)
        out1 = model_ft2(imgs)
        out1 = torch.nn.functional.softmax(out1, 1)
        predict_test = out1.detach().cpu().numpy()[:, -1]
        prediction_value = predict_test[0]
    except:
        print(file_location)
        print('image input:{} process error' .format(file_location))
        return 1.0, 1.0

    # print("IQA prediction_value: {}" .format(prediction_value))

    if prediction_value >= threshold:
        class_result = 1
        probability_1 = 0.5 + ((prediction_value - threshold) / (1 - threshold)) * 0.5
        probability_0 = 1 - probability_1
    else:
        class_result = 0
        probability_1 = 0.5 - ((threshold - prediction_value) / threshold) * 0.5
        probability_0 = 1 - probability_1

    return prediction_value, probability_1


def test_sample_folder(img_path='./test_imgs'):
    running_device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
    iqa_model = init_iqa_model(device=running_device)

    for file in os.listdir(img_path):
        # check if the image ends with png
        if (file.endswith(".png") or file.endswith(".jpg") or file.endswith(".PNG") or file.endswith(".JPG") or \
                file.endswith(".tif") or file.endswith(".TIF") or file.endswith(".bmp") or \
                file.endswith(".BMP") or file.endswith(".jpeg") or file.endswith(".JPEG")):
            images = os.path.join(img_path, file)

            # test for IQA model
            prediction_value, probability_1 = get_ai_iqa_prediction(images, iqa_model, device=running_device)
            print('[File(%s) / IQA Prediction: %0.4f] -- Poor image probability: %.4f, QA Value: %.4f' % (images, prediction_value, probability_1, (1-probability_1)))

def test_crys_folder(img_path='./test_imgs'):
    # running_device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
    running_device = "cpu"
    iqa_model = init_iqa_model(MODEL_NAME="/data/xiaofeng/image_quality_control/swin_v2_iqa_class_KFold/eyeQA_threshold049_model4swin_v2_lr2e-05-AUC9883-crys_T410img-statedict.pth", device=running_device)

    os.makedirs("vi-good")
    os.makedirs("vi-poor")

    poor_quality_list = []
    for file in os.listdir(img_path):
        # check if the image ends with png
        if (file.endswith(".png") or file.endswith(".jpg") or file.endswith(".PNG") or file.endswith(".JPG") or \
                file.endswith(".tif") or file.endswith(".TIF") or file.endswith(".bmp") or \
                file.endswith(".BMP") or file.endswith(".jpeg") or file.endswith(".JPEG")):
            images = os.path.join(img_path, file)

            # test for IQA model
            prediction_value, probability_1 = get_ai_iqa_prediction(images, iqa_model, threshold=0.40, device=running_device)
            # print('[File(%s) / IQA Prediction: %0.4f] -- Poor image probability: %.4f\n' % (images, prediction_value, probability_1))

            if probability_1 >= 0.5:
                poor_quality_list.append(file)
                cmd_line = 'cp "{}" "{}"'.format(images, "vi-poor")
                os.system(cmd_line)
            else:
                cmd_line = 'cp "{}" "{}"'.format(images, "vi-good")
                os.system(cmd_line)

    print("there are total {} poor image in folder {}" .format(len(poor_quality_list), img_path))

def test_syseye_folder(img_path='./iqa-data/new-clear-images/test_tmp'):
    # running_device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
    running_device = "cpu"
    iqa_model = init_iqa_model(
        MODEL_NAME="/data/xiaofeng/image_quality_control/vit_base_eyeqc_class_4nus/eyeqc_class_model4vit_base_lr1e-06-AUC9961-FD-statedict.pth",
        model_backbone='vit_base',
        device=running_device)

    # os.makedirs("vi-good")
    # os.makedirs("vi-poor")

    poor_quality_list = []
    good_quality_list = []
    for file in os.listdir(img_path):
        # check if the image ends with png
        if (file.endswith(".png") or file.endswith(".jpg") or file.endswith(".PNG") or file.endswith(".JPG") or \
                file.endswith(".tif") or file.endswith(".TIF") or file.endswith(".bmp") or \
                file.endswith(".BMP") or file.endswith(".jpeg") or file.endswith(".JPEG")):
            images = os.path.join(img_path, file)

            # test for IQA model
            vit_threshold = 0.76
            prediction_value, probability_1 = get_ai_iqa_prediction(images, iqa_model, threshold=vit_threshold, device=running_device, image_size=224)
            print('[File(%s) / IQA Prediction: %0.4f] -- Poor image probability: %.4f, QA Value: %.4f' %(images, prediction_value, probability_1, (1 - probability_1)))
            if probability_1 > 0.76:
                poor_quality_list.append(file)
                # cmd_line = 'cp "{}" "{}"'.format(images, "vi-poor")
                # os.system(cmd_line)
            else:
                good_quality_list.append(file)
                # cmd_line = 'cp "{}" "{}"'.format(images, "vi-good")
                # os.system(cmd_line)

    print("there are total {} poor images and {} good images in folder {}" \
          .format(len(poor_quality_list), len(good_quality_list), img_path))

# main entry of the test code
if __name__ == '__main__':
    # test_crys_folder("/data2/xiaofeng/Foundation_Model/RETFound_MAE/crystalvue-aviri/pilot-crys-split/test/normal")
    # test_crys_folder("/data2/xiaofeng/Foundation_Model/RETFound_MAE/crystalvue-aviri/pilot-crys-split/test/vi")
    test_syseye_folder("./test_imgs")
    print("eye QA model test is done!!!")