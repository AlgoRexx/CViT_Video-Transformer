import sys,os
import argparse
import torch
import json
from time import perf_counter
from datetime import datetime
from facenet_pytorch import MTCNN
from model.cvit import CViT
from helpers.loader import load_data
from torch.cuda.amp import autocast, GradScaler
import torch
from model.pred_func import *


'''
I hv inverted the prediction of images
Change the if needed
Both are in this
'''



def load_cvit(cvit_weight, fp16):
    weight_path = os.path.join("weight", cvit_weight)

    if not os.path.exists(weight_path):
        raise FileNotFoundError(f"Model weight file {weight_path} not found.")

    try:
        # Load model checkpoint
        checkpoint = torch.load(weight_path, map_location=torch.device('cpu'))

        model = CViT()  # Initialize your model here
        model.load_state_dict(checkpoint['state_dict'])

        # Check if CUDA is available
        if torch.cuda.is_available():
            device = torch.device("cuda")
            model = model.to(device)
            if fp16:
                model = model.half()  # Use half precision on CUDA-supported device
        else:
            device = torch.device("cpu")
            model = model.to(device)
            fp16 = False  # Disable fp16 on CPU, fallback to float32
            print("CUDA not available, using CPU. Disabling fp16.")

        print(f"Loaded model weights from {weight_path}")
        return model

    except Exception as e:
        raise RuntimeError(f"Failed to load the model weights: {str(e)}")
    


def vids(
    cvit_weight, root_dir="sample_prediction_data", dataset=None, num_frames=15, net=None, fp16=False
):
    result = set_result()
    r = 0
    f = 0
    count = 0
    
    model = load_cvit(cvit_weight, fp16)

    for filename in os.listdir(root_dir):
        curr_vid = os.path.join(root_dir, filename)

        try:
            if is_video(curr_vid) or is_image(curr_vid):
                result, accuracy, count, pred = predict(
                    curr_vid,
                    model,
                    fp16,
                    result,
                    num_frames,
                    net,
                    "uncategorized",
                    count,
                )
                f, r = (f + 1, r) if "FAKE" == real_or_fake(pred[0]) else (f, r + 1)
                print(
                    f"\nPrediction: {pred[1]} {real_or_fake(pred[0])} \t\tFake: {f} Real: {r}"
                )
            else:
                print(f"Invalid file: {curr_vid}. Please provide a valid video or image file.")

        except Exception as e:
            print(f"An error occurred: {str(e)}")

    return result



def faceforensics(
    cvit_weight, root_dir="FaceForensics\\data", dataset=None, num_frames=15, net=None, fp16=False
):
    vid_type = ["original_sequences", "manipulated_sequences"]
    result = set_result()
    result["video"]["compression"] = []
    ffdirs = [
        "DeepFakeDetection",
        "Deepfakes",
        "Face2Face",
        "FaceSwap",
        "NeuralTextures",
    ]

    # load files not used in the training set, the files are appended with compression type, _c23 or _c40
    with open(os.path.join("json_file", "ff_file_list.json")) as j_file:
        ff_file = list(json.load(j_file))

    count = 0
    accuracy = 0
    model = load_cvit(cvit_weight, fp16)

    for v_t in vid_type:
        for dirpath, dirnames, filenames in os.walk(os.path.join(root_dir, v_t)):
            klass = next(
                filter(lambda x: x in dirpath.split(os.path.sep), ffdirs),
                "original",
            )
            label = "REAL" if klass == "original" else "FAKE"
            for filename in filenames:
                try:
                    if filename in ff_file:
                        curr_vid = os.path.join(dirpath, filename)
                        compression = "c23" if "c23" in curr_vid else "c40"
                        if is_video(curr_vid):
                            result, accuracy, count, _ = predict(
                                curr_vid,
                                model,
                                fp16,
                                result,
                                num_frames,
                                net,
                                klass,
                                count,
                                accuracy,
                                label,
                                compression,
                            )
                        else:
                            print(f"Invalid video file: {curr_vid}. Please provide a valid video file.")

                except Exception as e:
                    print(f"An error occurred: {str(e)}")

    return result



def timit(cvit_weight, root_dir="DeepfakeTIMIT", dataset=None, num_frames=15, net=None, fp16=False):
    keywords = ["higher_quality", "lower_quality"]
    result = set_result()
    model = load_cvit(cvit_weight, fp16)
    count = 0
    accuracy = 0
    i = 0
    for keyword in keywords:
        keyword_folder_path = os.path.join(root_dir, keyword)
        for subfolder_name in os.listdir(keyword_folder_path):
            subfolder_path = os.path.join(keyword_folder_path, subfolder_name)
            if os.path.isdir(subfolder_path):
                # Loop through the AVI files in the subfolder
                for filename in os.listdir(subfolder_path):
                    if filename.endswith(".avi"):
                        curr_vid = os.path.join(subfolder_path, filename)
                        try:
                            if is_video(curr_vid):
                                result, accuracy, count, _ = predict(
                                    curr_vid,
                                    model,
                                    fp16,
                                    result,
                                    num_frames,
                                    net,
                                    "DeepfakeTIMIT",
                                    count,
                                    accuracy,
                                    "FAKE",
                                )
                            else:
                                print(f"Invalid video file: {curr_vid}. Please provide a valid video file.")

                        except Exception as e:
                            print(f"An error occurred: {str(e)}")

    return result



def dfdc(
    cvit_weight,
    root_dir="deepfake-detection-challenge\\train_sample_videos",
    dataset=None,
    num_frames=15,
    net=None,
    fp16=False,
):
    result = set_result()
    if os.path.isfile(os.path.join("json_file", "dfdc_files.json")):
        with open(os.path.join("json_file", "dfdc_files.json")) as data_file:
            dfdc_data = json.load(data_file)

    if os.path.isfile(os.path.join(root_dir, "metadata.json")):
        with open(os.path.join(root_dir, "metadata.json")) as data_file:
            dfdc_meta = json.load(data_file)
    model = load_cvit(cvit_weight, fp16)
    count = 0
    accuracy = 0
    
    for dfdc in dfdc_data:
        dfdc_file = os.path.join(root_dir, dfdc)

        try:
            if is_video(dfdc_file):
                result, accuracy, count, _ = predict(
                    dfdc_file,
                    model,
                    fp16,
                    result,
                    num_frames,
                    net,
                    "dfdc",
                    count,
                    accuracy,
                    dfdc_meta[dfdc]["label"],
                )
            else:
                print(f"Invalid video file: {dfdc_file}. Please provide a valid video file.")

        except Exception as e:
            print(f"An error occurred: {str(e)}")

    return result



def celeb(cvit_weight, root_dir="Celeb-DF-v2", dataset=None, num_frames=15, net=None, fp16=False):
    with open(os.path.join("json_file", "celeb_test.json"), "r") as f:
        cfl = json.load(f)
    result = set_result()
    ky = ["Celeb-real", "Celeb-synthesis"]
    count = 0
    accuracy = 0
    model = load_cvit(cvit_weight, fp16)

    for ck in cfl:
        ck_ = ck.split("/")
        klass = ck_[0]
        filename = ck_[1]
        correct_label = "FAKE" if klass == "Celeb-synthesis" else "REAL"
        vid = os.path.join(root_dir, ck)

        try:
            if is_video(vid):
                result, accuracy, count, _ = predict(
                    vid,
                    model,
                    fp16,
                    result,
                    num_frames,
                    net,
                    klass,
                    count,
                    accuracy,
                    correct_label,
                )
            else:
                print(f"Invalid video file: {vid}. Please provide a valid video file.")

        except Exception as e:
            print(f"An error occurred x: {str(e)}")

    return result



def is_image(file_path):
    """
    Check if the file is an image based on the extension.
    """
    image_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.gif']
    return any(file_path.lower().endswith(ext) for ext in image_extensions)



# Preprocess image before prediction
def preprocess_image(image_path, fp16=False):
    """
    Load and preprocess a single image for the model.
    """
    img = Image.open(image_path)
    img = img.resize((224, 224))  # Resize the image to the required input size for the model
    img_tensor = torch.tensor(np.array(img)).permute(2, 0, 1).unsqueeze(0)  # Convert image to tensor and add batch dimension

    if fp16:
        img_tensor = img_tensor.half()  # Convert to half precision if specified
    else:
        img_tensor = img_tensor.float()

    return img_tensor / 255.0  # Normalize the image tensor to [0, 1] range



def predict(
    input_path,
    model,
    fp16,
    result,
    num_frames,
    net,
    klass,
    count=0,
    accuracy=-1,
    correct_label="unknown",
    compression=None,
):
    count += 1
    print(f"\n\n{str(count)} Loading... {input_path}")

    # Check if the input is a video or an image
    if is_video(input_path):
        df = df_face(input_path, num_frames)  # Extract face from video frames
    elif is_image(input_path):
        df = preprocess_image(input_path, fp16)  # Process a single image
    else:
        # Unsupported file type
        return result, accuracy, count, [None, None]

    # Prepare data in the correct precision
    if fp16:
        df = df.half()  # Convert input data to half precision
    else:
        df = df.float()

    try:
        # Use autocast for half-precision operations where possible
        with autocast(enabled=fp16):
            y, y_val = pred_vid(df, model) if len(df) >= 1 else (torch.tensor(0).item(), torch.tensor(0.5).item())
        
        # Reverse prediction labels if it's an image
        if is_image(input_path):
            #y = 1 - y  # Flip the predicted label (0 -> 1 and 1 -> 0)
            y = 1 - y

        result = store_result(
            result, os.path.basename(input_path), y, y_val, klass, correct_label, compression
        )

        if accuracy > -1:
            if correct_label == real_or_fake(y):
                accuracy += 1
            print(f"\nPrediction: {y_val} {real_or_fake(y)} \t\t {accuracy}/{count} {accuracy/count}")

    except RuntimeError as e:
        # Fallback to full precision if error occurs in half precision
        print(f"Half precision failed, retrying with full precision. Error: {str(e)}")
        df = df.float()  # Convert data back to full precision
        model = model.float()  # Convert model back to full precision
        y, y_val = pred_vid(df, model) if len(df) >= 1 else (torch.tensor(0).item(), torch.tensor(0.5).item())

        # Reverse prediction labels if it's an image
        if is_image(input_path):
            y = 1 - y
            #y = 1 - y  # Flip the predicted label

        result = store_result(
            result, os.path.basename(input_path), y, y_val, klass, correct_label, compression
        )

    return result, accuracy, count, [y, y_val]



def gen_parser():
    parser = argparse.ArgumentParser("CViT prediction")
    parser.add_argument("--p", type=str, help="video or image path")
    parser.add_argument(
        "--f", type=int, help="number of frames to process for prediction"
    )
    parser.add_argument(
        "--d", type=str, help="dataset type, dfdc, faceforensics, timit, celeb"
    )

    parser.add_argument(
        "--w", help="weight for cvit or for cvit2.",
    )
    parser.add_argument("--n", type=str, help="network cvit or cvit2")
    parser.add_argument("--fp16", type=str, help="half precision support")

    args = parser.parse_args()
    path = args.p if args.p else 'sample_prediction_data'
    num_frames = args.f if args.f else 15
    dataset = args.d if args.d else "other"
    net = args.n if args.n in ["cvit", "cvit2"] else "cvit2"
    fp16 = True if args.fp16 else False

    if args.w and net in ['cvit', 'cvit2']:
        cvit_weight = args.w
    else:
        cvit_weight = 'cvit2_deepfake_detection_ep_50'
    
    print(f'\nUsing Model: {net}, \nWeight: {cvit_weight}')

    cvit_weight += '.pth'
    return path, dataset, num_frames, net, fp16, cvit_weight



def main():
    start_time = perf_counter()
    path, dataset, num_frames, net, fp16, cvit_weight = gen_parser()

    # Check if the input is an image or video and handle accordingly
    if dataset in ["dfdc", "faceforensics", "timit", "celeb"]:
        # Call the appropriate function based on dataset type
        result = globals()[dataset](cvit_weight, path, dataset, num_frames, net, fp16)
    else:
        # For 'other' or unrecognized datasets, use the general `vids` function
        result = vids(cvit_weight, path, dataset, num_frames, net, fp16)

    curr_time = datetime.now().strftime("%B_%d_%Y_%H_%M_%S")
    file_path = os.path.join("result", f"prediction_{dataset}_{net}_{curr_time}.json")

    with open(file_path, "w") as f:
        json.dump(result, f)

    end_time = perf_counter()
    print("\n\n--- %s seconds ---" % (end_time - start_time))


if __name__ == "__main__":
    main()