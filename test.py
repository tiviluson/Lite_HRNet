import requests
import numpy as np
import matplotlib.pyplot as plt
import argparse


def main(args):
    url = f"{args.ip}/predict"

    headers = {
        "accept": "application/json",
        # "Content-Type": "multipart/form-data"
    }
    files = {
        "file": open(args.image, "rb")
    }

    response = requests.post(url, headers=headers, files=files).json()

    keypoint_boxes, image = np.array(
        response["keypoints"]), np.array(response["resized_img"])
    keypoints = keypoint_boxes.reshape(-1, 3)
    # The model resizes and processes the resized image, so we need to scale the keypoints back to the original image size
    if args.resize:
        original_image = plt.imread(args.image)
        scale = np.array([original_image.shape[1] / image.shape[1],
                          original_image.shape[0] / image.shape[0]])
        image = original_image
        keypoints[:, :2] *= scale
    if args.plt or args.output:
        plt.imshow(image)
        scatter = plt.scatter(
            keypoints[:, 0], keypoints[:, 1], c=keypoints[:, 2], s=5)
        plt.colorbar(scatter, label='Confidence Score')
        if args.output is not None:
            plt.savefig(args.output)
        if args.plt:
            plt.show()
    if args.verbose == 1:
        print("Keypoints:\n", keypoint_boxes)
    elif args.verbose == 2:
        print("Keypoints:\n", keypoint_boxes)
        print("Keypoins Shape:", keypoint_boxes.shape)
        print("Reponse image Shape:", image.shape)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", "-i", type=str,
                        default="sample_img.jpg", help="Path to image file")
    parser.add_argument("--ip", type=str, default="http://127.0.0.1:8000",
                        help="URL of the server, default at localhost:8000")
    parser.add_argument("--plt", "-p", action='store_true',
                        help='Plot the image with keypoints')
    parser.add_argument("--output", "-o", type=str,
                        help='Output file name to save the plot')
    parser.add_argument("--verbose", "-v", type=int, default=1, choices=[0, 1, 2],
                        help="Print the keypoints (verbosity from 0 to 2)")
    parser.add_argument("--resize", "-r", action='store_true',
                        help="Resize the output to the original image size")
    args = parser.parse_args()
    main(args)
