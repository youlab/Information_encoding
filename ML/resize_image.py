import argparse
import os
from tqdm import tqdm
from resources.utils import img_resize

# Resize the pattern images to 80*80 before training/testing
def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--input_dir",
        default=None,
        type=str,
        required=True,
        help="Path to the directory where the input data resides."
    )

    parser.add_argument(
        "--image_prefix",
        default="",
        type=str,
        required=True,
        help="Prefix used in naming the inputs."
    )

    parser.add_argument(
        "--image_size",
        default=-1,
        type=int,
        help="Image size after resizing."
    )

    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help="The output directory where we will write the processed data."
    )

    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        print("  Output directory does not exist")
        print("  Creating the directory at: {}".format(args.output_dir))
        os.mkdir(args.output_dir)

    image_names = [name for name in os.listdir(args.input_dir) if name.startswith(args.image_prefix)]
    print("  Initiating image resizing.")
    print("  Input dir: {}".format(args.input_dir))
    print("  Output dir: {}".format(args.output_dir))

    for image_name in tqdm(image_names, total=len(image_names), desc="Resizing images"):
        img_resize(
            in_path=os.path.join(args.input_dir, image_name),
            out_path=os.path.join(args.output_dir, image_name),
            image_size=args.image_size
        )

    print("  DONE: All images have been processed.")


if __name__ == "__main__":
    main()
