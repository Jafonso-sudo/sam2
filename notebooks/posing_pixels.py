
import argparse
import ast
import os
from PIL import Image
import numpy as np
import torch
from sam2.build_sam import build_sam2_video_predictor
from tqdm import tqdm

def parse_arguments():
    # Initialize argument parser
    parser = argparse.ArgumentParser(description="Posing Pixels Script")  # noqa: F821

    # Required positional arguments
    parser.add_argument('input_dir', type=str, help='Input directory path')
    parser.add_argument('output_dir', type=str, help='Output directory path')

    # Optional prompt argument
    parser.add_argument(
        '-p', '--prompts', 
        type=str, 
        help='Optional list of 2D coordinates in the form [(x1, y1), (x2, y2), ...]'
    )

    # Parse arguments
    args = parser.parse_args()

    # Process prompts if they are passed
    if args.prompts:
        try:
            # Safely evaluate the string as a list of tuples
            prompts = ast.literal_eval(args.prompts)
            
            if not isinstance(prompts, list):
                raise ValueError("Prompts should be a list of tuples")

            for prompt in prompts:
                if not (isinstance(prompt, tuple) and len(prompt) == 2):
                    raise ValueError("Each prompt must be a tuple of two coordinates")

            args.prompts = prompts
        except (SyntaxError, ValueError) as e:
            raise ValueError(f"Invalid format for prompts: {e}")
    else:
        # Set default value if prompts not passed
        args.prompts = []

    return args

def segment_and_save(input, output, prompts):
    print(os.path.dirname(os.path.realpath(__file__)))
    checkpoint = "/home/joao/Documents/repositories/segment-anything-2/checkpoints/sam2.1_hiera_large.pt"
    model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
    predictor = build_sam2_video_predictor(model_cfg, checkpoint)

    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        state = predictor.init_state(input)
        
        # if there's no prompts, prompt middle of the image
        if not prompts:
            print("No prompts provided. Using the center of the image as prompt.")
            frame_names = [
                p for p in os.listdir(input)
                if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
            ]
            frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))
            video_frame = Image.open(os.path.join(input, frame_names[0]))
            height, width = video_frame.size
            prompts = [
                (height // 2, width // 2)
            ]
        prompts = np.array(prompts, dtype=np.float32)
        labels = np.ones(len(prompts))

        # add new prompts and instantly get the output on the same frame
        frame_idx, object_ids, masks = predictor.add_new_points_or_box(state, frame_idx=0, obj_id=0, points=prompts, labels=labels)

        # propagate the prompts to get masklets throughout the video
        probs = []
        for frame_idx, object_ids, masks in predictor.propagate_in_video(state):
            prob_mask = torch.sigmoid(masks[0][0])
            mask_image = Image.fromarray((prob_mask * 255).byte().cpu().numpy())
            mask_image.save(os.path.join(output, f"{frame_idx}.png"))
            probs.append(prob_mask)
            
        # save probs as npy
        torch.save(probs, os.path.join(output, "mask.npy"))

if __name__ == "__main__":
    args = parse_arguments()

    print(f"Input Directory: {args.input_dir}")
    print(f"Output Directory: {args.output_dir}")
    print(f"Prompts: {args.prompts}")
    
    segment_and_save(args.input_dir, args.output_dir, args.prompts)