import glob
import cv2
import numpy as np
import random



def mosaic_data_augmentation(images, masks, output_size=(512, 512)):
    random.seed(random.randint(0,1000))
    random.shuffle(images)
    random.shuffle(masks)
    
    mosaic_img = np.zeros((output_size[0], output_size[1], 3), dtype=np.uint8)
    mosaic_mask = np.zeros((output_size[0], output_size[1], 3), dtype=np.uint8)
    
    # Randomly determine the sizes of each quadrant
    cut_x = random.randint(output_size[1] // 4, 3 * output_size[1] // 4)
    cut_y = random.randint(output_size[0] // 4, 3 * output_size[0] // 4)

    indices = random.sample(range(len(images)), 4)  # Randomly select 4 images

    # Top-left
    img = cv2.resize(images[indices[0]], (cut_x, cut_y))
    mask = cv2.resize(masks[indices[0]], (cut_x, cut_y), interpolation=cv2.INTER_NEAREST)
    mosaic_img[:cut_y, :cut_x] = img
    mosaic_mask[:cut_y, :cut_x] = mask

    # Top-right
    img = cv2.resize(images[indices[1]], (output_size[1] - cut_x, cut_y))
    mask = cv2.resize(masks[indices[1]], (output_size[1] - cut_x, cut_y), interpolation=cv2.INTER_NEAREST)
    mosaic_img[:cut_y, cut_x:] = img
    mosaic_mask[:cut_y, cut_x:] = mask

    # Bottom-left
    img = cv2.resize(images[indices[2]], (cut_x, output_size[0] - cut_y))
    mask = cv2.resize(masks[indices[2]], (cut_x, output_size[0] - cut_y), interpolation=cv2.INTER_NEAREST)
    mosaic_img[cut_y:, :cut_x] = img
    mosaic_mask[cut_y:, :cut_x] = mask

    # Bottom-right
    img = cv2.resize(images[indices[3]], (output_size[1] - cut_x, output_size[0] - cut_y))
    mask = cv2.resize(masks[indices[3]], (output_size[1] - cut_x, output_size[0] - cut_y), interpolation=cv2.INTER_NEAREST)
    mosaic_img[cut_y:, cut_x:] = img
    mosaic_mask[cut_y:, cut_x:] = mask

    return mosaic_img, mosaic_mask