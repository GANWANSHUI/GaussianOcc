import os
import cv2
import argparse
import pdb
from tools.export_occupancy_vis import visualize_occ_dict


def create_video_from_images(input_folder, sem_only=False):

    # pdb.set_trace()

    rgb_depth_occ_path = input_folder + '_vis'

    os.makedirs(rgb_depth_occ_path, exist_ok=True)

    # output_video = rgb_depth_occ_path + '\\.avi'

    output_video = os.path.join(rgb_depth_occ_path, 'video.avi')

    image_files = [f for f in os.listdir(input_folder) if f.endswith(".jpg")]
    image_files.sort()

    frame = cv2.imread(os.path.join(input_folder, image_files[0]))
    height, width, _ = frame.shape

    fourcc = cv2.VideoWriter_fourcc(*'I420')

    if 'ddad' in input_folder:
        num_frame = len(image_files)
        fps = 8.0
    else:
        num_frame = len(image_files) // 8

        fps = 12.0

    if sem_only:
        out = cv2.VideoWriter(output_video, fourcc, fps, (width, height * 6))
    else:
        out = cv2.VideoWriter(output_video, fourcc, fps, (width, height * 3))

    # pdb.set_trace()
    # if 'ddad' in input_folder:
    #     num_frame = len(image_files)
    # else:
    #     num_frame = len(image_files) // 8


    for i in range(num_frame):

        # image_up = cv2.imread(os.path.join(input_folder, os.path.join(input_folder, '{:03d}-up.jpg'.format(i))))
        # image_down = cv2.imread(os.path.join(input_folder, os.path.join(input_folder, '{:03d}-down.jpg'.format(i))))

        image_up = cv2.imread(os.path.join(input_folder, '{:03d}-up.jpg'.format(i)))
        image_down = cv2.imread(os.path.join(input_folder, '{:03d}-down.jpg'.format(i)))


        occ_f = os.path.join(input_folder, '{:03d}-out.npy-front.jpg'.format(i))
        occ_fl = os.path.join(input_folder, '{:03d}-out.npy-front_left.jpg'.format(i))
        occ_fr = os.path.join(input_folder, '{:03d}-out.npy-front_right.jpg'.format(i))
        occ_up = cv2.hconcat([cv2.imread(occ_fl), cv2.imread(occ_f), cv2.imread(occ_fr)])
        occ_b = os.path.join(input_folder, '{:03d}-out.npy-back.jpg'.format(i))
        occ_bl = os.path.join(input_folder, '{:03d}-out.npy-back_left.jpg'.format(i))
        occ_br = os.path.join(input_folder, '{:03d}-out.npy-back_right.jpg'.format(i))
        occ_down = cv2.hconcat([cv2.imread(occ_bl), cv2.imread(occ_b), cv2.imread(occ_br)])

        # pdb.set_trace()
        # frame = cv2.vconcat([image_up, occ_up, image_down, occ_down])

        # depth
        depth_up = cv2.imread(os.path.join(input_folder, '{:03d}-depth-up.jpg'.format(i)))
        depth_down = cv2.imread(os.path.join(input_folder, '{:03d}-depth-down.jpg'.format(i)))


        frame = cv2.vconcat([image_up, depth_up, occ_up, image_down, depth_down, occ_down])

        rgb_depth_occ_img_path = os.path.join(rgb_depth_occ_path, '{:03d}.jpg'.format(i))

        # 保存图像
        try:
            cv2.imwrite(rgb_depth_occ_img_path, frame)
            out.write(frame)
        except:
            pass

            # pdb.set_trace()

    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a video from jpg images in a folder")
    parser.add_argument("input_folder", help="Input folder containing jpg images")
    parser.add_argument("--sem_only", action="store_true")
    args = parser.parse_args()
    
    dict_list = os.listdir(args.input_folder)
    dict_list.sort()

    for dict_name in dict_list:
        # pdb.set_trace()
        if dict_name.endswith('.npy'):
            # if dict_name[:3] == '006':
            print('proceding:', dict_name)
            visualize_occ_dict(os.path.join(args.input_folder, dict_name), offscreen=False, render_w=320)

    create_video_from_images(args.input_folder, sem_only=args.sem_only)

    print('finish!')

# python gen_scene_videos.py logs_scene_path