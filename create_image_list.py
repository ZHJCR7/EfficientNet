"""
Extract face image from input video and save the image with the format

Usage:None

Author: Jeffrey Chao
"""
import os
from os.path import join
import cv2
import dlib
from PIL import Image as pil_image
from tqdm import tqdm
import argparse
from pathlib import Path
import pandas as pd

def get_boundingbox(face, width, height, scale=1.3, minsize=None):
    """
    Expects a dlib face to generate a quadratic bounding box.
    :param face: dlib face class
    :param width: frame width
    :param height: frame height
    :param scale: bounding box size multiplier to get a bigger face region
    :param minsize: set minimum bounding box size
    :return: x, y, bounding_box_size in opencv form
    """
    x1 = face.left()
    y1 = face.top()
    x2 = face.right()
    y2 = face.bottom()
    size_bb = int(max(x2 - x1, y2 - y1) * scale)
    if minsize:
        if size_bb < minsize:
            size_bb = minsize
    center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2

    # Check for out of bounds, x-y top left corner
    x1 = max(int(center_x - size_bb // 2), 0)
    y1 = max(int(center_y - size_bb // 2), 0)
    # Check for too big bb size for given x, y
    size_bb = min(width - x1, size_bb)
    size_bb = min(height - y1, size_bb)

    return x1, y1, size_bb

def crop_face_area_from_video(video_path, output_path,start_frame=0, end_frame=None, index=100):

    print('Starting: {}'.format(video_path))

    # Read and write
    reader = cv2.VideoCapture(video_path)

    video_fn = video_path.split('/')[-1].split('.')[0]+'.avi'
    os.makedirs(output_path, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    fps = reader.get(cv2.CAP_PROP_FPS)
    num_frames = int(reader.get(cv2.CAP_PROP_FRAME_COUNT))
    writer = None

    # Face detector
    face_detector = dlib.get_frontal_face_detector()

    # Text variables
    font_face = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 2
    font_scale = 1

    # Frame numbers and length of output video
    frame_num = 0
    assert start_frame < num_frames - 1
    end_frame = end_frame if end_frame else num_frames
    pbar = tqdm(total=end_frame-start_frame)

    while reader.isOpened():
        _, image = reader.read()
        if image is None:
            break
        frame_num += 1

        if frame_num < start_frame:
            continue
        pbar.update(1)

        # Image size
        height, width = image.shape[:2]

        ## Init output writer
        #if writer is None:
        #    writer = cv2.VideoWriter(join(output_path, video_fn), fourcc, fps,
        #                             (height, width)[::-1])

        # 2. Detect with dlib
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_detector(gray, 1)
        img_new = []
        if len(faces):
            # For now only take biggest face
            face = faces[0]

            # --- Prediction ---------------------------------------------------
            # Face crop with dlib and bounding box scale enlargement
            x, y, size = get_boundingbox(face, width, height)
            cropped_face = image[y:y+size, x:x+size]
            img_new = cv2.resize(cropped_face, (300, 300))


        if frame_num >= index:
            break

        #creat the name of the new image
        filepath_img = output_path + "/" + str(frame_num).zfill(4) + '.jpg'

        # Show
        # cv2.imshow(img_new)
        cv2.imwrite(filepath_img, img_new, [int(cv2.IMWRITE_JPEG_QUALITY),95])

    pbar.close()
    print('Finished! Output saved under {}'.format(output_path))

def get_file_list(path):
    file_lst = []

    for f in os.listdir(path):
        if os.path.isfile(os.path.join(path, f)):
            file_lst.append(f)

    return file_lst

def get_video_file_list(dirname, outputpath):
    # filelistlog = dirname + "\\filelistlog.txt"  # 保存文件路径
    postfix = set(['jpg'])  # 设置要保存的文件格式
    for maindir, subdir, file_name_list in os.walk(dirname):
        for filename in file_name_list:
            apath = os.path.join(maindir, filename)
            if True:        # 保存全部文件名。若要保留指定文件格式的文件名则注释该句
                if apath.split('.')[-1] in postfix:   # 匹配后缀，只保存所选的文件格式。若要保存全部文件，则注释该句
                    try:
                        if "manipulated_sequences" in apath:
                            flag = 0
                        else:
                            flag = 1

                        with open(outputpath, 'a+') as fo:
                            fo.writelines(apath.replace("\\", "/"))
                            fo.write(" %d"%(flag))
                            fo.write('\n')
                    except:
                        pass    # 所有异常全部忽略即可

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_source_path', type=str, default='./FaceForensicspp', help='Source Video Path')
    parser.add_argument('--extract_face_path', type=str, default='./output/face', help='Path to save the output face')
    parser.add_argument('--video_pathname_list', type=str, default='./data_list', help='Save the list of video path')

    args = parser.parse_args()

    ## Parameters parsing
    video_source_path = args.video_source_path
    extract_face_path = args.extract_face_path
    video_pathname_list = args.video_pathname_list

    video_path_list_filename = video_pathname_list + "/Img_data_c23_path.txt"

    # if not os.path.exists(extract_face_path):
    #     os.makedirs(extract_face_path)

    if not os.path.exists(video_pathname_list):
        os.makedirs(video_pathname_list)

    if os.path.exists(video_path_list_filename):
        os.remove(video_path_list_filename)

    # get the source path of video
    get_video_file_list(video_source_path, video_path_list_filename)

    # #extract the face and save the image to the file and print the output path list
    # video_path_list = []
    # with open(video_path_list_filename) as read_file:
    #     for line in read_file:
    #         video_path_list.append(line.strip())
    #
    # for i in range(len(video_path_list)):
    #     video_path = video_path_list[i].split(" ")[0].split(".mp4")[0].split("./")[1]
    #     face_path = extract_face_path + "/" + video_path
    #     if not os.path.exists(face_path):
    #         os.makedirs(face_path)
    #     crop_face_area_from_video(video_path_list[i].split(" ")[0], face_path, index= 100)

if __name__ == '__main__':
    main()