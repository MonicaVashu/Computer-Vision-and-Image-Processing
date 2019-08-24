"""
Character Detection
(Due date: March 8th, 11: 59 P.M.)

The goal of this task is to experiment with template matching techniques. Specifically, the task is to find ALL of
the coordinates where a specific character appears using template matching.

There are 3 sub tasks:
1. Detect character 'a'.
2. Detect character 'b'.
3. Detect character 'c'.

You need to customize your own templates. The templates containing character 'a', 'b' and 'c' should be named as
'a.jpg', 'b.jpg', 'c.jpg' and stored in './data/' folder.

Please complete all the functions that are labelled with '# TODO'. Whem implementing the functions,
comment the lines 'raise NotImplementedError' instead of deleting them. The functions defined in utils.py
and the functions you implement in task1.py are of great help.

Hints: You might want to try using the edge detectors to detect edges in both the image and the template image,
and perform template matching using the outputs of edge detectors. Edges preserve shapes and sizes of characters,
which are important for template matching. Edges also eliminate the influence of colors and noises.

Do NOT modify the code provided.
Do NOT use any API provided by opencv (cv2) and numpy (np) in your code.
Do NOT import any library (function, module, etc.).
"""
import argparse
import json
import os
import utils
from task1 import *   # you could modify this line


def parse_args():
    parser = argparse.ArgumentParser(description="cse 473/573 project 1.")
    parser.add_argument(
        "--img_path", type=str, default="./data/characters.jpg",
        help="path to the image used for character detection (do not change this arg)")
    parser.add_argument(
        "--template_path", type=str, default="",
        choices=["./data/a.jpg", "./data/b.jpg", "./data/b_uppercase.jpg","./data/c.jpg", "./data/c_small_italics.jpg","./data/c_uppercase.jpg","./data/c_italics.jpg","./data/A_uppercase.jpg","./data/A_italics.jpg", "./data/b_uppercase_italics.jpg"],
        help="path to the template image")
    parser.add_argument(
        "--result_saving_directory", dest="rs_directory", type=str, default="./results/",
        help="directory to which results are saved (do not change this arg)")
    args = parser.parse_args()
    return args


def detect(img, template, template_number):
    """Detect a given character, i.e., the character in the template image.

    Args:
        img: nested list (int), image that contains character to be detected.
        template: nested list (int), template image.

    Returns:
        coordinates: list (tuple), a list whose elements are coordinates where the character appears.
            format of the tuple: (x (int), y (int)), x and y are integers.
            x: row that the character appears (starts from 0).
            y: column that the character appears (starts from 0).
    """
    # TODO: implement this function.
    # raise NotImplementedError
    img_r = np.array(img).shape[0]
    img_d = np.array(img).shape[1]
    template_r = np.array(template).shape[0]
    template_d = np.array(template).shape[1]

    coordinates=[]
    if template_number == 1 or template_number == 5:
        for i in range(0, img_r):
            for j in range(0, img_d):
                ff = False
                sum_ = 0
                for m in range(0, template_r):
                    f = False
                    for n in range(0, template_d):
                        if i + m < img_r and j + n < img_d and int(img[i + m][j + n])-int(template[m][n]) > 150:
                            f = True
                            break
                    if f:
                        ff = True
                        break
                if not ff and i < 655 and j < 1059:
                    coordinates.append((j, i))
    elif template_number == 2:
        for i in range(0,img_r):
            for j in range(0,img_d):
                ff=False
                sum_=0
                for m in range(0,template_r):
                    f = False
                    for n in range(0,template_d):
                        if i+m<img_r and j+n<img_d :
                            sum_ += (int(img[i + m][j + n])-int(template[m][n]))**2
                if sum_<275000:
                    coordinates.append((j,i))
    elif template_number == 3:
        for i in range(0,img_r):
            for j in range(0,img_d):
                ff = False
                sum_=0
                for m in range(0,template_r):
                    f = False
                    for n in range(0,template_d):
                        if i+m < img_r and j+n < img_d:
                            sum_ += (int(img[i + m][j + n])-int(template[m][n]))**2
                if sum_ < 750000:
                    coordinates.append((j,i))
    elif template_number == 4:
        for i in range(0, img_r):
            for j in range(0, img_d):
                ff = False
                sum_ = 0
                for m in range(0, template_r):
                    f = False
                    for n in range(0, template_d):
                        if i + m < img_r and j + n < img_d:
                            sum_ += (int(img[i + m][j + n]) - int(template[m][n])) ** 2
                if sum_ < 250000:
                    coordinates.append((j, i))

    img = np.array(img)
    for pt in coordinates:
        if pt[0] + template_d<1059 and pt[1] + template_r<655:
            cv2.rectangle(img, pt, (pt[0] + template_d, pt[1] + template_r), (0, 255, 255), 1)

    cv2.imwrite("./results/x.jpg",img)
    return coordinates

def save_results(coordinates, template, template_name, rs_directory):
    results = {}
    results["coordinates"] = sorted(coordinates, key=lambda x: x[0])
    results["templat_size"] = (len(template), len(template[0]))
    with open(os.path.join(rs_directory, template_name), "w") as file:
        json.dump(results, file)

def main():
    args = parse_args()

    template_number = 0
    if args.template_path == './data/a.jpg' or args.template_path=='./data/b.jpg' or args.template_path == './data/A_uppercase.jpg' or args.template_path == './data/b_uppercase_italics.jpg':
        template_number = 1
    elif args.template_path == './data/c.jpg':
        template_number = 2
    elif args.template_path == './data/A_italics.jpg':
        template_number = 3
    elif args.template_path == './data/c_small_italics.jpg':
        template_number = 4
    elif args.template_path == './data/c_uppercase.jpg':
        template_number = 5

    img = read_image(args.img_path)
    template = read_image(args.template_path)

    coordinates = detect(img, template,template_number)

    template_name = "{}.json".format(os.path.splitext(os.path.split(args.template_path)[1])[0])
    save_results(coordinates, template, template_name, args.rs_directory)


if __name__ == "__main__":
    main()
