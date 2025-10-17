import os
import cv2

def convert_annotations(annotations_path, images_dir, labels_dir):
    os.makedirs(labels_dir, exist_ok=True)
    with open(annotations_path, 'r') as f:
        lines = f.readlines()
    for line in lines:
        parts = line.strip().split(',')
        if len(parts) != 6:
            continue  # skip malformed lines
        img_name, xmin, ymin, xmax, ymax, class_id = parts
        img_path = os.path.join(images_dir, img_name)
        if not os.path.exists(img_path):
            print(f"Image not found: {img_path}")
            continue
        img = cv2.imread(img_path)
        h, w = img.shape[:2]
        xmin, ymin, xmax, ymax = map(float, [xmin, ymin, xmax, ymax])
        x_center = ((xmin + xmax) / 2) / w
        y_center = ((ymin + ymax) / 2) / h
        width = (xmax - xmin) / w
        height = (ymax - ymin) / h
        label_path = os.path.join(labels_dir, os.path.splitext(img_name)[0] + '.txt')
        with open(label_path, 'a') as lf:
            lf.write(f"{class_id} {x_center} {y_center} {width} {height}\n")
    print(f"Conversion complete. Labels saved to {labels_dir}")
