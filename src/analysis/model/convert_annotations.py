import os
import cv2

def load_classes(classes_path):
    with open(classes_path, 'r') as f:
        return [line.strip() for line in f if line.strip()]

def convert_annotations(annotations_path, images_dir, labels_dir, classes_path):
    classes = load_classes(classes_path)
    os.makedirs(labels_dir, exist_ok=True)
    with open(annotations_path, 'r') as f:
        lines = f.readlines()
    for line in lines:
        parts = line.strip().split(',')
        # Format: filename,xmin,ymin,xmax,ymax,class_name
        if len(parts) != 6:
            continue  # skip header or malformed lines
        img_name, xmin, ymin, xmax, ymax, class_name = parts
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
        class_id = classes.index(class_name)
        label_path = os.path.join(labels_dir, os.path.splitext(img_name)[0] + '.txt')
        with open(label_path, 'a') as lf:
            lf.write(f"{class_id} {x_center} {y_center} {width} {height}\n")
    print(f"Conversion complete! YOLO labels saved in {labels_dir}")

# Example usage for all splits
def auto_convert_all_splits():
    for split in ['train', 'valid', 'test']:
        images_dir = f'data/yolo/images/{split}'
        annotations_path = os.path.join(images_dir, '_annotations.txt')
        classes_path = os.path.join(images_dir, '_classes.txt')
        labels_dir = f'data/yolo/labels/{split}'
        convert_annotations(annotations_path, images_dir, labels_dir, classes_path)
