import os
import shutil


DATA_ALL_DIR = os.path.join('.', 'data_all')

DATA_OUT_DIR = os.path.join('.', 'data')

for set_ in ['train', 'validation', 'test']:
    for dir_ in [os.path.join(DATA_OUT_DIR, set_),
                 os.path.join(DATA_OUT_DIR, set_, 'imgs'),
                 os.path.join(DATA_OUT_DIR, set_, 'anns')]:
        if os.path.exists(dir_):
            shutil.rmtree(dir_)
        os.mkdir(dir_)

alpaca_id = '/m/0pcr'

train_bboxes_filename = os.path.join('.', 'oidv6-train-annotations-bbox.csv')
validation_bboxes_filename = os.path.join('.', 'validation-annotations-bbox.csv')
test_bboxes_filename = os.path.join('.', 'test-annotations-bbox.csv')

max_images = 500  # Set image limit
image_count = 0  # Track number of images copied

for j, filename in enumerate([train_bboxes_filename, validation_bboxes_filename, test_bboxes_filename]):
    set_ = ['train', 'validation', 'test'][j]  # Assign train/val/test based on index
    print(filename)

    with open(filename, 'r') as f:
        line = f.readline()
        while len(line) != 0 and image_count < max_images:  # Stop if 500 images are reached
            id, _, class_name, _, x1, x2, y1, y2, _, _, _, _, _ = line.split(',')[:13]

            if class_name in [alpaca_id]:  # If image contains an Alpaca
                if not os.path.exists(os.path.join(DATA_OUT_DIR, set_, 'imgs', f"{id}.jpg")):
                    shutil.copy(
                        os.path.join(DATA_ALL_DIR, f"{id}.jpg"),
                        os.path.join(DATA_OUT_DIR, set_, 'imgs', f"{id}.jpg")
                    )
                    image_count += 1  # Increment image count

                with open(os.path.join(DATA_OUT_DIR, set_, 'anns', f"{id}.txt"), 'a') as f_ann:
                    # Convert COCO to YOLO format
                    x1, x2, y1, y2 = [float(j) for j in [x1, x2, y1, y2]]
                    xc = (x1 + x2) / 2
                    yc = (y1 + y2) / 2
                    w = x2 - x1
                    h = y2 - y1

                    f_ann.write(f"0 {xc} {yc} {w} {h}\n")

            line = f.readline()

        f.close()  # Close the annotation file after processing
