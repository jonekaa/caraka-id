import os
import cv2


def remove_exceeded_files(ROOT_FOLDERS):
    for folders in os.listdir(ROOT_FOLDERS):
        folder = os.path.join(ROOT_FOLDERS, folders)

        image_files = [file for file in os.listdir(folder) if file.lower().endswith(('.jpg', '.jpeg', '.png'))]
        image_files.sort()

        if len(image_files) > 350:
            excess_image = image_files[350:]
            for image_file in excess_image:
                os.remove(os.path.join(folder, image_file))


def resize_image(ROOT_FOLDERS):
    for folders in os.listdir(ROOT_FOLDERS):
        folder = os.path.join(ROOT_FOLDERS, folders)

        for filenames in os.listdir(folder):
            filename = os.path.join(folder, filenames)

            image = cv2.imread(filename)
            resized_image = cv2.resize(image, (150, 150))

            cv2.imwrite(str(filename), resized_image)


ROOT_FOLDERS = '../data/Aksara-Lampung'
remove_exceeded_files(ROOT_FOLDERS)
resize_image(ROOT_FOLDERS)
