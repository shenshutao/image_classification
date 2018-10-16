from cnn_model import util
import os
import shutil
import hashlib
import uuid
import subprocess


def is_same_image(image_path1, image_path2):
    hash_md5 = hashlib.md5()
    with open(image_path1, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    md5_1 = hash_md5.hexdigest()

    hash_md5_2 = hashlib.md5()
    with open(image_path2, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5_2.update(chunk)
    md5_2 = hash_md5_2.hexdigest()

    if md5_1 == md5_2:
        return True
    else:
        return False


def merger_folder(merge_zip_list, target_folder_name):
    if not os.path.exists(target_folder_name):
        os.makedirs(target_folder_name)

    for i, zip_file_path in enumerate(merge_zip_list):
        if zip_file_path.startswith('gs://'):
            local_zip_file_path = 'train_data' + str(i) + '.zip'
            subprocess.check_call([
                'gsutil', '-m', '-q', 'cp', '-r', zip_file_path, local_zip_file_path
            ])
            zip_file_path = local_zip_file_path

        util.extract_zipfile(zip_file_path, 'tmp_folder' + str(i))
        merged_class_folder_list = os.listdir(target_folder_name)
        class_folder_list = os.listdir('tmp_folder' + str(i))

        for class_folder in class_folder_list:
            if class_folder not in merged_class_folder_list:  # class folder not exist in merged folder
                shutil.copytree(os.path.join('tmp_folder' + str(i), class_folder),
                                os.path.join(target_folder_name, class_folder))
            else:  # class folder is already inside the merged folder, do merge !
                for image_name in os.listdir(os.path.join('tmp_folder' + str(i), class_folder)):
                    if image_name not in os.listdir(os.path.join(target_folder_name, class_folder)):
                        shutil.copy(os.path.join('tmp_folder' + str(i), class_folder, image_name),
                                    os.path.join(target_folder_name, class_folder, image_name))
                    else:
                        if is_same_image(os.path.join('tmp_folder' + str(i), class_folder, image_name),
                                         os.path.join(target_folder_name, class_folder, image_name)):
                            pass  # do not copy the image
                        else:
                            shutil.copy(os.path.join('tmp_folder' + str(i), class_folder, image_name),
                                        os.path.join(target_folder_name, class_folder, str(uuid.uuid4()) + image_name))

        shutil.rmtree('tmp_folder' + str(i))


def get_data(input_path: str, output_folder):
    if input_path.startswith('Merged;'):
        merge_list = input_path.replace('Merged;', '').split(',')
        merger_folder(merge_list, output_folder)
    elif input_path.startswith('gs://'):
        subprocess.check_call([
            'gsutil', '-m', '-q', 'cp', '-r', input_path, 'train_data.zip'
        ])
        util.extract_zipfile('train_data.zip', output_folder)
    else:
        util.extract_zipfile(input_path, output_folder)

    # Just for Mac user ... delete all the .DS_Store files",
    for arg, dirname, names in os.walk(output_folder):
        if '.DS_Store' in names:
            os.remove(os.path.join(arg, '.DS_Store'))


# Only for Test
if __name__ == "__main__":
    # merge_list = ['aaa.zip', 'bbb.zip']

    li = 'Merged;C:/Users/AlphaCat/Desktop/test.zip,C:/Users/AlphaCat/Desktop/test2.zip'
    merge_list = li.replace('Merged;', '').split(',')

    merger_folder(merge_list, 'folder_merged')
