import os
import cv2
import time
import yaml
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import albumentations as A
import multiprocessing

import torch
from torchvision import transforms

class DataProcessor:
    def __init__(self, config_path):
        """Khởi tạo DataProcessor với đường dẫn config."""
        with open(config_path) as f:
            self.config = yaml.load(f, Loader=yaml.FullLoader)

        self.data = None
        self.resize_shape = self.config['preprocessing']['resize']
        self.images_path = self.config['data']['raw_images_path']
        self.batch_size = self.config['hyperparameters']['batch_size']
        self.n_workers = multiprocessing.cpu_count()
        
    def load_raw_data(self):
        """Đọc dữ liệu từ file CSV."""
        raw_csv_path = self.config['data']['raw_csv_path']
        self.data = pd.read_csv(raw_csv_path)

    def data_check(self):
        """Kiểm tra dữ liệu để phát hiện các vấn đề."""
        print('Data shape:', self.data.shape)
        print()
        # Show data types
        print('Data types:')
        print(self.data.dtypes)
        print()

        # Show unique values for each column (skip image_id and lesion_id)
        print('Unique values for each column:')
        for col in self.data.columns:
            if col in ['image_id', 'lesion_id']:
                continue
            print(f"{col}: {self.data[col].unique()[:5]}...")
        print()

        # Count NaN values
        print('NaN values in each column:')
        nan_counts = self.data.isnull().sum()
        print(nan_counts[nan_counts > 0])
        print()

        # Count 'unknown' values (only for object columns)
        print('Unknown values in each column:')
        for col in self.data.select_dtypes(include=['object']).columns:
            unknown_count = self.data[self.data[col] == 'unknown'].shape[0]
            if unknown_count > 0:
                print(f"{col}: {unknown_count}")
        print()

        # Summary
        print("Summary of potential data issues:")
        print(f"- Columns with NaN values: {nan_counts[nan_counts > 0].index.tolist()}")
        print(f"- Columns with 'unknown' values: {[col for col in self.data.columns if 'unknown' in self.data[col].values]}")
        

    def data_cleaning(self):
        """Tiền xử lý dữ liệu bằng cách làm sạch và mã hóa."""
        self.data.drop_duplicates(subset=['image_id'], inplace=True)
        self.data.dropna(subset=['age', 'sex', 'localization'], inplace=True)

        self.data = self.data[self.data['age'] != 'unknown']
        self.data = self.data[self.data['sex'] != 'unknown']
        self.data = self.data[self.data['localization'] != 'unknown']

        # Chuyển đổi 'age' sang int
        self.data['age'] = self.data['age'].astype(int)

        # Mã hóa nhãn
        self._encode_labels()
        
        # Thêm đường dẫn của ảnh
        self.data['image_id'] = self.data['image_id'].apply(lambda x: os.path.join(self.images_path, x + '.jpg'))
        
        print('Data after cleaning:')
        print(self.data.head())

        

    def _encode_labels(self):
        """Mã hóa các nhãn cho các cột danh mục."""
        label_encoder_dx = LabelEncoder()
        self.data['dx_code'] = label_encoder_dx.fit_transform(self.data['dx'])
        
        label_encoder_dx_type = LabelEncoder()
        self.data['dx_type_code'] = label_encoder_dx_type.fit_transform(self.data['dx_type'])
        
        label_encoder_sex = LabelEncoder()
        self.data['sex_code'] = label_encoder_sex.fit_transform(self.data['sex'])
        
        label_encoder_localization = LabelEncoder()
        self.data['localization_code'] = label_encoder_localization.fit_transform(self.data['localization'])

        """Lưu trữ các label encoder."""
        self.label_encoders = {
            'dx': label_encoder_dx,
            'dx_type': label_encoder_dx_type,
            'sex': label_encoder_sex,
            'localization': label_encoder_localization
        }
        encoder_file_name = self.config['encoder']['path'] + str(self.config['preprocessing']['resize'][0]) + 'x' + str(self.config['preprocessing']['resize'][1]) + '.pt'
        torch.save(self.label_encoders, encoder_file_name)


    def split_data(self, test_size=0.2, random_state=42):
        """Chia tách dữ liệu thành tập huấn luyện và tập kiểm tra."""
        train_data, test_data = train_test_split(self.data, test_size=test_size, random_state=random_state)
        return train_data.reset_index(drop=True), test_data.reset_index(drop=True)

    def load_images_worker(self, batch):
        """Worker để đọc ảnh trong một batch."""
        images = []
        for index, row in batch.iterrows():
            image_path = row['image_id']
            image = cv2.imread(image_path)
            if image is None:
                print('Image not found: ', image_path)
                continue
            image = cv2.resize(image, (self.config['preprocessing']['resize'][0], self.config['preprocessing']['resize'][1]))
            images.append(image)
        return images

    def load_images(self, data):
        """Đọc tất cả ảnh từ dữ liệu."""
        all_images = []
        batches = [data[i:i + self.batch_size] for i in range(0, len(data), self.batch_size)]
        with ThreadPoolExecutor(max_workers=self.n_workers) as executor:
            futures = {executor.submit(self.load_images_worker, batch): batch for batch in batches}
            for future in tqdm(as_completed(futures), total=len(futures)):
                img_batch = future.result()
                if img_batch is not None:
                    all_images.extend(img_batch)
        return all_images
    
    def type_count(self, data, rate):
        lesion_type = data.groupby(data['dx']).count()
        lesion_type = lesion_type.reset_index()
        lesion_type = lesion_type[['dx', 'image_id']]
        lesion_type.columns = ['dx', 'count']
        lesion_type = lesion_type.sort_values(by='count', ascending=False)
        print(lesion_type)
        max_count = lesion_type['count'].max()
        augmentation_counts = {}
        for index, row in lesion_type.iterrows():
            augmentation_counts[row['dx']] = int(max_count / row['count'] * rate)
        return augmentation_counts
    
    def augment_image_worker(self, row, image, augmentations, augmentation_counts):
        augmented_images = []
        augmented_data = []

        lesion_type = row['dx']
        count = augmentation_counts.get(lesion_type, 0)
        if count == 0:
            return None, None

        for _ in range(count):
            if _ == 0:
                augmented_images.append(image)
                augmented_row = row.copy()
                augmented_row['image_id'] = row['image_id'] + '_original'
                augmented_data.append(augmented_row)
                continue

            augment_image = augmentations(image=image)
            augmented_images.append(augment_image['image'])
            augmented_row = row.copy()
            augmented_row['image_id'] = row['image_id'] + '_augmented_' + str(_)
            augmented_data.append(augmented_row)

        return augmented_data, augmented_images

    def augment_images(self, data, images_array, augmentation_counts):
        augmentations = A.Compose([
            A.HorizontalFlip(p=0.25),
            A.VerticalFlip(p=0.25),
            A.Rotate(limit=(0, 180), p=0.25),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        ])

        print(f'Number of images: {len(images_array)}, Number of rows in data: {len(data)}')

        augmented_images = []
        augmented_data = []

        with ThreadPoolExecutor(max_workers=self.n_workers) as executor:
            futures = []

            for index, row in tqdm(data.iterrows(), total=len(data), desc="Augmenting Images"):
                futures.append(
                    executor.submit(self.augment_image_worker, row, images_array[index], augmentations, augmentation_counts)
                )

            for future in tqdm(as_completed(futures), total=len(futures), desc="Collecting Results"):
                aug_data, aug_images = future.result()
                if aug_data is not None and aug_images is not None:
                    augmented_data.extend(aug_data)
                    augmented_images.extend(aug_images)

        augmented_data = pd.DataFrame(augmented_data)

        return augmented_data, augmented_images
    
    def convert_to_tensor(self, image):
        """Chuyển đổi ảnh thành tensor."""
        to_tensor = transforms.ToTensor()
        return to_tensor(image)
    
    def save_data(self, data, images, file_suffix):
        """Lưu dữ liệu và ảnh vào file."""
        # Convert images to tensors
        augmented_images = [self.convert_to_tensor(image) for image in images]

        # Tạo đường dẫn tệp tin
        suffix = str(self.resize_shape[0]) + 'x' + str(self.resize_shape[1]) + '_' + file_suffix
        data_path = os.path.join(self.config['data']['processed_path'], f'HAM10000_{suffix}.pt')
        images_path = os.path.join(self.config['data']['processed_images_path'], f'images_{suffix}.pt')

        # Tạo thư mục nếu nó không tồn tại
        os.makedirs(os.path.dirname(data_path), exist_ok=True)  # Tạo thư mục cho dữ liệu
        os.makedirs(os.path.dirname(images_path), exist_ok=True)  # Tạo thư mục cho hình ảnh

        # Lưu dữ liệu
        torch.save(data, data_path)
        torch.save(augmented_images, images_path)

    def run(self):
        """Chạy toàn bộ quy trình xử lý dữ liệu."""
        start_time = time.time()
        self.load_raw_data()
        self.data_check()
        self.data_cleaning()
        train_data, test_data = self.split_data()
        train_images = self.load_images(train_data)
        test_images = self.load_images(test_data)

        augmentation_counts = self.type_count(train_data, self.config['preprocessing']['augmentation']['ratio'])
        augmented_train_data, augmented_train_images = self.augment_images(train_data, train_images, augmentation_counts)
        

        print('Augmented train data shape:', augmented_train_data.shape)
        print('Raw test data shape:', test_data.shape)

        # Lưu dữ liệu đã tăng cường
        self.save_data(augmented_train_data, augmented_train_images, 'train')
        self.save_data(test_data, test_images, 'test')
        print('Data processing completed in', round(time.time() - start_time, 2), 'seconds.')
        print('Data saved successfully.')

if __name__ == "__main__":
    processor = DataProcessor(config_path='config.yaml')
    processor.run()

"""
Giải thích cấu trúc:
Khởi tạo class:

Class DataProcessor nhận vào đường dẫn đến file cấu hình (config.yaml).
Các phương thức chính:

load_raw_data(): Đọc dữ liệu từ file CSV.
data_check(): Kiểm tra dữ liệu và in ra các vấn đề tiềm ẩn.
data_cleaning(): Làm sạch dữ liệu, mã hóa nhãn và thêm đường dẫn của ảnh.
split_data(): Chia tách dữ liệu thành tập huấn luyện và tập kiểm tra.
load_images_worker(): Đọc ảnh từ một batch.
load_images(): Đọc tất cả ảnh từ tập dữ liệu.
type_count(): Tính số lần tăng cường cho từng loại bệnh.
augment_image_worker(): Tăng cường ảnh cho một hàng.
augment_images(): Tăng cường ảnh cho tất cả dữ liệu.
convert_to_tensor(): Chuyển ảnh thành tensor.
save_data(): Lưu dữ liệu và ảnh vào file.
run(): Chạy toàn bộ quy trình xử lý dữ liệu.
"""