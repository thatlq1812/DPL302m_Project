import os
import time
import yaml
import multiprocessing
import cv2
import torch
import pandas as pd
import numpy as np
import albumentations as A
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from concurrent.futures import ThreadPoolExecutor, as_completed
from torchvision import transforms

class DataProcessor:
    def __init__(self, config_path):
        """Create a new DataProcessor instance."""
        with open(config_path) as f:
            self.config = yaml.load(f, Loader=yaml.FullLoader)

        self.data = None
        self.resize_shape = self.config['preprocessing']['resize']
        self.images_path = self.config['data']['raw_images_path']
        self.batch_size = self.config['hyperparameters']['batch_size']
        self.n_workers = multiprocessing.cpu_count()
        
    def load_raw_data(self, rate=1):
        """Read the raw data from the CSV file."""
        raw_csv_path = self.config['data']['raw_csv_path']
        self.data = pd.read_csv(raw_csv_path)

        # Chỉ lấy một phần dữ liệu để xử lý nhanh hơn
        self.data = self.data.sample(frac=rate, random_state=42)
        print('Raw data shape:', self.data.shape)


    def data_check(self):
        """Check the data for potential issues."""
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
        """Preprocess the data by cleaning, encoding labels, and adding image paths."""
        self.data.drop_duplicates(subset=['image_id'], inplace=True)
        self.data.dropna(subset=['age', 'sex', 'localization'], inplace=True)

        self.data = self.data[self.data['age'] != 'unknown']
        self.data = self.data[self.data['sex'] != 'unknown']
        self.data = self.data[self.data['localization'] != 'unknown']

        # Convert age to integer
        self.data['age'] = self.data['age'].astype(int)

        # Encode labels
        self._encode_labels()
        
        # Add image paths
        self.data['image_id'] = self.data['image_id'].apply(lambda x: os.path.join(self.images_path, x + '.jpg'))
        
        print('Data after cleaning:')
        print(self.data.head())

        

    def _encode_labels(self):
        """Encode the categorical labels using LabelEncoder."""
        label_encoder_dx = LabelEncoder()
        self.data['dx_code'] = label_encoder_dx.fit_transform(self.data['dx'])
        
        label_encoder_dx_type = LabelEncoder()
        self.data['dx_type_code'] = label_encoder_dx_type.fit_transform(self.data['dx_type'])
        
        label_encoder_sex = LabelEncoder()
        self.data['sex_code'] = label_encoder_sex.fit_transform(self.data['sex'])
        
        #label_encoder_localization = LabelEncoder()
        #self.data['localization_code'] = label_encoder_localization.fit_transform(self.data['localization'])

        """Save the label encoders to a file."""
        self.label_encoders = {
            'dx': label_encoder_dx,
            'dx_type': label_encoder_dx_type,
            'sex': label_encoder_sex,
            #'localization': label_encoder_localization
        }
        encoder_file_name = self.config['encoder']['path'] + str(self.config['preprocessing']['resize'][0]) + 'x' + str(self.config['preprocessing']['resize'][1]) + '.pt'
        torch.save(self.label_encoders, encoder_file_name)
    
    def drop_unnecessary_columns(self):
        """Drop unnecessary columns."""
        self.data = self.data.drop(columns=['dx', 'dx_type', 'sex', 'localization'])
        self.data.reset_index(drop=True, inplace=True)

    def split_data(self, test_size=0.2, random_state=42):
        """Split the data into training and testing sets."""
        train_data, test_data = train_test_split(self.data, test_size=test_size, random_state=random_state)
        return train_data.reset_index(drop=True), test_data.reset_index(drop=True)

    def load_images_worker(self, batch):
        """Worker function to load images from a batch."""
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
        """Read all images from the dataset."""
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
        lesion_type = data.groupby(data['dx_code']).count()
        lesion_type = lesion_type.reset_index()
        lesion_type = lesion_type[['dx_code', 'image_id']]
        lesion_type.columns = ['dx_code', 'count']
        lesion_type = lesion_type.sort_values(by='count', ascending=False)
        print(lesion_type)
        max_count = lesion_type['count'].max()
        augmentation_counts = {}
        for index, row in lesion_type.iterrows():
            augmentation_counts[row['dx_code']] = int(max_count / row['count'] * rate)
        return augmentation_counts
    
    def augment_image_worker(self, row, image, augmentations, augmentation_counts):
        augmented_images = []
        augmented_data = []

        lesion_type = row['dx_code']
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
    
    def compute_img_mean_std_worker(self, batch_images):
        img_w, img_h = self.resize_shape
        batch_imgs = []

        for image in batch_images:
            img = cv2.resize(image, (img_w, img_h))
            batch_imgs.append(img)

        return np.array(batch_imgs)

    def compute_img_mean_std(self, images):
        means = np.zeros(3)
        stdevs = np.zeros(3)
        total_count = 0
        # Split images into batches
        batches = [images[i:i + self.batch_size] for i in range(0, len(images), self.batch_size)]

        with ThreadPoolExecutor(max_workers=self.n_workers) as executor:
            futures = {executor.submit(self.compute_img_mean_std_worker, batch): batch for batch in batches}
            for future in tqdm(as_completed(futures), total=len(futures), desc="Processing Batches"):
                img_batch = future.result()
                if img_batch is not None:
                    img_batch = img_batch.astype(np.float32) / 255.0  # Normalize

                    # Compute mean and std for each channel in the batch
                    for i in range(3):  # RGB channels
                        means[i] += img_batch[:, :, :, i].mean() * img_batch.shape[0]
                        stdevs[i] += (img_batch[:, :, :, i] ** 2).mean() * img_batch.shape[0]

                    total_count += img_batch.shape[0]  # Update total count

        # Compute means and stds
        means /= total_count
        stdevs = np.sqrt(stdevs / total_count - means ** 2)
        print('Means:', means)
        print('Stdevs:', stdevs)

        return means, stdevs
    
    def convert_to_tensor(self, image):
        """Convert an image to a PyTorch tensor."""
        to_tensor = transforms.ToTensor()
        return to_tensor(image)
    
    def save_data(self, data, images, file_suffix):
        """Save the data and images to a file."""
        # Convert images to tensors
        augmented_images = [self.convert_to_tensor(image) for image in images]

        # Stack all images into a single tensor
        augmented_images = torch.stack(augmented_images)

        # Stack all data into a single tensor
        # Compute mean and standard deviation
        norm_mean, norm_std = self.compute_img_mean_std(images)
        norm_mean = torch.tensor(norm_mean)
        norm_std = torch.tensor(norm_std)

        # Create file paths
        suffix = str(self.resize_shape[0]) + 'x' + str(self.resize_shape[1]) + '_' + file_suffix
        file_path = os.path.join(self.config['data']['processed_path'], f'HAM10000_{suffix}.pt')

        # If the directory doesn't exist, create it
        os.makedirs(os.path.dirname(file_path), exist_ok=True)  # Create directory for data

        # Save all data in a single file
        torch.save({
            'data': data,
            'images': augmented_images,
            'mean': norm_mean,
            'std': norm_std
        }, file_path)

    def run(self, aug=True, rate=1):
        """Main - Run the data processing pipeline."""
        start_time = time.time()
        self.load_raw_data(rate)
        self.data_check()
        self.data_cleaning()
        self.drop_unnecessary_columns()
        train_data, test_data = self.split_data()
        train_images = self.load_images(train_data)
        test_images = self.load_images(test_data)

        augmentation_counts = self.type_count(train_data, self.config['preprocessing']['augmentation']['ratio'])
        
        if aug:
            augmented_train_data, augmented_train_images = self.augment_images(train_data, train_images, augmentation_counts)
        else:
            augmented_train_data, augmented_train_images = train_data, train_images

        augmented_train_data = pd.DataFrame(augmented_train_data)
        # Split augmented data into features and labels
        train_data_features = augmented_train_data.drop(columns=['dx_code'])
        train_data_labels = augmented_train_data['dx_code']
        test_data_labels = test_data['dx_code']
        print('Augmented train data shape:', augmented_train_data.shape)
        print('Augmented train images shape:', augmented_train_images[0].shape)
        print("Train data labels shape:", train_data_labels.shape)
        print('Raw test data shape:', test_data.shape)

        print("Saving data...")
        print("Saving train data...")
        self.save_data(train_data_labels, augmented_train_images, 'train')
        print("Saving test data...")
        self.save_data(test_data_labels, test_images, 'test')
        print('Data preprocessing completed in', round(time.time() - start_time, 2), 'seconds.')
        

if __name__ == "__main__":
    processor = DataProcessor(config_path='config.yaml')
    processor.run(aug=True)

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
compute_img_mean_std_worker(): Tính giá trị trung bình và độ lệch chuẩn cho một batch ảnh.
compute_img_mean_std(): Tính giá trị trung bình và độ lệch chuẩn cho tất cả ảnh.
save_data(): Lưu dữ liệu và ảnh vào file.
run(): Chạy toàn bộ quy trình xử lý dữ liệu.
"""