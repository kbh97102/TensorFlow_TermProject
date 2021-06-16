import glob

import numpy as np
import tensorflow as tf
from PIL import Image
from keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Dropout
from keras_preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
import os


train_dir = "./data/train"
testdir = "./data/test"
img_size = 48
categories = ["angry", "disgusted", "fearful", "happy", "neutral", "sad", "surprised"]
nb_classes = len(categories)
epochs = 30

# 이미지 데이터 읽어 들이기
all_image_paths = []
all_onehot_labels = []

for idx, cat in enumerate(categories):
    # 레이블 지정
    label = [0 for i in range(nb_classes)]  # one-hot준비 [0,0,0,0,0]
    label[idx] = 1  # one-hot 리스트 생성
    # 이미지
    image_dir = train_dir + "/" + cat
    # 각 폴더에 있는 모든 파일이름에 대한 리스트 생성
    files = glob.glob(image_dir + "/*.png")
    for f in files:
        all_image_paths.append(f)
        all_onehot_labels.append(label)


# tf.iamge형태로 이미지 로딩
# jpg를 디코딩하고 사이즈 조절과 정규화를 동시 진행
# label인자에 대한 처리는 하지 않고 단순히 받아서 그대로 리턴함
def load_image_path_label(path, label):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [img_size, img_size])
    image /= 255.0  # normalize to [0,1] range
    return image, label


full_dataset = tf.data.Dataset.from_tensor_slices((all_image_paths, all_onehot_labels))
full_dataset = full_dataset.map(load_image_path_label)


# 전체 데이터 갯수 계산
DATASET_SIZE = len(all_image_paths)

# 3:1학습과 테스트 데이터 분리
train_size = int(0.75 * DATASET_SIZE)
test_size = DATASET_SIZE - train_size
# 랜덤하게 shuffling
full_dataset = full_dataset.shuffle(buffer_size=int(DATASET_SIZE * 1.5))

# 학습 데이터 생성
train_ds = full_dataset.take(train_size)
train_ds = train_ds.batch(30)

# 나머지를 테스트 용으로 사용
test_ds = full_dataset.skip(train_size)
test_ds = test_ds.batch(30)



# 모델링 작업

# TODO 모델링 작업시 필터나 커털 사이즈, stride 를 조정
# TODO Dropout 조절
# TODO padding 옵션 same, valid
# TODO Running rate 조절
model = Sequential()

model.add(Conv2D(32, kernel_size=(3, 3), padding='same', activation='relu', input_shape=(img_size, img_size, 1)))
model.add(MaxPool2D(pool_size=(2, 2), strides=2))

model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2), strides=2))

# 데이터를 일자로 쭉 핌
model.add(Flatten())

# Fully connected network 만들기
model.add(Dense(7, activation='softmax'))

model.compile(
    optimizer=Adam(lr=0.0001),  # optimizer = tf.keras.optimizers.Adam(0.001)
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# model.fit(x=train_datas,epochs=epochs, validation_data=test_datas)
model.fit(train_ds, epochs=epochs)

model.evaluate(test_ds)

print("Label:      ", test_ds)

pred = model.predict(test_ds)
print("Prediction: ", tf.math.argmax(pred, 1).numpy())
