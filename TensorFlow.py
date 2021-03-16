# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
"""
    소셜네트워크 분석 과제
    순천향대학교 빅데이터공학과
    20171483 한태규
    
    ------------------------- memo -------------------------
    https://www.tensorflow.org/tutorials/load_data/images
    
    이미지 로드하는 방법 정리
    
    --------------------------------------------------------
     
    email : gksxorb147@naver.com
    update : 2021.03.15 17:31
     
"""

#-----------------------------------------------------------------------------------#
## modul 연결
import numpy as np
import os

# 이미지 분석 및 처리를 쉽게 할 수 있는 라이브러리
import PIL 
import PIL.Image

import matplotlib.pyplot as plt
import matplotlib.image as img
import matplotlib.pyplot as pp

# 머신러닝 라이브러
import tensorflow as tf
import tensorflow_datasets as tfds
#-----------------------------------------------------------------------------------#


#-----------------------------------------------------------------------------------#
# version 확인
print(tf.__version__)
#-----------------------------------------------------------------------------------#

"""
    꽃 데이터세트 다운로드하기
    이 튜토리얼에서는 수천 장의 꽃 사진 데이터세트를 사용합니다.
    꽃 데이터세트에는 클래스당 하나씩 5개의 하위 디렉토리가 있습니다.
"""

#-----------------------------------------------------------------------------------#
# 사진 다운로드
import pathlib

dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
data_dir = tf.keras.utils.get_file(origin=dataset_url, 
                                    fname='flower_photos', 
                                    untar=True)
data_dir = pathlib.Path(data_dir)
#-----------------------------------------------------------------------------------#

#-----------------------------------------------------------------------------------#
## 이미지 개수 확인하기
image_count = len(list(data_dir.glob('*/*.jpg')))
print(image_count)
#-----------------------------------------------------------------------------------#


"""
    각 디렉토리에는 해당 유형의 꽃 이미지가 포함되어 있습니다. 다음은 장미입니다.
"""

roses = list(data_dir.glob('roses/*'))

# 이미지 객체 출력
print(PIL.Image.open(str(roses[0])))

## 이미지 크기 출력
print(PIL.Image.open(str(roses[0])).size)

## 이미지 확장자 출력
print(PIL.Image.open(str(roses[0])).format)

# 이미지 확인하기
PIL.Image.open(str(roses[0])).show()


#-----------------------------------------------------------------------------------#
## 데이터세트 만들기
## 로더를 위해 일부 매개변수를 정합니다.
batch_size = 32
img_height = 180
img_width = 180
#-----------------------------------------------------------------------------------#


"""
    모델을 개발할 때 검증 분할을 사용하는 것이 좋습니다. 훈련에 이미지의 80%를 사용하고 검증에 20%를 사용합니다.
"""
#-----------------------------------------------------------------------------------#
## 훈련 데이터
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
  ## 데이터 디렉토리
  data_dir, 
  ## 0과 1 사이의 선택적 부동액, 유효성 확인을 위해 예약할 데이터의 일부입니다.
  validation_split=0.2,
  ## 트레이닝
  subset="training",
  ## 섞기 및 변환을 위한 선택적 랜덤 시드입니다.
  seed=123,
  ## 이미지 사이즈
  image_size=(img_height, img_width),
  ## 데이터를 묶는 사이즈
  batch_size=batch_size)
#-----------------------------------------------------------------------------------#

#-----------------------------------------------------------------------------------#
## 검증 데이터
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)
#-----------------------------------------------------------------------------------#

"""
    이러한 데이터세트의 class_names 속성에서 클래스 이름을 찾을 수 있습니다.
"""
#-----------------------------------------------------------------------------------#
class_names = train_ds.class_names
print(class_names)
#-----------------------------------------------------------------------------------#

#-----------------------------------------------------------------------------------#
## 데이터 시각화 하기
plt.figure(figsize=(10, 10))


for images, labels in train_ds.take(1):
  for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(images[i].numpy().astype("uint8"))
    plt.title(class_names[labels[i]])
    plt.axis("off")





