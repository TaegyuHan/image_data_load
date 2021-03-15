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

print(PIL.Image.open(str(roses[0])))
print(PIL.Image.open(str(roses[0])).size)
print(PIL.Image.open(str(roses[0])).format)

# 이미지 확인하기
PIL.Image.open(str(roses[0])).show()

# ndarray = img.imread()
# pp.imshow()
# pp.show()









