# -*- coding: utf-8 -*-
"""
Created on Fri Jun 11 11:07:19 2021

@author: Siti Hinggit
"""
from __future__ import division
#from skimage import io
#from skimage import feature
#from skimage import data, exposure
#from skimage.color import rgb2hsv
import cv2 
import numpy as np
import os
import glob
#rambutan = ['0_1.jpg', '0_2.jpg', '0_3.jpg',
  #          '1_1.jpg','1_2.jpg', '1_3.jpg', '1_4.jpg', '1_5.jpg', '1_6.jpg', '1_7.jpg','1_.jpg',
#            '2_1.jpg', '2_2.jpg', '2_3.jpg', '2_4.jpg', '2_5.jpg', '2_6.jpg', '2_7.jpg', '2_8.jpg', '2_9.jpg', '2_10.jpg']
# ...

# Path folder utama yang berisi subfolder untuk setiap kelas
folder_utama = 'Praproses'

# Membaca semua subfolder di dalam folder utama
kelas_folders = glob.glob(os.path.join(folder_utama, '*'))

for kelas_folder in kelas_folders:
    # Membaca semua file gambar dalam subfolder kelas
    path = os.path.join(kelas_folder, '*.*')
    
    for bb, file in enumerate(glob.glob(path)):
        # Membuat kernel untuk nanti masking
        kernelOpen = np.ones((5, 5))
        kernelClose = np.ones((20, 20))

        # Baca gambar rgb
        rgb = cv2.imread(file)

        # Convert BGR to HSV
        hsv = cv2.cvtColor(rgb, cv2.COLOR_BGR2HSV)

        lower_white = np.array([0, 0, 168])
        upper_white = np.array([172, 111, 255])

        whitemask = cv2.inRange(hsv, lower_white, upper_white)

        # Invers mask putih agar yang terdeteksi adalah yang bukan putih
        non_whitemask = cv2.bitwise_not(whitemask)

        # Gunakan operasi open dengan kernel yang tadi di atas
        maskOpen = cv2.morphologyEx(non_whitemask, cv2.MORPH_OPEN, kernelOpen)

        # Hasil dari open dipakai di close ini
        maskClose = cv2.morphologyEx(maskOpen, cv2.MORPH_CLOSE, kernelClose)

        # Hasil akhirnya masking dari close
        maskFinal = maskClose

        # Setelah itu gunakan bitwise untuk menyatukan gambarnya, hanya menggunakan hsv bukan rgbnya
        maskFinally = cv2.bitwise_and(hsv, hsv, mask=maskFinal)

        # Simpan gambar hasil masking sesuai dengan struktur folder
        output_folder = os.path.join('Praproses', 'output_image', os.path.basename(kelas_folder))
        os.makedirs(output_folder, exist_ok=True)
        output_path = os.path.join(output_folder, 'masking{}.jpg'.format(bb))
        cv2.imwrite(output_path, maskFinally)
