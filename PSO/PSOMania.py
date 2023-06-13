# Objective Function
# 1. Tentukan jumlah partikel
# 2. Tentukan arah gerak dan kecepatan partikel
# 3. Tentukan nilai pbest dan gbest
# 4. Pada tiap iterasi, update kecepatan 
# 5. Stop 

import numpy as np
import random

# Menentukan Jumlah Partikel
x1 = np.random.rand(1,10)# list: storing semua partikel
x2 = [10, 15, 18, 15, 30, 19, 40, 41, 31, 29]
# Menentukan Kecepatan Partikel
vx1 = [random.random() for i in range(10)]
vx2 = [random.random() for i in range(10)]

allpbest = []

for nilai_partikel in range(10):
  pbest = (x1[nilai_partikel]-3.14)**2 + (x2[nilai_partikel]-2.72)**2 + np.sin(3*x1[nilai_partikel]+1.41) + np.sin(4*x2[nilai_partikel]-1.73)
  allpbest.append(pbest)

indexmin = np.argmin(allpbest)
gbestx1 = x1[indexmin]
gbestx2 = x2[indexmin]
print(f"ini adalah data pbest :{allpbest}")
print(f"\nini adalah data gbest :{[gbestx1,gbestx2]}")
