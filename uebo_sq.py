#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import matplotlib.pyplot as plt
import cv2
import random

#randomシード値固定np.random.seed(1)

# 配列全表示
np.set_printoptions(threshold = np.inf)

#小数点以下 2 桁
# np.set_printoptions(precision=2, suppress=True)

get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


# 画像読み込み
# Annotation
anno1 = cv2.cvtColor(cv2.imread("../hh_ano/sq/uebo_left.png"), cv2.COLOR_BGR2RGB)
anno2 = cv2.cvtColor(cv2.imread("../hh_ano/sq/uebo_right.png"), cv2.COLOR_BGR2RGB)

# HSI
def hyprawread(name,hor,ver,SpectDim):
    with open(name,'rb') as f:
        #転置
        img = np.fromfile(f,np.uint16,-1)
    img = np.reshape(img,(ver,SpectDim,hor))
    img = np.transpose(img, (0,2,1))
    return img

hand1 = hyprawread("../hand/uebo_left_Img-d(s23,g50,43.45ms,350-1100)_20220301_180111.nh7",1280,1024,151)
hand2 = hyprawread("../hand/uebo_right_Img-d(s23,g50,43.45ms,350-1100)_20220301_175740.nh7",1280,1024,151)


# In[4]:


# アノテーションの要素を0,1に変更(0が黒)
anno1 = anno1 / 255
anno2 = anno2 / 255


# In[5]:


# 3次元配列から2次元配列にそれぞれ変更
anno1_reshape = anno1.reshape(1024 * 1280, 3)
anno2_reshape = anno2.reshape(1024 * 1280, 3)

hand1_reshape = hand1.reshape(1024 * 1280, 151)
hand2_reshape = hand2.reshape(1024 * 1280, 151)


# In[6]:


# アノテーション画像より要素が1の行番号を取得する(白，手の部分)(2500)
anno1_1 = np.where(np.all(anno1_reshape == 1, axis=1))[0]
anno2_1 = np.where(np.all(anno2_reshape == 1, axis=1))[0]


# In[7]:


# HSIより手の部分を2500ピクセル分取得する
hand1_reshape_choice = np.take(hand1_reshape, anno1_1, axis=0)
hand2_reshape_choice = np.take(hand2_reshape, anno2_1, axis=0)


# In[8]:


# 相関係数行列
corr_matrix = np.corrcoef(hand1_reshape_choice.T, hand2_reshape_choice.T)
print(corr_matrix)
print(corr_matrix.shape)


# In[9]:


# Rx = Rxx^(-1)RxyRyy^(-1)Ryx の計算を行う
# Ry = Ryy^(-1)RyxRxx^(-1)Rxy の計算を行う
R_xx = corr_matrix[0:151, 0:151]
R_yy = corr_matrix[151:302, 151:302]
R_xy = corr_matrix[0:151, 151:302]
R_yx = corr_matrix[151:302, 0:151]

# 逆行列
R_xx_1 = np.linalg.inv(R_xx)
R_yy_1 = np.linalg.inv(R_yy)

# R_x
R_x_1 = np.dot(R_xx_1, R_xy)
R_x_2 = np.dot(R_x_1, R_yy_1)
R_x = np.dot(R_x_2, R_yx)


# R_y
R_y_1 = np.dot(R_yy_1, R_yx)
R_y_2 = np.dot(R_y_1, R_xx_1)
R_y = np.dot(R_y_2, R_xy)

print(R_x)
print(R_y)


# In[10]:


# それぞれの固有値，固有ベクトルを求める
eigenvalues_x, eigenvectors_x = np.linalg.eig(R_x)
print("固有値_x:", eigenvalues_x)
print("固有ベクトル_x:", eigenvectors_x)

eigenvalues_y, eigenvectors_y = np.linalg.eig(R_y)
print("固有値_y:", eigenvalues_y)
print("固有ベクトル_y:", eigenvectors_y)


# In[11]:


# R_xについて
print("R_x")
# 固有値
print("固有値:", eigenvalues_x)

# 固有ベクトル
print("固有ベクトル:", eigenvectors_x)

# 第1 - 第151固有値
print("第1, 第2, ..., 第151固有値:", np.sqrt(eigenvalues_x))

# 第1固有値ベクトル
ta1 = eigenvectors_x[:, 0]
c1_1 = np.dot(ta1.T, R_xx)
c1 = np.dot(c1_1, ta1)
a1 = ta1 / np.sqrt(c1)
print("a1:", a1)

# 第2固有値ベクトル
ta2 = eigenvectors_x[:, 1]
c2_1 = np.dot(ta2.T, R_xx)
c2 = np.dot(c2_1, ta2)
a2 = ta2 / np.sqrt(c2)
print("a2:", a2)

# 第3固有値ベクトル
ta3 = eigenvectors_x[:, 2]
c3_1 = np.dot(ta3.T, R_xx)
c3 = np.dot(c3_1, ta3)
a3 = ta3 / np.sqrt(c3)
print("a3:", a3)


# In[12]:


# R_yについて
print("R_y")
# 固有値
print("固有値:", eigenvalues_y)

# 固有ベクトル
print("固有ベクトル:", eigenvectors_y)

# 第1 - 第151固有値
print("第1, 第2, ..., 第151固有値:", np.sqrt(eigenvalues_y))

# 第1固有値ベクトル
tb = eigenvectors_y[:, 0]
tb1 = -tb
d1_1 = np.dot(tb1.T, R_yy)
d1 = np.dot(d1_1, tb1)
b1 = tb1 / np.sqrt(d1)
print("b1:", b1)

# 第2固有値ベクトル
tb2 = eigenvectors_y[:, 1]
d2_1 = np.dot(tb2.T, R_yy)
d2 = np.dot(d2_1, tb2)
b2 = tb2 / np.sqrt(d2)
print("b2:", b2)

# 第3固有値ベクトル
tb3 = eigenvectors_y[:, 2]
d3_1 = np.dot(tb3.T, R_yy)
d3 = np.dot(d3_1, tb3)
b3 = tb3 / np.sqrt(d3)
print("b3:", b3)


# In[13]:


# 正準相関係数 = 最大固有値
max_corr_1 = np.dot(a1.T, R_xy)
max_corr = np.dot(max_corr_1, b1)
print(max_corr)


# In[14]:


# 行列とベクトルによる正準負荷量(本来の変数がどの程度の影響があるのか)の計算(-1 <= n <= 1)
print("hand1(第1正準変数で変換した値)と各バンドの正準負荷", np.dot(R_xx, a1))
print("hand2(第1正準変数で変換した値)と各バンドの正準負荷", np.dot(R_yy, b1))


# In[15]:


# バンドのhand1に対する寄与率(正準変数によりどの程度の情報量の説明ができるか)(0 <= n <= 1)
print("バンドのhand1に対する寄与率(第1正準変数):", np.sum(np.dot(R_xx, a1) ** 2) / 151)
# バンドのhand2に対する寄与率
print("バンドのhand2に対する寄与率(第1正準変数):", np.sum(np.dot(R_yy, b1) ** 2) / 151)


# In[16]:


# 冗長性係数
print("hand1(正準変数)がhand2の変数", (sum(np.dot(R_yx, a1) ** 2)) / 151)
print("hand1(正準変数)がhand2の変数", (sum(np.dot(R_xy, b1) ** 2)) / 151)


# In[17]:


plt.plot(np.abs(np.dot(R_xx, a1)), label = "左手")
#plt.plot(np.abs(np.dot(R_yy, b1)), label = "右手")

plt.title("被験者8", fontname="MS Gothic")
plt.xlabel("バンド", fontname="MS Gothic")
plt.ylabel("負荷量", fontname="MS Gothic")
plt.xlim(0, 151)
plt.ylim(0, 1)


# In[ ]:




