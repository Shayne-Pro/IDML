import os
import shutil
import random

# 定义数据集路径
ct_folder = './data/CT_jpg'
covid_folder = os.path.join(ct_folder, 'COVID')
noncovid_folder = os.path.join(ct_folder, 'nonCOVID')

# 定义目标文件夹路径
covid_tr_folder = './data/CT/COVID_tr'
covid_ts_folder = './data/CT/COVID_ts'
noncovid_tr_folder = './data/CT/nonCOVID_tr'
noncovid_ts_folder = './data/CT/nonCOVID_ts'

# 创建目标文件夹
os.makedirs(covid_tr_folder, exist_ok=True)
os.makedirs(covid_ts_folder, exist_ok=True)
os.makedirs(noncovid_tr_folder, exist_ok=True)
os.makedirs(noncovid_ts_folder, exist_ok=True)

# 获取COVID和nonCOVID文件夹中的所有jpg图像文件
covid_files = [f for f in os.listdir(covid_folder) if f.endswith('.jpg')]
noncovid_files = [f for f in os.listdir(noncovid_folder) if f.endswith('.jpg')]

# 随机打乱文件列表
random.shuffle(covid_files)
random.shuffle(noncovid_files)

# 计算COVID和nonCOVID的训练集和验证集数量
covid_train_count = int(len(covid_files) * 0.7)
noncovid_train_count = int(len(noncovid_files) * 0.7)

# 分别复制COVID和nonCOVID的训练集和验证集到目标文件夹
for i, file in enumerate(covid_files):
    if i < covid_train_count:
        shutil.copy(os.path.join(covid_folder, file), os.path.join(covid_tr_folder, file))
    else:
        shutil.copy(os.path.join(covid_folder, file), os.path.join(covid_ts_folder, file))

for i, file in enumerate(noncovid_files):
    if i < noncovid_train_count:
        shutil.copy(os.path.join(noncovid_folder, file), os.path.join(noncovid_tr_folder, file))
    else:
        shutil.copy(os.path.join(noncovid_folder, file), os.path.join(noncovid_ts_folder, file))
