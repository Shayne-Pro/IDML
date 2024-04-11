import os  
from PIL import Image  
  
def convert_png_to_jpg(input_folder, output_folder):  
    # 确保输出文件夹存在  
    if not os.path.exists(output_folder):  
        os.makedirs(output_folder)  
  
    # 遍历输入文件夹中的所有文件  
    for filename in os.listdir(input_folder):  
        # 检查文件是否为PNG图像  
        if filename.endswith(".png"):  
            # 读取PNG图像  
            img = Image.open(os.path.join(input_folder, filename))  
            # 转换为RGB模式（如果图像是RGBA）
            if img.mode == 'RGBA':
                img = img.convert('RGB')
            # 转换图像为JPG格式  
            jpg_filename = filename.replace(".png", ".jpg")  
            img.save(os.path.join(output_folder, jpg_filename), "JPEG")  
  
# 设置输入和输出文件夹路径  
input_folder = "./data/CT/non-COVID"  # 请替换为你的输入文件夹路径  
output_folder = "./data/CT_jpg/non-COVID"  # 请替换为你的输出文件夹路径  
  
# 调用函数进行转换  
convert_png_to_jpg(input_folder, output_folder)