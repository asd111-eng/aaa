import os

# 手动拼接路径
disease_path = r"C:\Users\杜金澎\Desktop\新建文件夹\水稻-白叶枯病"
print("检查的路径是:", disease_path)

# 列出文件夹下的所有文件
all_files = os.listdir(disease_path)
print("文件夹里的所有文件:", all_files)

# 筛选图片
image_extensions = (".png", ".jpg", ".jpeg", ".bmp")
image_files = [f for f in all_files if f.lower().endswith(image_extensions)]
print("筛选出的图片文件:", image_files)