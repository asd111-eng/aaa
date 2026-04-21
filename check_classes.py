import json

# 读取classes.json文件
with open("classes.json", "r", encoding="utf-8") as f:
    classes = json.load(f)

# 打印所有病害名称（语法完全正确）
print("=== 病害类别清单（共{}个）===".format(len(classes)))
for i, name in enumerate(classes):
    print(f"{i+1}. {name}")