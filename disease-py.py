# -*- coding: utf-8 -*-  # 强制Python识别UTF-8编码
import streamlit as st
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from PIL import Image
import tensorflow as tf
import os
import json
from datetime import datetime
import warnings
import sys

# ============ 安全编码配置 ============
os.environ['PYTHONIOENCODING'] = 'utf-8'
os.environ['STREAMLIT_SERVER_HEADLESS'] = 'true'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 屏蔽TensorFlow无关日志

warnings.filterwarnings("ignore", message="use_column_width is deprecated")

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model

# ===================== 配置 =====================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_FILE = os.path.join(BASE_DIR, "crop_model.h5")
CLASS_FILE = os.path.join(BASE_DIR, "classes.json")
EXAMPLES_ROOT = os.path.join(BASE_DIR, "examples")

# ===================== 病害防治建议库 =====================
ADVICE = {
    "小麦-健康": {"建议": "小麦植株健康，长势良好。建议继续保持常规田间管理，合理水肥，注意防倒、防早衰。"},
    "小麦-条锈病": {
        "病因": "由条形柄锈菌引起的真菌病害",
        "物理防治": "及时清除病叶，合理轮作",
        "生物防治": "利用枯草芽孢杆菌等生防菌",
        "化学防治": "使用三唑酮、戊唑醇等药剂",
        "禁限用提醒": "禁止使用甲拌磷等高毒农药",
        "施药间隔": "7-10天，安全间隔期15天"
    },
    "小麦-叶锈病": {
        "病因": "由隐匿柄锈菌引起的真菌病害",
        "物理防治": "选用抗病品种，合理密植",
        "生物防治": "喷施枯草芽孢杆菌制剂",
        "化学防治": "使用烯唑醇、氟环唑等药剂",
        "禁限用提醒": "禁止使用甲拌磷等高毒农药",
        "施药间隔": "7-10天，安全间隔期15天"
    },
    "小麦-白粉病": {
        "病因": "由禾本科布氏白粉菌引起的真菌病害",
        "物理防治": "合理施肥，避免偏施氮肥",
        "生物防治": "利用木霉菌等生防菌",
        "化学防治": "使用醚菌酯、戊唑醇等药剂",
        "禁限用提醒": "禁止使用甲拌磷等高毒农药",
        "施药间隔": "7-10天，安全间隔期15天"
    },
    "水稻-健康": {"建议": "水稻生长正常，叶色清秀。建议保持浅水层管理，适时晒田，促进根系下扎，预防后期倒伏。"},
    "水稻-稻瘟病(叶瘟)": {
        "病因": "由稻瘟病菌引起的真菌病害",
        "物理防治": "合理密植，浅水勤灌",
        "生物防治": "释放稻瘟病拮抗细菌",
        "化学防治": "使用三环唑、稻瘟灵等药剂",
        "禁限用提醒": "禁止使用甲基异柳磷等农药",
        "施药间隔": "5-7天，安全间隔期20天"
    },
    "水稻-白叶枯病": {
        "病因": "由水稻黄单胞菌引起的细菌性病害",
        "物理防治": "选用抗病品种，避免串灌",
        "生物防治": "使用枯草芽孢杆菌制剂",
        "化学防治": "使用噻唑锌、噻霉酮等药剂",
        "禁限用提醒": "禁止使用高毒有机磷农药",
        "施药间隔": "7-10天，安全间隔期15天"
    },
    "水稻-纹枯病": {
        "病因": "由立枯丝核菌引起的真菌病害",
        "物理防治": "合理密植，浅水勤灌，适时晒田",
        "生物防治": "使用井冈霉素、芽孢杆菌制剂",
        "化学防治": "使用噻呋酰胺、戊唑醇等药剂",
        "禁限用提醒": "禁止使用高毒有机磷农药",
        "施药间隔": "7-10天，安全间隔期15天"
    },
    "玉米-健康": {"建议": "玉米生长正常，株型紧凑。建议注意控旺防倒，大喇叭口期追施穗肥，促进灌浆。"},
    "玉米-大斑病": {
        "病因": "由大斑凸脐蠕孢引起的真菌病害",
        "物理防治": "选用抗病品种，轮作倒茬",
        "生物防治": "使用木霉菌、芽孢杆菌制剂",
        "化学防治": "使用苯醚甲环唑、吡唑醚菌酯等药剂",
        "禁限用提醒": "禁止使用甲拌磷等高毒农药",
        "施药间隔": "7-10天，安全间隔期15天"
    },
    "玉米-小斑病": {
        "病因": "由玉蜀黍平脐蠕孢引起的真菌病害",
        "物理防治": "选用抗病品种，及时清除病残体",
        "生物防治": "使用木霉菌、芽孢杆菌制剂",
        "化学防治": "使用代森锰锌、苯醚甲环唑等药剂",
        "禁限用提醒": "禁止使用甲拌磷等高毒农药",
        "施药间隔": "7-10天，安全间隔期15天"
    },
    "玉米-锈病": {
        "病因": "由玉米柄锈菌引起的真菌病害",
        "物理防治": "选用抗病品种，合理密植",
        "生物防治": "使用枯草芽孢杆菌制剂",
        "化学防治": "使用三唑酮、戊唑醇等药剂",
        "禁限用提醒": "禁止使用甲拌磷等高毒农药",
        "施药间隔": "7-10天，安全间隔期15天"
    },
    "大豆-健康": {"建议": "大豆长势正常，叶片浓绿。注意清沟排涝，花荚期喷施叶面肥，保花保荚。"},
    "大豆-霜霉病": {
        "病因": "由霜霉菌引起的真菌病害，高湿低温易发病。",
        "物理防治": "轮作倒茬，选用抗病品种，及时清除病残体。",
        "生物防治": "施用枯草芽孢杆菌等生防菌剂。",
        "化学防治": "发病初期喷施甲霜灵、霜脲氰等药剂。",
        "禁限用提醒": "避免在花期使用高毒农药。",
        "施药间隔": "7-10天"
    },
    "大豆-根腐病": {
        "病因": "由多种镰刀菌等引起的土传病害，连作地发病重。",
        "物理防治": "轮作，选用包衣种子，合理密植。",
        "生物防治": "使用木霉菌、芽孢杆菌等进行种子处理。",
        "化学防治": "发病初期用恶霉灵、多菌灵灌根。",
        "禁限用提醒": "避免使用高残留农药。",
        "施药间隔": "10-15天"
    },
    "大豆-灰斑病": {
        "病因": "由大豆灰斑病菌引起，多雨高湿年份易流行。",
        "物理防治": "选用抗病品种，收获后深翻土地。",
        "生物防治": "喷施枯草芽孢杆菌制剂。",
        "化学防治": "发病初期喷施甲基硫菌灵、百菌清。",
        "禁限用提醒": "按推荐剂量使用，避免药害。",
        "施药间隔": "7-10天"
    },
    "大豆-食心虫": {
        "病因": "大豆食心虫幼虫蛀食豆荚，造成虫孔和虫粪。",
        "物理防治": "轮作，及时翻耕，设置杀虫灯诱杀成虫。",
        "生物防治": "释放赤眼蜂，施用白僵菌。",
        "化学防治": "成虫盛发期喷施氯氟氰菊酯等药剂。",
        "禁限用提醒": "注意农药安全间隔期。",
        "施药间隔": "7天"
    },
    "棉花-健康": {"建议": "棉花生长正常，株型舒展。建议合理整枝，及时打顶，促进棉铃发育，注意防早衰。"},
    "棉花-黄萎病": {
        "病因": "由大丽轮枝菌引起的维管束病害，高温高湿易发病。",
        "物理防治": "轮作倒茬，选用抗病品种，清除病株。",
        "生物防治": "施用芽孢杆菌、假单胞菌等生防菌。",
        "化学防治": "发病初期用恶霉灵、多菌灵灌根。",
        "禁限用提醒": "避免在棉田使用高毒农药。",
        "施药间隔": "15天"
    },
    "棉花-枯萎病": {
        "病因": "由尖孢镰刀菌引起，连作地发病严重。",
        "物理防治": "轮作，选用抗病品种，种子消毒。",
        "生物防治": "使用木霉菌进行土壤处理。",
        "化学防治": "发病初期喷施咪鲜胺、苯醚甲环唑。",
        "禁限用提醒": "严格遵守农药安全间隔期。",
        "施药间隔": "10-15天"
    },
    "棉花-棉铃虫": {
        "病因": "棉铃虫幼虫蛀食蕾、花、铃，造成脱落和烂铃。",
        "物理防治": "安装杀虫灯，人工捕捉幼虫。",
        "生物防治": "释放赤眼蜂，施用核型多角体病毒。",
        "化学防治": "卵孵化盛期喷施氯虫苯甲酰胺、甲维盐。",
        "禁限用提醒": "注意保护天敌，轮换用药。",
        "施药间隔": "7-10天"
    },
    "棉花-蚜虫": {
        "病因": "棉蚜吸食汁液，分泌蜜露，诱发煤污病。",
        "物理防治": "黄板诱杀，清除田间杂草。",
        "生物防治": "保护和利用瓢虫、草蛉等天敌。",
        "化学防治": "喷施吡虫啉、啶虫脒等内吸性药剂。",
        "禁限用提醒": "避免使用高毒农药，注意对蜜蜂的影响。",
        "施药间隔": "7天"
    },
    "马铃薯-健康": {"建议": "马铃薯生长正常，薯块膨大良好。建议合理控旺，中耕培土，注意排涝，促进块茎膨大。"},
    "马铃薯-晚疫病": {
        "病因": "由疫霉菌引起，低温高湿条件下易流行。",
        "物理防治": "选用无病种薯，轮作，及时排水。",
        "生物防治": "施用枯草芽孢杆菌、假单胞菌。",
        "化学防治": "发病初期喷施甲霜灵、霜霉威。",
        "禁限用提醒": "严格控制用药剂量和次数。",
        "施药间隔": "7-10天"
    },
    "马铃薯-早疫病": {
        "病因": "由链格孢菌引起，高温干旱易发病。",
        "物理防治": "轮作，合理密植，增施钾肥。",
        "生物防治": "喷施木霉菌制剂。",
        "化学防治": "发病初期喷施代森锰锌、百菌清。",
        "禁限用提醒": "避免在高温时段施药。",
        "施药间隔": "10天"
    },
    "马铃薯-环腐病": {
        "病因": "由细菌引起，通过种薯传播，造成维管束坏死。",
        "物理防治": "选用无病种薯，切刀消毒，轮作。",
        "生物防治": "使用芽孢杆菌进行种子处理。",
        "化学防治": "目前无特效药剂，重点在预防。",
        "禁限用提醒": "禁止使用高毒农药。",
        "施药间隔": "无"
    },
    "油菜-健康": {"建议": "油菜长势正常，群体结构合理。建议清沟排水，花期喷施硼肥，预防“花而不实”。"},
    "油菜-菌核病": {
        "病因": "由核盘菌引起，花期多雨易发病。",
        "物理防治": "轮作，选用抗病品种，及时清沟排水。",
        "生物防治": "施用盾壳霉、木霉菌等生防菌。",
        "化学防治": "初花期喷施腐霉利、异菌脲。",
        "禁限用提醒": "注意农药安全间隔期。",
        "施药间隔": "7-10天"
    },
    "油菜-霜霉病": {
        "病因": "由霜霉菌引起，低温高湿易发病。",
        "物理防治": "轮作，合理密植，清除病叶。",
        "生物防治": "喷施枯草芽孢杆菌制剂。",
        "化学防治": "发病初期喷施甲霜灵、霜脲氰。",
        "禁限用提醒": "避免在花期施药。",
        "施药间隔": "7天"
    },
    "油菜-蚜虫": {
        "病因": "油菜蚜吸食汁液，传播病毒病。",
        "物理防治": "黄板诱杀，清除田间杂草。",
        "生物防治": "保护瓢虫、草蛉等天敌。",
        "化学防治": "喷施吡虫啉、啶虫脒。",
        "禁限用提醒": "注意对蜜蜂的影响。",
        "施药间隔": "7天"
    }
}

# ===================== 病害防治建议（纯原生组件，无HTML） =====================
def show_disease_advice(disease_name):
    st.subheader("病害防治建议")
    
    advice = ADVICE.get(disease_name, None)
    if not advice:
        st.info("暂未收录该病害的防治建议，请咨询农业技术人员。")
        return
    
    if "健康" in disease_name:
        st.write(advice["建议"])
        return
    
    # 用columns分栏+纯文本，彻底避免expander标题渲染乱码
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("病因分析")
        st.write(advice.get("病因", "暂无"))
        
        st.subheader("物理防治")
        st.write(advice.get("物理防治", "暂无"))
        
        st.subheader("生物防治")
        st.write(advice.get("生物防治", "暂无"))
    
    with col2:
        st.subheader("化学防治")
        st.write(advice.get("化学防治", "暂无"))
        
        st.subheader("禁限用提醒")
        st.write(advice.get("禁限用提醒", "暂无"))
        
        st.subheader("施药间隔")
        st.write(advice.get("施药间隔", "暂无"))

# ===================== 训练模型 =====================
def train_all():
    datagen = ImageDataGenerator(
        rescale=1./255, validation_split=0.2,
        rotation_range=30, width_shift_range=0.2, height_shift_range=0.2,
        shear_range=0.15, zoom_range=0.2, horizontal_flip=True, vertical_flip=True
    )
    train_gen = datagen.flow_from_directory(
        os.path.join(BASE_DIR, "images"), 
        target_size=(224,224), 
        batch_size=32, 
        subset="training", 
        class_mode='categorical'
    )
    val_gen = datagen.flow_from_directory(
        os.path.join(BASE_DIR, "images"), 
        target_size=(224,224), 
        batch_size=32, 
        subset="validation", 
        class_mode='categorical'
    )
    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(train_gen.classes), y=train_gen.classes)
    class_weight_dict = {i: class_weights[i] for i in range(len(class_weights))}

    base = MobileNetV2(weights="imagenet", include_top=False, input_shape=(224,224,3))
    base.trainable = False
    x = base.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation="relu")(x)
    out = Dense(train_gen.num_classes, activation="softmax")(x)
    model = Model(inputs=base.input, outputs=out)

    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    history = model.fit(train_gen, epochs=50, validation_data=val_gen, class_weight=class_weight_dict)

    model.save(MODEL_FILE)
    with open(CLASS_FILE, "w", encoding="utf-8") as f:
        json.dump(train_gen.class_indices, f, ensure_ascii=False, indent=2)

# ===================== 预测 =====================
def load_classes():
    if not os.path.exists(CLASS_FILE):
        return {}
    with open(CLASS_FILE, "r", encoding="utf-8") as f:
        c = json.load(f)
    return {v: k for k, v in c.items()}

def predict(img):
    if not os.path.exists(MODEL_FILE) or not os.path.exists(CLASS_FILE):
        return "模型未训练，请先在终端训练", 0.0
    img = img.convert("RGB").resize((224,224))
    x = np.array(img) / 255.0
    x = np.expand_dims(x, axis=0)
    model = tf.keras.models.load_model(MODEL_FILE)
    pred = model.predict(x, verbose=0)
    idx = np.argmax(pred)
    conf = float(pred[0][idx]) * 100
    classes = load_classes()
    return classes.get(idx, "未知病害"), conf

# ===================== 病害示例图 =====================
def show_disease_examples():
    st.sidebar.subheader("病害对照示例")
    if not os.path.exists(EXAMPLES_ROOT):
        st.sidebar.warning("未找到示例图文件夹")
        return
    disease_folders = [f for f in os.listdir(EXAMPLES_ROOT) if os.path.isdir(os.path.join(EXAMPLES_ROOT, f))]
    if not disease_folders:
        st.sidebar.info("无病害示例文件夹")
        return
    selected_disease = st.sidebar.selectbox("选择病害查看示例", disease_folders)
    disease_path = os.path.join(EXAMPLES_ROOT, selected_disease)
    image_files = [f for f in os.listdir(disease_path) if f.lower().endswith((".png",".jpg",".jpeg"))]
    if not image_files:
        st.sidebar.warning("无示例图片")
        return
    for img_file in image_files:
        try:
            img = Image.open(os.path.join(disease_path, img_file))
            st.sidebar.image(img, caption=img_file, use_container_width=True)
        except:
            pass

# ===================== 页面样式（清空，避免渲染冲突） =====================
def set_page_style():
    pass  # 完全移除自定义CSS，只用Streamlit默认样式

# ===================== 主函数 =====================
def main():
    if "diagnosis_history" not in st.session_state:
        st.session_state.diagnosis_history = []

    st.set_page_config(page_title="作物病害智能识别系统", page_icon=None, layout="wide")
    set_page_style()

    # 纯文本标题，无HTML
    st.title("多作物病害智能识别系统")
    st.write("支持：小麦 | 水稻 | 玉米 | 大豆 | 棉花 | 马铃薯 | 油菜")
    st.divider()

    # 侧边栏
    with st.sidebar:
        show_disease_examples()
        st.divider()
        st.subheader("诊断历史")
        for idx, record in enumerate(reversed(st.session_state.diagnosis_history)):
            st.write(f"{idx+1}. **{record['result']}**")
            st.write(f"置信度：{record['confidence']:.2f}% | {record['time']}")
            st.divider()

    # 上传识别
    st.subheader("上传叶片图片诊断")
    uploaded_file = st.file_uploader("请上传作物叶片图片（JPG/PNG）", type=["jpg","jpeg","png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        col_img, _ = st.columns([1,1])
        with col_img:
            st.image(image, caption="已上传图片", use_container_width=True)

        if st.button("开始AI智能诊断", type="primary"):
            with st.spinner("正在分析..."):
                result, confidence = predict(image)
                st.success(f"诊断结果：{result}")
                st.info(f"置信度：{confidence:.2f}%")
                show_disease_advice(result)

                current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                st.session_state.diagnosis_history.append({
                    "result":result, "confidence":confidence, "time":current_time
                })

    st.divider()

    # 专家咨询
    st.subheader("农技专家咨询")
    with st.form("expert_form"):
        col1, col2 = st.columns([1,2])
        with col1:
            crop = st.selectbox("作物类型", ["小麦","水稻","玉米","大豆","棉花","马铃薯","油菜"])
            contact = st.text_input("联系方式")
        with col2:
            desc = st.text_area("病害描述", height=120)
        if st.form_submit_button("提交咨询"):
            if contact and desc:
                st.success("提交成功！")
            else:
                st.warning("请填写完整信息")

# ===================== 入口 =====================
if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "train":
        train_all()
    else:
        main()