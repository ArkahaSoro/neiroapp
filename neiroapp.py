# Установка библиотек 
import streamlit as st 
import requests
from PIL import Image 
import torch 
from transformers import AutoImageProcessor, AutoModelForImageClassification 

# Загрузка моделей 
@st.cache_resource()
def load_auto_image_processor(): 
    return AutoImageProcessor.from_pretrained("google/vit-base-patch16-224") 

@st.cache_resource()
def load_auto_model(): 
    return AutoModelForImageClassification.from_pretrained("google/vit-base-patch16-224")

# Инициализация процессора и модели 
processor = load_auto_image_processor() 
model = load_auto_model() 

# Реализация функции распознания объектов на изображении 
def predict_step(image): 
    try: 
        if image.mode != "RGB": 
            image = image.convert(mode="RGB") 
        pixel_values = processor(images=image, return_tensors="pt").pixel_values 
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
        pixel_values = pixel_values.to(device) 
        model.to(device) 

        with torch.no_grad(): 
            outputs = model(pixel_values) 
            logits = outputs.logits 

        predicted_class_idx = logits.argmax(-1).item() 
        labels = model.config.id2label  
        predicted_label = labels[predicted_class_idx] 

        return predicted_label 
    except Exception as e: 
        st.error(f"Произошла ошибка при предсказании: {e}") 
        return None 

# Использование в Streamlit 
st.title("Фотоклассификатор") 
uploaded_file = st.file_uploader("Выберите изображение...", type=["jpg", "jpeg", "png"]) 

if uploaded_file is not None: 
    image = Image.open(uploaded_file) 
    st.image(image, caption='Uploaded Image.', use_container_width=True) 
    st.write("") 
    st.write("Определяет вид...") 
    
    prediction = predict_step(image) 
    if prediction is not None: 
        st.write(f"Относится к виду: {prediction}") 
