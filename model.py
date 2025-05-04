import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # تعطيل الـ GPU

from flask import Flask, request, jsonify
import cv2
import numpy as np
from PIL import Image
import io
import tensorflow as tf
from tensorflow.keras.models import Model
import base64
import requests

app = Flask(__name__)

# تحميل الموديل و compile
model = tf.keras.models.load_model('modelv2.h5')  # استبدل بالمسار الصحيح
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# فئات الموديل (من HAM10000 أو استبدل بتاعك)
class_labels = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']

# فحص جودة الصورة
def analyze_image_quality(image_bytes, sharpness_threshold=100.0, brightness_threshold=(50, 200), min_resolution=(256, 256), noise_threshold=70.0, saturation_threshold=0.1):
    try:
        image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            return {"overall": False, "message": "الصورة غير صالحة أو تالفة."}

        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        laplacian_var = cv2.Laplacian(gray_image, cv2.CV_64F).var()
        sharpness_ok = laplacian_var >= sharpness_threshold

        brightness = np.mean(gray_image)
        brightness_ok = brightness_threshold[0] <= brightness <= brightness_threshold[1]

        height, width = gray_image.shape
        resolution_ok = height >= min_resolution[0] and width >= min_resolution[1]

        noise_level = np.std(gray_image)
        noise_ok = noise_level < noise_threshold

        try:
            img_pil = Image.open(io.BytesIO(image_bytes))
            img_format = img_pil.format
            format_ok = img_format.lower() in ["jpeg", "jpg", "png"]
        except Exception:
            format_ok = False
            img_format = "unknown"

        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        saturation = np.mean(hsv_image[:,:,1]) / 255
        saturation_ok = saturation >= saturation_threshold

        avg_b = np.mean(image[:,:,0])
        avg_g = np.mean(image[:,:,1])
        avg_r = np.mean(image[:,:,2])
        white_balance_ok = (abs(avg_r - avg_g) < 30 and abs(avg_g - avg_b) < 30)

        distortion_ok = True

        results = {
            "sharpness": {"value": laplacian_var, "ok": bool(sharpness_ok)},
            "brightness": {"value": brightness, "ok": bool(brightness_ok)},
            "resolution": {"value": (height, width), "ok": bool(resolution_ok)},
            "noise": {"value": noise_level, "ok": bool(noise_ok)},
            "format": {"value": img_format, "ok": bool(format_ok)},
            "saturation": {"value": saturation, "ok": bool(saturation_ok)},
            "white_balance": {"value": (avg_r, avg_g, avg_b), "ok": bool(white_balance_ok)},
            "distortion": {"value": distortion_ok, "ok": bool(distortion_ok)},
            "overall": bool(sharpness_ok and brightness_ok and resolution_ok and noise_ok and format_ok and saturation_ok and white_balance_ok and distortion_ok),
        }

        return results
    except Exception as e:
        return {"error": "Error processing the image", "details": str(e)}

# فنكشن لتوليد Grad-CAM
def generate_gradcam(image_bytes, model):
    try:
        img_pil = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        img_pil = img_pil.resize((299, 299))
        img_array = np.array(img_pil) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        last_conv_layer = None
        for layer in model.layers[::-1]:
            if 'conv' in layer.name.lower():
                last_conv_layer = layer
                break
        if last_conv_layer is None:
            return {"error": "No convolutional layer found"}

        grad_model = Model(inputs=[model.inputs], outputs=[last_conv_layer.output, model.output])

        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_array)
            class_idx = tf.argmax(predictions[0])
            loss = predictions[:, class_idx]

        grads = tape.gradient(loss, conv_outputs)[0]
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        conv_outputs = conv_outputs[0]
        heatmap = tf.reduce_mean(tf.multiply(conv_outputs, pooled_grads), axis=-1)
        heatmap = np.maximum(heatmap, 0)
        heatmap = heatmap / np.max(heatmap, axis=(0, 1), keepdims=True)
        heatmap = cv2.resize(heatmap, (299, 299))

        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_HOT)

        original_img = cv2.resize(np.array(img_pil), (299, 299))
        superimposed_img = heatmap * 0.4 + original_img * 0.6
        superimposed_img = np.clip(superimposed_img, 0, 255)
        superimposed_img = np.uint8(superimposed_img)

        _, buffer = cv2.imencode('.png', superimposed_img)
        heatmap_base64 = base64.b64encode(buffer).decode('utf-8')

        return {
            "heatmap": f"data:image/png;base64,{heatmap_base64}"
        }
    except Exception as e:
        return {"error": "Error generating Grad-CAM", "details": str(e)}

# فنكشن التنبؤ بالمرض
def predict_disease(image_bytes):
    try:
        # تحميل الصورة وتحويلها لـ RGB
        img_pil = Image.open(io.BytesIO(image_bytes)).convert('RGB')  # تحويل لـ 3 قنوات
        img_pil = img_pil.resize((299, 299))  # تغيير الحجم لـ 299x299
        img_array = np.array(img_pil) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        prediction = model.predict(img_array)
        predicted_class_idx = int(np.argmax(prediction, axis=1)[0])
        predicted_class = class_labels[predicted_class_idx]
        confidence = float(prediction[0][predicted_class_idx])

        return {
            "class_id": predicted_class_idx,
            "class_name": predicted_class,
            "confidence": confidence
        }
    except Exception as e:
        return {"error": "Error predicting disease", "details": str(e)}

# فنكشن لتفسير سبب المرض باستخدام Gemini API
def explain_disease(prediction_result, gradcam_result):
    try:
        disease_id = prediction_result.get("class_id")
        disease_name = prediction_result.get("class_name")
        confidence = prediction_result.get("confidence")

        if disease_id is None or disease_name is None:
            return {"error": "No prediction result available"}

        prompt = f"""
        You are a medical AI assistant specialized in interpreting medical image analysis results. A deep learning model analyzed a medical image and predicted a disease: {disease_name} (ID {disease_id}) with confidence {confidence:.4f}. The model also generated a Grad-CAM heatmap highlighting the regions of interest in the image that contributed to this prediction.

        Please provide a clear and concise explanation of why the model predicted this disease. Include:
        - A description of what {disease_name} represents in medical imaging (e.g., characteristics of the condition).
        - An explanation of how the highlighted regions in the heatmap (indicating areas of focus) might relate to the disease.
        - Any potential causes or characteristics of the disease that could be inferred from the image analysis.

        Ensure the explanation is professional, medically accurate, and understandable to non-experts. Avoid speculative or overly technical jargon.
        """

        gemini_api_key = "AIzaSyD1FGT_b-nE4M4m1AjlpumpaaoaXZ96m3E"  # استبدل بالـ key الجديد
        gemini_url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
        headers = {"Content-Type": "application/json"}
        data = {
            "contents": [{"parts": [{"text": prompt}]}]
        }
        response = requests.post(f"{gemini_url}?key={gemini_api_key}", json=data, headers=headers)

        if response.status_code == 200:
            explanation = response.json()['candidates'][0]['content']['parts'][0]['text']
        else:
            explanation = f"Could not generate explanation due to API error: {response.text}"

        return {
            "explanation": explanation,
            "disease_name": disease_name,
            "confidence": confidence
        }
    except Exception as e:
        return {"error": "Error generating explanation", "details": str(e)}

# endpoint لرفع الصورة ومعالجتها
@app.route('/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    try:
        image_bytes = file.read()

        quality_check_results = analyze_image_quality(image_bytes)

        if not quality_check_results.get("overall", False):
            return jsonify({
                "message": "Image quality check failed",
                "quality_check": quality_check_results
            }), 400

        prediction_results = predict_disease(image_bytes)
        gradcam_results = generate_gradcam(image_bytes, model)
        explanation_results = explain_disease(prediction_results, gradcam_results)

        return jsonify({
            "message": "Image processed successfully",
            "quality_check": quality_check_results,
            "prediction": prediction_results,
            "gradcam": gradcam_results,
            "explanation": explanation_results
        }), 200

    except Exception as e:
        return jsonify({"error": "Internal server error", "details": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8780)
