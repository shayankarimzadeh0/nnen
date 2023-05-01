
# A very simple Flask Hello World app for you to get started with...
import tensorflow as tf
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import cv2
import os
import numpy as np
app = Flask("Tensorflow")
model = tf.keras.models.load_model("ensemble.h5")
def predict(temp_file):
    global model
    os.remove(temp_file, ".image")
    test_image = cv2.imread(".image")
    os.remove(".image")
    test_image = cv2.resize(test_image, (100, 100))
    test_image = np.asarray(test_image)
    test_image = np.expand_dims(test_image, axis = 0)
    pr = model.predict(test_image)
    en_diseases = ["Acne and Rosacea Photos", "Actinic Keratosis Basal Cell Carcinoma and other Malignant Lesions", "Atopic Dermatitis Photos", "Bullous Disease Photos", "Cellulitis Impetigo and other Bacterial Infections", "Eczema Photos", "Exanthems and Drug Eruptions", "Hair Loss Photos Alopecia and other Hair Diseases", "Herpes HPV and other STDs Photos", "Light Diseases and Disorders of Pigmentation", "Lupus and other Connective Tissue diseases", "Melanoma Skin Cancer Nevi and Moles", "Nail Fungus and other Nail Disease", "Poison Ivy Photos and other Contact Dermatitis", "Psoriasis pictures Lichen Planus and related diseases", "Scabies Lyme Disease and other Infestations and Bites", "Seborrheic Keratoses and other Benign Tumors", "Systemic Disease", "Tinea Ringworm Candidiasis and other Fungal Infections", "Urticaria Hives", "Vascular Tumors", "Vasculitis Photos", "Warts Molluscum and other Viral Infections"]
    fa_diseases = [ "عکس های آکنه و روزاسه", "کارسینوم بازال اکتینیک کراتوزیس و سایر ضایعات بدخیم", "عکس های درماتيت پوستی" , "عکس های بیماری بولووس" , "سلولیت ایمپتیگو و سایر عفونت های باکتریایی", "عکسهای اگزما" , "اگزانتم ها و فوران های دارویی" , "عکسهای از دست دادن مو آلوپسی و سایر بیماریهای مو" , "هرپس اچ پي وي و سایر عکسهای مقاربتي" , "بیماریهای سبک و اختلالات رنگدانه", "لوپوس و سایر بیماریهای بافتی همبند" , "ملانوما سرطان پوست نوی و خال" , "قارچ ناخن و سایر بیماری های ناخن" , "عکس های پیچک سمی و سایر درماتیت های تماسی", "تصاویر پسوریازیس لیچن پلان و بیماریهای مرتبط با آن" , "بیماری لایم زخم و سایر آلودگیها و نیشها" , "کراتوزهای سبورئیک و سایر تومورهای خوش خیم" , "بیماری سیستمیک", "کاندیدیازیس کرم حلقوی کچلی و سایر عفونت های قارچی" , "کهیر" , "تومور عروقی" , "عکس واسکولیت" , "زگیل نرم تن و سایر عفونت های ویروسی" ]
    rates = np.fliplr( np.sort(pr, axis=1)[:,-5:] )
    categories = np.fliplr( np.argsort(pr, axis=1)[:,-5:] )
    en_result = ""
    fa_result = ""
    for i in range(np.count_nonzero(rates)):
      for rate in rates:
        for category in categories:
          print(en_diseases[category[i]], " : ", rate[i])
          en_result = en_result + str(en_diseases[category[i]]) + " : " + str(rate[i]) + "\n"
          fa_result = fa_result + str(fa_diseases[category[i]]) + " : " + str(rate[i]) + "\n"
    return en_result, fa_result

@app.route('/')
def main():
    return render_template("index.html")

@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file_req']
        file.save(f"/tmp/{secure_filename(file.filename)}")
        return predict(f"/tmp/{secure_filename(file.filename)}")
app.run(host="127.0.0.1", port=80)
