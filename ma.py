import logging
from telegram.ext import Updater
from telegram.ext import CommandHandler
from telegram.ext import MessageHandler
from telegram.ext import filters, Application
import tensorflow as tf
import numpy as np

async def start(update, context):
  await update.message.reply_text(
    "FA: لطفا تصویر خود را ارسال کنید و پس از پردازش نتیجه به شما ارسال میشود\n\nEN: Please send us a picture and we will tell you the results after processing"
  )

async def help_command(update, context):
  await update.message.reply_text(
    "یک تصویر ارسال کنید و من نتیجه را به شما خواهم گفت."
  )

def main():
  dp = Application.builder().token("6113782626:AAFI8Gnx1P8i2g8Z0wgONxx8UNoatx3Rrw8").build()
  dp.add_handler(CommandHandler("start", start))
  dp.add_handler(CommandHandler("help", help_command))
  dp.add_handler(MessageHandler(filters.PHOTO, detect))
  dp.run_polling()

def load_model():
  global model
  model = tf.keras.models.load_model("ensemble.h5")
  print("Model loaded")

async def detect(update, context):
  global model
  user = update.message.from_user
  photo_file = update.message.photo[-1].get_file()
  photo_file.download("user_photo.jpg")
  image = tf.image.decode_jpeg("user_photo.jpg")
  image = tf.image.resize(image, [100, 100])
  image = np.asarray(image)
  en_diseases = ["Acne and Rosacea Photos", "Actinic Keratosis Basal Cell Carcinoma and other Malignant Lesions", "Atopic Dermatitis Photos", "Bullous Disease Photos", "Cellulitis Impetigo and other Bacterial Infections", "Eczema Photos", "Exanthems and Drug Eruptions", "Hair Loss Photos Alopecia and other Hair Diseases", "Herpes HPV and other STDs Photos", "Light Diseases and Disorders of Pigmentation", "Lupus and other Connective Tissue diseases", "Melanoma Skin Cancer Nevi and Moles", "Nail Fungus and other Nail Disease", "Poison Ivy Photos and other Contact Dermatitis", "Psoriasis pictures Lichen Planus and related diseases", "Scabies Lyme Disease and other Infestations and Bites", "Seborrheic Keratoses and other Benign Tumors", "Systemic Disease", "Tinea Ringworm Candidiasis and other Fungal Infections", "Urticaria Hives", "Vascular Tumors", "Vasculitis Photos", "Warts Molluscum and other Viral Infections"]
  fa_diseases = [ "عکس های آکنه و روزاسه", "کارسینوم بازال اکتینیک کراتوزیس و سایر ضایعات بدخیم", "عکس های درماتيت پوستی" , "عکس های بیماری بولووس" , "سلولیت ایمپتیگو و سایر عفونت های باکتریایی", "عکسهای اگزما" , "اگزانتم ها و فوران های دارویی" , "عکسهای از دست دادن مو آلوپسی و سایر بیماریهای مو" , "هرپس اچ پي وي و سایر عکسهای مقاربتي" , "بیماریهای سبک و اختلالات رنگدانه", "لوپوس و سایر بیماریهای بافتی همبند" , "ملانوما سرطان پوست نوی و خال" , "قارچ ناخن و سایر بیماری های ناخن" , "عکس های پیچک سمی و سایر درماتیت های تماسی", "تصاویر پسوریازیس لیچن پلان و بیماریهای مرتبط با آن" , "بیماری لایم زخم و سایر آلودگیها و نیشها" , "کراتوزهای سبورئیک و سایر تومورهای خوش خیم" , "بیماری سیستمیک", "کاندیدیازیس کرم حلقوی کچلی و سایر عفونت های قارچی" , "کهیر" , "تومور عروقی" , "عکس واسکولیت" , "زگیل نرم تن و سایر عفونت های ویروسی" ]
  predict = model.predict(image)
  await update.message.reply_text("EN: {0}\n\nFA: {1}".format(en_diseases[np.argmax(predict, -1)], np.argmax(predict, -1)))

if __name__ == "__main__":
  load_model()
  main()

