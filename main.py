import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox, Toplevel
from PIL import Image, ImageTk
from ultralytics import YOLO
import threading

# MODEL YÜKLEME
# Modeli 'yolov8m.pt' (Medium) yaptık.
print("Loading AI Model (YOLOv8 Medium)... This may take a while for the first time.")
model = YOLO("yolov8m.pt") 
print("Model Loaded Successfully!")

def process_and_count(image_path):
    """
    Uses Medium model for high accuracy and preserves resolution.
    """
     # Görüntüyü Okur
    original_img = cv2.imread(image_path)
    if original_img is None:
        return None, None, 0

    # #BOYUTLANDIRMA AYARI
    # Şimdi sınırı 1920 yaptık. 
    height, width = original_img.shape[:2]
    if width > 1920:
        scale = 1920 / width
        original_img = cv2.resize(original_img, (1920, int(height * scale)))

    # GÖRÜNTÜ İŞLEME
    gray = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # GERÇEK SAYIM (YOLOv8 AI)
    # conf=0.35 -> Orta karar bir güven eşiği.
    # iou=0.50  -> Çakışma önleme.
    # agnostic_nms=True -> Kamyon/Araba karışıklığını önlemek için.
    results = model(
        original_img, 
        classes=[2, 3, 5, 7], 
        conf=0.35, 
        iou=0.50, 
        agnostic_nms=True, 
        verbose=False
    )
    
    # Tespit edilenleri çizer
    annotated_frame = results[0].plot()
    
    # Sayıyı alır
    vehicle_count = len(results[0].boxes)

   # Renk Formatı Düzeltme
    final_img_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
    
    # Ön izleme resmi
    preprocessing_img = blur

    return final_img_rgb, preprocessing_img, vehicle_count

# ARAYÜZ (GUI) FONKSİYONLARI
def upload_image():
    file_path = filedialog.askopenfilename(
        title="Select Image for Analysis",
        filetypes=[("Image Files", "*.jpg *.png *.jpeg")]
    )
    if not file_path:
        return

    lbl_status.config(text="Processing with Medium Model... (May take a moment)", fg="blue")
    root.update() 

    try:
        final_rgb, pre_img, count = process_and_count(file_path)
        
        if final_rgb is None:
            messagebox.showerror("Error", "Could not read image file!")
            lbl_status.config(text="An error occurred.", fg="red")
            return

        # Ana Resmi Göster
        pil_img = Image.fromarray(final_rgb)
        pil_img.thumbnail((700, 500)) 
        img_tk = ImageTk.PhotoImage(pil_img)
        
        lbl_image.config(image=img_tk)
        lbl_image.image = img_tk 
        
        # Sonuç
        lbl_result.config(text=f"Detected Vehicles: {count}", fg="#2E7D32")
        lbl_status.config(text="Processing Complete.", fg="black")

        global current_pre_img
        current_pre_img = pre_img
        btn_show_pre.config(state="normal", bg="#FF9800")

    except Exception as e:
        messagebox.showerror("Error", f"Error: {str(e)}")
        lbl_status.config(text="An error occurred.", fg="red")

def show_preprocessing():
    if current_pre_img is None:
        return
    top = Toplevel()
    top.title("Preprocessing View")
    pil_img_pre = Image.fromarray(current_pre_img)
    pil_img_pre.thumbnail((600, 450))
    img_tk_pre = ImageTk.PhotoImage(pil_img_pre)
    tk.Label(top, image=img_tk_pre).pack(pady=10)
    top.image = img_tk_pre 

# ARAYÜZ TASARIMI
root = tk.Tk()
root.title("Vehicle Counting System - Final Project")
root.geometry("800x750")
root.configure(bg="#f5f5f5")

current_pre_img = None 

frame_header = tk.Frame(root, bg="#3F51B5", pady=15)
frame_header.pack(fill="x")

tk.Label(frame_header, text="Counting Vehicles in Traffic", font=("Segoe UI", 18, "bold"), fg="white", bg="#3F51B5").pack()

frame_btns = tk.Frame(root, bg="#f5f5f5", pady=15)
frame_btns.pack()

btn_upload = tk.Button(frame_btns, text="📷 Upload Image", command=upload_image, font=("Segoe UI", 11, "bold"), bg="#4CAF50", fg="white", padx=15, pady=5)
btn_upload.pack(side=tk.LEFT, padx=10)

btn_show_pre = tk.Button(frame_btns, text="🔍 Preprocessing", command=show_preprocessing, font=("Segoe UI", 11), bg="#BDBDBD", fg="white", padx=15, pady=5, state="disabled")
btn_show_pre.pack(side=tk.LEFT, padx=10)

lbl_result = tk.Label(root, text="Vehicle Count: -", font=("Segoe UI", 22, "bold"), fg="#757575", bg="#f5f5f5")
lbl_result.pack(pady=5)

lbl_status = tk.Label(root, text="Ready", font=("Segoe UI", 10), fg="#757575", bg="#f5f5f5")
lbl_status.pack()

frame_img = tk.Frame(root, bg="white", bd=2, relief="groove")
frame_img.pack(pady=10, padx=20, fill=tk.BOTH, expand=True)
lbl_image = tk.Label(frame_img, bg="white", text="Image Area")
lbl_image.pack(expand=True)

root.mainloop()