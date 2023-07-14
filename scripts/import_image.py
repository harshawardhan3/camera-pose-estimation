import os
from PIL import Image, ImageTk
import tkinter as tk

def import_images(main_app, canvas_list, image_paths):
    # Enable button at index 1
    main_app.buttons[1].config(state=tk.NORMAL)

    for i, path in enumerate(image_paths):
        img = Image.open(path)
        img = img.resize((canvas_list[i].winfo_width(), canvas_list[i].winfo_height()))
        img = ImageTk.PhotoImage(img)
        canvas_list[i].create_image(0, 0, anchor="nw", image=img)
        canvas_list[i].image = img  # Keep a reference to the image to prevent garbage collection
