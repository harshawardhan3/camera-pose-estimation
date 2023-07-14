# Author - Harshawardhan Mane, Anthony Cook, Sherry Liang

#-----------------------------------------------------------------------------------------------------------------------------------------------------------------

# Importing GUI Libraries
import tkinter as tk
from tkinter import ttk
from PIL import ImageTk, Image

# Importing the Supporting Libraries
from scipy.spatial import KDTree
import random
import numpy as np 
import cv2
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import math
import pandas as pd
import json
#from skimage import measure

# Importing supporting functions
from scripts.import_image import import_images
from scripts.find_circles import find_circles
from scripts.errorcompute import point_to_ellipse_distance
from scripts.errorcompute import calculate_residuals
from scripts.math_func import get_center
from scripts.math_func import angle
from scripts.parse_json import parse_json

def on_closing():
    # Clean up any resources or perform necessary actions before closing the window
    window.destroy()  # Close the Tkinter window

class CalibrateGUI:
   
    def __init__(self, master):
        self.master = master
        self.master.title("Calibrate GUI")

        self.property_entries = []
        
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------   
        # Section to handle all the frames in the entire Tkinter window
        
        # Button Frame
        self.button_frame = tk.Frame(window)
        self.button_frame.grid(row=0, column=0, rowspan=2, padx=10, pady=10)
        
        # Image Frame
        self.image_frame = tk.Frame(window)
        self.image_frame.grid(row=0, column=1, padx=10, pady=10)
        
        # Create a Notebook Widget to hold different tabs
        self.notebook = ttk.Notebook(self.image_frame)
        self.notebook.pack(fill="both", expand=True)
        
        # Create the tabs
        self.loaded_images = tk.Frame(self.notebook)
        self.contour_plots = tk.Frame(self.notebook)
        self.connected_components = tk.Frame(self.notebook)
        self.centroid_labels = tk.Frame(self.notebook)

#---------------------------------------------------------------------------------------------------------------------------------------------------------------------       
        # Section to handle all the required buttons
        
        # Import Images Button
        self.button1 = tk.Button(self.button_frame, text="Import Images", command = lambda: import_images(self, self.image_list, self.image_paths()))
        self.button1.pack(pady=10)
        
        # Find Circles Button
        self.button2 = tk.Button(self.button_frame, text="Detect Clusters", command = lambda: self.process_images())
        self.button2.pack(pady=10)
        
        #Plot Contours Button
        self.button3 = tk.Button(self.button_frame, text="Plot Centroids", command = lambda: self.process_contours())
        self.button3.pack(pady=10)
        
        # Connected Components Button
        self.button4 = tk.Button(self.button_frame, text="Detect Targets", command = lambda: self.conn_comp())
        self.button4.pack(pady=10)
        
        # Centroid Labels Button
        self.button5 = tk.Button(self.button_frame, text="Label Centroids", command = lambda: self.label_centroid())
        self.button5.pack(pady=10)
        
        # Tester Button
        self.button6 = tk.Button(self.button_frame, text="Pose in 3D", command = lambda: self.posing_3D())
        self.button6.pack(pady=10)

        # Set all buttons to disabled to start with except first
        self.buttons = [
            self.button1,
            self.button2,
            self.button3,
            self.button4,
            self.button5,
            self.button6
        ]
        for button in self.buttons[1:]:
            button.config(state=tk.DISABLED)
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # Section to handle the Loaded Images Tab
        
        self.image_list = []
        self.title_list = ["Camera 11 - Left", "Camera 11 - Right", "Camera 71", "Camera 72", "Camera 73", "Camera 74"]
        for i, image_title in enumerate(self.title_list):
            canvas_frame = tk.LabelFrame(self.loaded_images, text=image_title)
            canvas_frame.grid(row=i//2, column=i%2, padx=10, pady=10)
            canvas = tk.Canvas(canvas_frame, width=500, height=300)
            canvas.pack(fill="both", expand=True)
            self.image_list.append(canvas)
            
            
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # Section to handle the Contour Plots Tab
        
        # Store contour details of circles from each loaded image
        self.contour_plots_paths = []
        self.contours_list = []
        
        # Create canvas for displaying the plots
        self.contours_canvas_list = []
        self.contours_title_list = ["Camera 11 - Left", "Camera 11 - Right", "Camera 71", "Camera 72", "Camera 73", "Camera 74"]
        for j, contour_title in enumerate(self.contours_title_list):
            contour_canvas_frame = tk.LabelFrame(self.contour_plots, text=contour_title)
            contour_canvas_frame.grid(row=j//2, column=j%2, padx=10, pady=10)
            contour_canvas = tk.Canvas(contour_canvas_frame, width=500, height=300)
            contour_canvas.pack(fill="both", expand=True)
            self.contours_canvas_list.append(contour_canvas)

#---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # Section to handle the Connected Components Tab
        
        # Create canvas for displaying the connected components
        self.connected_components_list = []
        self.connected_components_title_list = ["Camera 11 - Left", "Camera 11 - Right", "Camera 71", "Camera 72", "Camera 73", "Camera 74"]
        for k, cc_title in enumerate(self.connected_components_title_list):
            connected_components_canvas_frame = tk.LabelFrame(self.connected_components, text=cc_title)
            connected_components_canvas_frame.grid(row=k//2, column=k%2, padx=10, pady=10)
            connected_components_canvas = tk.Canvas(connected_components_canvas_frame, width=500, height=300)
            connected_components_canvas.pack(fill="both", expand=True)
            self.connected_components_list.append(connected_components_canvas)

#---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # Section to handle the Centroids with labels
        
        # Create canvas for displaying the centroids with labels
        self.targets_list = []
        self.centroid_labels_list = []
        self.centroid_labels_title_list = ["Camera 11 - Left", "Camera 11 - Right", "Camera 71", "Camera 72", "Camera 73", "Camera 74"]
        for l, cc_title in enumerate(self.centroid_labels_title_list):
            centroid_labels_canvas_frame = tk.LabelFrame(self.centroid_labels, text=cc_title)
            centroid_labels_canvas_frame.grid(row=l//2, column=l%2, padx=10, pady=10)
            centroid_labels_canvas = tk.Canvas(centroid_labels_canvas_frame, width=500, height=300)
            centroid_labels_canvas.pack(fill="both", expand=True)
            self.centroid_labels_list.append(centroid_labels_canvas)

#---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # Section to handle the matched points tab

        # Create the matched points tab
        self.pose_3D_tab = tk.Frame(self.notebook)
        # Create a canvas to display the segmented image
        self.pose_3D_canvas = tk.Canvas(self.pose_3D_tab)
        self.pose_3D_canvas.pack()

#---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # Section to handle the editable properties tab

        # Create the editable properties tab
        self.editable_properties_tab = tk.Frame(self.notebook)

        LRmask_label = tk.Label(self.editable_properties_tab, text="Lower Red mask:")
        LRmask_label.grid(row=0, column=0, padx=10, pady=5, sticky=tk.W)
        # Red Mask entry boxes
        LR_LH_mask_label = tk.Label(self.editable_properties_tab, text="Lower H:")
        LR_LH_mask_label.grid(row=0, column=1, padx=10, pady=5, sticky=tk.W)
        self.Lred_h_lower_entry = tk.Entry(self.editable_properties_tab)
        self.Lred_h_lower_entry.insert(0, '0')
        self.Lred_h_lower_entry.grid(row=0, column=2, padx=10, pady=5)
        LR_UH_mask_label = tk.Label(self.editable_properties_tab, text="Upper H:")
        LR_UH_mask_label.grid(row=0, column=3, padx=10, pady=5, sticky=tk.W)
        self.Lred_h_upper_entry = tk.Entry(self.editable_properties_tab)
        self.Lred_h_upper_entry.insert(0, '40')
        self.Lred_h_upper_entry.grid(row=0, column=4, padx=10, pady=5)
        LR_LS_mask_label = tk.Label(self.editable_properties_tab, text="Lower S:")
        LR_LS_mask_label.grid(row=1, column=1, padx=10, pady=5, sticky=tk.W)
        self.Lred_s_lower_entry = tk.Entry(self.editable_properties_tab)
        self.Lred_s_lower_entry.insert(0, '40')
        self.Lred_s_lower_entry.grid(row=1, column=2, padx=10, pady=5)
        LR_US_mask_label = tk.Label(self.editable_properties_tab, text="Upper S:")
        LR_US_mask_label.grid(row=1, column=3, padx=10, pady=5, sticky=tk.W)
        self.Lred_s_upper_entry = tk.Entry(self.editable_properties_tab)
        self.Lred_s_upper_entry.insert(0, '255')
        self.Lred_s_upper_entry.grid(row=1, column=4, padx=10, pady=5)
        LR_LV_mask_label = tk.Label(self.editable_properties_tab, text="Lower V:")
        LR_LV_mask_label.grid(row=2, column=1, padx=10, pady=5, sticky=tk.W)
        self.Lred_v_lower_entry = tk.Entry(self.editable_properties_tab)
        self.Lred_v_lower_entry.insert(0, '20')
        self.Lred_v_lower_entry.grid(row=2, column=2, padx=10, pady=5)
        LR_UV_mask_label = tk.Label(self.editable_properties_tab, text="Upper V:")
        LR_UV_mask_label.grid(row=2, column=3, padx=10, pady=5, sticky=tk.W)
        self.Lred_v_upper_entry = tk.Entry(self.editable_properties_tab)
        self.Lred_v_upper_entry.insert(0, '255')
        self.Lred_v_upper_entry.grid(row=2, column=4, padx=10, pady=5)

        URmask_label = tk.Label(self.editable_properties_tab, text="Upper Red mask:")
        URmask_label.grid(row=3, column=0, padx=10, pady=5, sticky=tk.W)
        UR_LH_mask_label = tk.Label(self.editable_properties_tab, text="Lower H:")
        UR_LH_mask_label.grid(row=3, column=1, padx=10, pady=5, sticky=tk.W)
        self.Ured_h_lower_entry = tk.Entry(self.editable_properties_tab)
        self.Ured_h_lower_entry.insert(0, '160')
        self.Ured_h_lower_entry.grid(row=3, column=2, padx=10, pady=5)
        UR_UH_mask_label = tk.Label(self.editable_properties_tab, text="Upper H:")
        UR_UH_mask_label.grid(row=3, column=3, padx=10, pady=5, sticky=tk.W)
        self.Ured_h_upper_entry = tk.Entry(self.editable_properties_tab)
        self.Ured_h_upper_entry.insert(0, '210')
        self.Ured_h_upper_entry.grid(row=3, column=4, padx=10, pady=5)
        UR_LS_mask_label = tk.Label(self.editable_properties_tab, text="Lower S:")
        UR_LS_mask_label.grid(row=4, column=1, padx=10, pady=5, sticky=tk.W)
        self.Ured_s_lower_entry = tk.Entry(self.editable_properties_tab)
        self.Ured_s_lower_entry.insert(0, '40')
        self.Ured_s_lower_entry.grid(row=4, column=2, padx=10, pady=5)
        UR_US_mask_label = tk.Label(self.editable_properties_tab, text="Upper S:")
        UR_US_mask_label.grid(row=4, column=3, padx=10, pady=5, sticky=tk.W)
        self.Ured_s_upper_entry = tk.Entry(self.editable_properties_tab)
        self.Ured_s_upper_entry.insert(0, '255')
        self.Ured_s_upper_entry.grid(row=4, column=4, padx=10, pady=5)
        UR_LV_mask_label = tk.Label(self.editable_properties_tab, text="Lower V:")
        UR_LV_mask_label.grid(row=5, column=1, padx=10, pady=5, sticky=tk.W)
        self.Ured_v_lower_entry = tk.Entry(self.editable_properties_tab)
        self.Ured_v_lower_entry.insert(0, '20')
        self.Ured_v_lower_entry.grid(row=5, column=2, padx=10, pady=5)
        UR_UV_mask_label = tk.Label(self.editable_properties_tab, text="Upper V:")
        UR_UV_mask_label.grid(row=5, column=3, padx=10, pady=5, sticky=tk.W)
        self.Ured_v_upper_entry = tk.Entry(self.editable_properties_tab)
        self.Ured_v_upper_entry.insert(0, '255')
        self.Ured_v_upper_entry.grid(row=5, column=4, padx=10, pady=5)

        # Green Mask entry boxes
        Gmask_label = tk.Label(self.editable_properties_tab, text="Green mask:")
        Gmask_label.grid(row=6, column=0, padx=10, pady=5, sticky=tk.W)
        G_LH_mask_label = tk.Label(self.editable_properties_tab, text="Lower H:")
        G_LH_mask_label.grid(row=6, column=1, padx=10, pady=5, sticky=tk.W)
        self.green_h_lower_entry = tk.Entry(self.editable_properties_tab)
        self.green_h_lower_entry.insert(0, '55')
        self.green_h_lower_entry.grid(row=6, column=2, padx=10, pady=5)
        G_UH_mask_label = tk.Label(self.editable_properties_tab, text="Upper H:")
        G_UH_mask_label.grid(row=6, column=3, padx=10, pady=5, sticky=tk.W)
        self.green_h_upper_entry = tk.Entry(self.editable_properties_tab)
        self.green_h_upper_entry.insert(0, '90')
        self.green_h_upper_entry.grid(row=6, column=4, padx=10, pady=5)
        G_LS_mask_label = tk.Label(self.editable_properties_tab, text="Lower S:")
        G_LS_mask_label.grid(row=7, column=1, padx=10, pady=5, sticky=tk.W)
        self.green_s_lower_entry = tk.Entry(self.editable_properties_tab)
        self.green_s_lower_entry.insert(0, '29')
        self.green_s_lower_entry.grid(row=7, column=2, padx=10, pady=5)
        G_US_mask_label = tk.Label(self.editable_properties_tab, text="Upper S:")
        G_US_mask_label.grid(row=7, column=3, padx=10, pady=5, sticky=tk.W)
        self.green_s_upper_entry = tk.Entry(self.editable_properties_tab)
        self.green_s_upper_entry.insert(0, '255')
        self.green_s_upper_entry.grid(row=7, column=4, padx=10, pady=5)
        G_LV_mask_label = tk.Label(self.editable_properties_tab, text="Lower V:")
        G_LV_mask_label.grid(row=8, column=1, padx=10, pady=5, sticky=tk.W)
        self.green_v_lower_entry = tk.Entry(self.editable_properties_tab)
        self.green_v_lower_entry.insert(0, '0')
        self.green_v_lower_entry.grid(row=8, column=2, padx=10, pady=5)
        G_UV_mask_label = tk.Label(self.editable_properties_tab, text="Upper V:")
        G_UV_mask_label.grid(row=8, column=3, padx=10, pady=5, sticky=tk.W)
        self.green_v_upper_entry = tk.Entry(self.editable_properties_tab)
        self.green_v_upper_entry.insert(0, '171')
        self.green_v_upper_entry.grid(row=8, column=4, padx=10, pady=5)

        # Blue Mask entry boxes
        Bmask_label = tk.Label(self.editable_properties_tab, text="Blue mask:")
        Bmask_label.grid(row=9, column=0, padx=10, pady=5, sticky=tk.W)
        B_LH_mask_label = tk.Label(self.editable_properties_tab, text="Lower H:")
        B_LH_mask_label.grid(row=9, column=1, padx=10, pady=5, sticky=tk.W)
        self.blue_h_lower_entry = tk.Entry(self.editable_properties_tab)
        self.blue_h_lower_entry.insert(0, '90')
        self.blue_h_lower_entry.grid(row=9, column=2, padx=10, pady=5)
        B_UH_mask_label = tk.Label(self.editable_properties_tab, text="Upper H:")
        B_UH_mask_label.grid(row=9, column=3, padx=10, pady=5, sticky=tk.W)
        self.blue_h_upper_entry = tk.Entry(self.editable_properties_tab)
        self.blue_h_upper_entry.insert(0, '150')
        self.blue_h_upper_entry.grid(row=9, column=4, padx=10, pady=5)
        B_LS_mask_label = tk.Label(self.editable_properties_tab, text="Lower S:")
        B_LS_mask_label.grid(row=10, column=1, padx=10, pady=5, sticky=tk.W)
        self.blue_s_lower_entry = tk.Entry(self.editable_properties_tab)
        self.blue_s_lower_entry.insert(0, '100')
        self.blue_s_lower_entry.grid(row=10, column=2, padx=10, pady=5)
        B_US_mask_label = tk.Label(self.editable_properties_tab, text="Upper S:")
        B_US_mask_label.grid(row=10, column=3, padx=10, pady=5, sticky=tk.W)
        self.blue_s_upper_entry = tk.Entry(self.editable_properties_tab)
        self.blue_s_upper_entry.insert(0, '255')
        self.blue_s_upper_entry.grid(row=10, column=4, padx=10, pady=5)
        B_LV_mask_label = tk.Label(self.editable_properties_tab, text="Lower V:")
        B_LV_mask_label.grid(row=11, column=1, padx=10, pady=5, sticky=tk.W)
        self.blue_v_lower_entry = tk.Entry(self.editable_properties_tab)
        self.blue_v_lower_entry.insert(0, '40')
        self.blue_v_lower_entry.grid(row=11, column=2, padx=10, pady=5)
        B_UV_mask_label = tk.Label(self.editable_properties_tab, text="Upper V:")
        B_UV_mask_label.grid(row=11, column=3, padx=10, pady=5, sticky=tk.W)
        self.blue_v_upper_entry = tk.Entry(self.editable_properties_tab)
        self.blue_v_upper_entry.insert(0, '255')
        self.blue_v_upper_entry.grid(row=11, column=4, padx=10, pady=5)

        empty_label = tk.Label(self.editable_properties_tab, text=" ")
        empty_label.grid(row=12, column=0, padx=10, pady=5, sticky=tk.W)

        cluster_filters_label = tk.Label(self.editable_properties_tab, text="Cluster Filters:")
        cluster_filters_label.grid(row=13, column=0, padx=10, pady=5, sticky=tk.W)
        min_area_label = tk.Label(self.editable_properties_tab, text="Min area:")
        min_area_label.grid(row=13, column=1, padx=10, pady=5, sticky=tk.W)
        self.min_area_entry = tk.Entry(self.editable_properties_tab)
        self.min_area_entry.insert(0, '1')
        self.min_area_entry.grid(row=13, column=2, padx=10, pady=5)
        self.max_area_entry = tk.Entry(self.editable_properties_tab)
        max_area_label = tk.Label(self.editable_properties_tab, text="Max area:")
        max_area_label.grid(row=13, column=3, padx=10, pady=5, sticky=tk.W)
        self.max_area_entry.insert(0, '250')
        self.max_area_entry.grid(row=13, column=4, padx=10, pady=5)
        aspect_ratio_label = tk.Label(self.editable_properties_tab, text="Aspect ratio:")
        aspect_ratio_label.grid(row=14, column=1, padx=10, pady=5, sticky=tk.W)
        self.aspect_ratio_entry = tk.Entry(self.editable_properties_tab)
        self.aspect_ratio_entry.insert(0, '2.5')
        self.aspect_ratio_entry.grid(row=14, column=2, padx=10, pady=5)
        circ_threshold_label = tk.Label(self.editable_properties_tab, text="Circularity:")
        circ_threshold_label.grid(row=15, column=1, padx=10, pady=5, sticky=tk.W)
        self.circ_threshold_entry = tk.Entry(self.editable_properties_tab)
        self.circ_threshold_entry.insert(0, '0.2')
        self.circ_threshold_entry.grid(row=15, column=2, padx=10, pady=5)

        empty_label = tk.Label(self.editable_properties_tab, text=" ")
        empty_label.grid(row=16, column=0, padx=10, pady=5, sticky=tk.W)

        target_filters_label = tk.Label(self.editable_properties_tab, text="Target Filters:")
        target_filters_label.grid(row=17, column=0, padx=10, pady=5, sticky=tk.W)
        error_thresh_label = tk.Label(self.editable_properties_tab, text="Error threshold:")
        error_thresh_label.grid(row=17, column=1, padx=10, pady=5, sticky=tk.W)
        self.error_thresh_entry = tk.Entry(self.editable_properties_tab)
        self.error_thresh_entry.insert(0, '0.1')
        self.error_thresh_entry.grid(row=17, column=2, padx=10, pady=5)
        residual_thresh_label = tk.Label(self.editable_properties_tab, text="Residual threshold:")
        residual_thresh_label.grid(row=18, column=1, padx=10, pady=5, sticky=tk.W)
        self.residual_thresh_entry = tk.Entry(self.editable_properties_tab)
        self.residual_thresh_entry.insert(0, '-0.3')
        self.residual_thresh_entry.grid(row=18, column=2, padx=10, pady=5)

#---------------------------------------------------------------------------------------------------------------------------------------------------------------------            

#---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # Section to add the tabs to the notebook
        
        self.notebook.add(self.loaded_images, text="Cluster Detection")
        self.notebook.add(self.contour_plots, text="Centroid Plots")
        self.notebook.add(self.connected_components, text="Target Detection")
        self.notebook.add(self.centroid_labels, text="Centroids with Labels")
        self.notebook.add(self.pose_3D_tab, text="Pose 3D")
        self.notebook.add(self.editable_properties_tab, text="Hyperparameters")

                
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------          
    # Section to Process Images
    
    def process_images(self):
        # Enable button at index 2
        self.buttons[2].config(state=tk.NORMAL)

        for canvas, image_path in zip(self.image_list, self.image_paths()):
            if canvas.image:
                # Convert PIL Image to OpenCV format
                pil_image = Image.open(image_path)
                open_cv_image = np.array(pil_image)
                open_cv_image = cv2.cvtColor(open_cv_image, cv2.COLOR_RGB2BGR)
                
                # Call find_circles function on the image
                processed_image, contours = find_circles(open_cv_image, 
                                                         min_area=int(self.min_area_entry.get()), 
                                                         max_area=int(self.max_area_entry.get()), 
                                                         aspect_ratio_threshold=float(self.aspect_ratio_entry.get()), 
                                                         circularity_threshold=float(self.circ_threshold_entry.get()),
                                                         lr1=[int(self.Lred_h_lower_entry.get()), int(self.Lred_s_lower_entry.get()), int(self.Lred_v_lower_entry.get())], 
                                                         ur1=[int(self.Lred_h_upper_entry.get()), int(self.Lred_s_upper_entry.get()), int(self.Lred_v_upper_entry.get())], 
                                                         lr2=[int(self.Ured_h_lower_entry.get()), int(self.Ured_s_lower_entry.get()), int(self.Ured_v_lower_entry.get())], 
                                                         ur2=[int(self.Ured_h_upper_entry.get()), int(self.Ured_s_upper_entry.get()), int(self.Ured_v_upper_entry.get())],
                                                         lb=[int(self.blue_h_lower_entry.get()), int(self.blue_s_lower_entry.get()), int(self.blue_v_lower_entry.get())], 
                                                         ub=[int(self.blue_h_upper_entry.get()), int(self.blue_s_upper_entry.get()), int(self.blue_v_upper_entry.get())], 
                                                         lg=[int(self.green_h_lower_entry.get()), int(self.green_s_lower_entry.get()),int(self.green_v_lower_entry.get())], 
                                                         ug=[int(self.green_h_upper_entry.get()), int(self.green_s_upper_entry.get()),int(self.green_v_upper_entry.get())]
                                                         )
                
                # Append the contours to a list for future use ->>>>>>>>>>>>>>>>>>
                self.contours_list.append(contours)
                
                
                # Plot the contour and save it as image ->>>>>>>>>>>>>>>>>>>
                fig, ax = plt.subplots()
                for contour, color, cX, cY in contours:
                    ax.scatter(cX, cY, color=color, label=color)

                # Add axis labels and legend ->>>>>>>>>>>>>>>>>>>>>
                ax.invert_yaxis()
                ax.set_aspect('equal')
                ax.set_xlabel('Centroid X Coordinate')
                ax.set_ylabel('Centroid Y Coordinate')
                
                # Save the contour graph as image ->>>>>>>>>>>>>>>>>
                filename = "plots/"+"contour_plot_" + image_path[16:-1] + ".png"
                fig.savefig(filename)
                self.contour_plots_paths.append(filename)                

                # Resize the processed image to the canvas size
                canvas_width, canvas_height = canvas.winfo_width(), canvas.winfo_height()
                resized_image = cv2.resize(processed_image, (canvas_width, canvas_height))

                # Convert the resized image back to PIL Image format
                pil_image = Image.fromarray(cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB))

                # Update the canvas with the processed image
                tk_image = ImageTk.PhotoImage(pil_image)
                canvas.create_image(0, 0, anchor=tk.NW, image=tk_image)
                canvas.image = pil_image
                canvas.tk_image = tk_image

        
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------      
    # Import the saved plots into the canvas in Plots tab and display them
    
    def process_contours(self):
        # Enable button at index 3
        self.buttons[3].config(state=tk.NORMAL)
        # Change to the contour_plots tab
        self.notebook.select(self.contour_plots)  

        self.contour_plots.after(100, lambda: import_images(self, self.contours_canvas_list, self.contour_plots_paths))
        
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------      
    # Perform component connection
    
    def conn_comp(self):
        # Enable button at index 4
        self.buttons[4].config(state=tk.NORMAL)
        # Change to connected components tab
        self.notebook.select(self.connected_components)

        images = []
        image_paths = self.image_paths()
        for path in image_paths:
            image = cv2.imread(path)
            images.append([image, path])
        
        conn_comp_path = []
        
        # Process connected components for each image
        for image, contours_with_color in zip(images,self.contours_list):
            
            # Get the cluster centroids from the contours previously stored
            centroids = []
            for contour, color, cX, cY in contours_with_color:
                centroids.append((color, cX, cY, 0))
            
            # Initialise required variables
            target_number = 0
            centroid_pool = centroids
            targets = []
            
            # Copy the image to a variable for further use
            image_with_ellipse = image[0].copy()
            
            
            # Select a random point
            for point in centroids:

                # Stop the loop if there are not enough centroids to make a target
                if len(centroid_pool) < 6:
                    break
                
                # Skip the point if it isn't in the pool or if the point isn't blue
                if point not in centroid_pool or point[0] != 'Blue':
                    continue

                # Compute the nearest neighbors of the random point
                kdtree = KDTree([c[1:3] for c in centroid_pool])
                _, indices = kdtree.query(point[1:3], k=6)  # include the random point itself

                # Extract the coordinates of the nearest neighbors
                points = [centroid_pool[i] for i in indices]

                # Populate points then fit an ellipse to the nearest neighbors
                e_points = np.array([p[1:3] for p in points], dtype=np.float32)
                ellipse = cv2.fitEllipse(e_points)

                # Calculate the errors for the 6 points
                points = [[p[0], float(p[1]), float(p[2]), int(p[3])] for p in points]
                errors = [point_to_ellipse_distance(p[1:3], ellipse) for p in points]

                # Calcualte the residuals for the 6 points
                residuals = calculate_residuals(points, ellipse)
                
                
                # -------------------------------------------------------->
                
                number_of_blue = len([p for p in points if p[0] == 'Blue'])
                ellipse_shape = abs(1 - (ellipse[1][1] / ellipse[1][0]))
                errors_ok = all(e < float(self.error_thresh_entry.get()) for e in errors)
                residuals_ok = all(r < float(self.residual_thresh_entry.get()) for r in residuals)

                if errors_ok and residuals_ok and number_of_blue == 1:
                    # Add ellipse to image
                    cv2.ellipse(image_with_ellipse, ellipse, (0, 255, 0), 2)

                    # Add points to target list
                    target_number += 1
                    for point in points:
                        point = list(point)
                        point[3] = target_number
                        targets.append(point)

                    # Remove points from centroid pool
                    points = [(t[0], float(t[1]), float(t[2]), t[3]) for t in points]
                    centroid_pool = [centroid for centroid in centroid_pool if centroid not in points]
            
            # Define colors for each target number
            colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255), (128, 0, 0), (0, 128, 0)]

            # Loop through the targets and draw them on the image
            for target in targets:
                # Get the target number and its corresponding color
                tn = target[3]
                color = colors[tn % len(colors)]

                # Draw a circle at the target point with its corresponding color
                cv2.circle(image_with_ellipse, (int(target[1]), int(target[2])), 5, color, -1)
            
            self.targets_list.append(targets)   
                
            # Convert this overlayed image from BGR to RGB before saving it (normally it isn't required while displaying it using cv2.imshow())
            rgb_variant = cv2.cvtColor(image_with_ellipse, cv2.COLOR_BGR2RGB)
            
            
            img = Image.fromarray(rgb_variant)
            filepath = "connected_components/"+"conncomp_"+image[1][16:-1]+".png"
            conn_comp_path.append(filepath)
            img.save(filepath)  

        self.connected_components.after(100, lambda: import_images(self, self.connected_components_list, conn_comp_path))    
     
     
    
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------      
# Generate centroid labels
    def label_centroid(self):
        # Enable button at index 5
        self.buttons[5].config(state=tk.NORMAL)
        # Change to centroid labels tab
        self.notebook.select(self.centroid_labels)
        self.alltargets = []
        targets_ordered_list = []
        
        for targets in self.targets_list:
            
            targets_ordered = []
            # Run the loop for every set of 6 values
            for i in range(int(len(targets) / 6)):
                
                # Filter points for current set of 6
                target_points = targets[i*6: (i+1)*6]

                center = get_center(target_points)
                angles = [angle(center, point) for point in target_points]
                sorted_points = [point for _, point in sorted(zip(angles, target_points), reverse=False)]

                blue_index = [i for i, point in enumerate(sorted_points) if point[0] == 'Blue'][0]
                ordered_points = sorted_points[blue_index:] + sorted_points[:blue_index]

                color_abbreviations = ''.join([c[0][0] for c in ordered_points if c[0] != 'Blue'])

                # Create the point label and add it to the list
                for i, point in enumerate(ordered_points):
                    hexa_target = f'HexaTarget_{color_abbreviations}_{i+1}'
                    point.append(hexa_target)

                targets_ordered.extend(ordered_points)
            
            targets_ordered_list.append(targets_ordered)
                
        images = []
        image_paths = self.image_paths()
        for path in image_paths:
            image = cv2.imread(path)
            images.append([image, path])
            
        # Define colors for plotting
        colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255), (255, 0, 255), (255, 255, 0)]
        
        labeled_image_path = []
        i = 0

        for targets_ordered, image in zip(targets_ordered_list, images):
            i += 1
            
            # Save the image to a variable for further use
            image_with_targets = image[0].copy()
            
            # Loop through the targets and draw them on the image
            
            for target in targets_ordered:
                # Get the target number and its corresponding color
                tn = target[3]
                color = colors[tn % len(colors)]

                # Draw a circle at the target point with its corresponding color
                cv2.circle(image_with_targets, (int(target[1]), int(target[2])), 2, color, -1)

                # Add label text with the HexaTarget item
                label = target[4]
                cv2.putText(image_with_targets, label, (int(target[1]) + 2, int(target[2])),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.2, color, 1)
                
                # Build self.alltargets for use in other parts of the code
                t_entry = list(target)
                t_entry.append(i)
                self.alltargets.append(t_entry)        
            
            # Convert the image to RGB for displaying with matplotlib
            rgb_variant = cv2.cvtColor(image_with_targets, cv2.COLOR_BGR2RGB)
            
            # Save the labeled images
            img = Image.fromarray(rgb_variant)
            filepath = "labeled/"+"labeled_"+image[1][16:-1]+".png"
            labeled_image_path.append(filepath)
            img.save(filepath)
        
        # Import the labeled images to out canvas
        self.centroid_labels.after(100, lambda: import_images(self, self.centroid_labels_list, labeled_image_path)) 
        
        
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------      
    # Extract the corresponding 3D coordinates of the 2D Feature points using triangulation
    def triangulation(self):
        json_paths = self.json_paths()
        parameters = []
        for camera in json_paths:
            for camera_view, path in json_paths[camera].items():
                parameter = parse_json(path)
                parameters.append(parameter)
        
        print(self.alltargets)

#---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    def posing_3D(self):
        # Change to 3D pose tab
        self.notebook.select(self.pose_3D_tab)

        def decompose_essential_matrix(E):
            U, S, Vt = np.linalg.svd(E)

            W = np.array([
                [0, -1, 0],
                [1, 0, 0],
                [0, 0, 1]
            ])

            R1 = U @ W @ Vt
            R2 = U @ W.T @ Vt

            if np.linalg.det(R1) < 0:
                R1 = -R1
            if np.linalg.det(R2) < 0:
                R2 = -R2

            T = U[:, 2]
            #T = T.reshape(3, 1)

            return (R1, R2, T)
        
        def compute_fundamental_matrix(points1, points2):
            assert points1.shape[0] == points2.shape[0], "Number of points should be the same."
            assert points1.shape[0] >= 8, "At least 8 points are required to compute the fundamental matrix."

            # Normalize the points
            points1_norm, T1 = normalize_points(points1)
            points2_norm, T2 = normalize_points(points2)

            # Construct the design matrix
            A = np.zeros((points1.shape[0], 9))
            for i in range(points1.shape[0]):
                x1, y1 = points1_norm[i]
                x2, y2 = points2_norm[i]
                A[i] = [x1 * x2, x1 * y2, x1, y1 * x2, y1 * y2, y1, x2, y2, 1]

            # Solve the linear system using SVD
            _, _, V = np.linalg.svd(A)
            F = V[-1].reshape(3, 3)

            # Enforce rank-2 constraint
            U, S, V = np.linalg.svd(F)
            S[-1] = 0
            F = U @ np.diag(S) @ V

            # Denormalize the fundamental matrix
            F = T2.T @ F @ T1

            return F

        def normalize_points(points):
            centroid = np.mean(points, axis=0)
            dist = np.sqrt(2) / np.std(points, axis=0)
            T = np.array([[dist[0], 0, -dist[0] * centroid[0]],
                        [0, dist[1], -dist[1] * centroid[1]],
                        [0, 0, 1]])
            points_homogeneous = np.concatenate([points, np.ones((points.shape[0], 1))], axis=1)
            points_normalized = (T @ points_homogeneous.T).T[:, :2]

            return points_normalized, T

        # Define target points        
        camera11_left = [record for record in self.alltargets if record[5] == 1]
        camera11_right = [record for record in self.alltargets if record[5] == 2]
        left11_xy = [[sublist[i] for i in range(len(sublist)) if i in (1, 2)] for sublist in camera11_left]
        right11_xy = [[sublist[i] for i in range(len(sublist)) if i in (1, 2)] for sublist in camera11_right]
        obj_points = [[sublist[i] for i in range(len(sublist)) if i in (1, 2)] for sublist in camera11_left]

        # Iterate over the main list
        for sublist in obj_points:
            # Insert 0 at the third position
            sublist.insert(2, 0)

        # Read JSON file and load data
        with open("data/camera parameters/zedLeft720p.json", "r") as file:
            data = json.load(file)

        # Extract focal length (f), and principal point coordinates (cx, cy) from the data
        f = data["f"]["val"]
        cx = data["cx"]["val"]
        cy = data["cy"]["val"]

        # Create 3x3 intrinsic matrix 1
        K1 = np.array([[f, 0, cx],
                        [0, f, cy],
                        [0, 0, 1]], dtype=np.float32)
        
        # Create 3x3 intrinsic matrix 2
        K2 = np.array([[f, 0, cx],
                        [0, f, cy],
                        [0, 0, 1]], dtype=np.float32)

        # Compute Fundamental matrix
        F = compute_fundamental_matrix(np.float32(left11_xy), np.float32(right11_xy))

        # Compute the essential matrix
        E = K2.T @ F @ K1

        # Decompose the essential matrix to obtain the rotation and translation
        R1, R2, T = decompose_essential_matrix(E)

        # Define rvec, tvec, and dist here
        _, rvec, tvec = cv2.solvePnP(np.array(obj_points), np.array(left11_xy), K1, None)
        dist = np.array([0, 0, 0, 0, 0], dtype=np.float32)

        # Load reference points
        ref_points = np.float32(left11_xy)

        # Undistort reference points
        und_ref = cv2.undistortPoints(ref_points, K1, dist)

        # Load the object points
        object_points = np.float32(obj_points)

        # Project points
        projected_points, _ = cv2.projectPoints(object_points, rvec, tvec, K1, dist)

        # Convert back to numpy array and remove single-dimensional entries
        projected_points = np.squeeze(np.array(projected_points))

        # Calculate distance between original points and projected points
        distances = np.linalg.norm(projected_points - ref_points, axis=1)
        mean_distance = np.mean(distances)

        # Extract the camera position from the inverse of the T matrix
        camera_position = -T

        # Retrieve the third item in the list of lists
        third_item = [item[0] for item in camera11_left]

        # Define a mapping of third item values to colors
        color_mapping = {
            'Blue': 'blue',
            'Red': 'red',
            'Green': 'green'
        }

        # Create a list of colors based on the third item values
        colors = [color_mapping[item] for item in third_item]

        # Set up the 3D plot
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Set custom view
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_box_aspect([1, 1, 1])

        # Object points scatter plot
        ax.scatter(object_points[:, 0], object_points[:, 1], object_points[:, 2], c=colors, s=3, label='Object points')

        # Camera position scatter plot
        ax.scatter([camera_position[0]], [camera_position[1]], [camera_position[2]], c='black', s=8, label='Camera 11')

        # Show the legend
        ax.legend()

        # Embed the matplotlib figure into the tkinter canvas
        canvas = FigureCanvasTkAgg(fig, master=self.pose_3D_canvas)
        canvas.draw()
        self.pose_3D_canvas.after(100, lambda: import_images(self, canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)))
        
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------              
    # Store all the paths for required data
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------      
    def camera_image_paths(self):
        
        # Define the camera image paths (All the paths, will filter through these in specific functions)
        camera_image_paths = {
            "camera_11":{"camera_left":"data\\camera 11\\2022_12_15_15_51_19_927_rgb_left.png", "camera_right":"data\\camera 11\\2022_12_15_15_51_19_927_rgb_right.png"},
            "camera_71":{"camera_left":"data\\camera 71\\2022_12_15_15_51_19_944_left.png", "camera_rgb":"data\\camera 71\\2022_12_15_15_51_19_944_rgb.png", "camera_right":"data\\camera 71\\2022_12_15_15_51_19_944_right.png"},
            "camera_72":{"camera_left":"data\\camera 72\\2022_12_15_15_51_19_956_left.png", "camera_rgb":"data\\camera 72\\2022_12_15_15_51_19_956_rgb.png", "camera_right":"data\\camera 72\\2022_12_15_15_51_19_956_right.png"},
            "camera_73":{"camera_left":"data\\camera 73\\2022_12_15_15_51_19_934_left.png", "camera_rgb":"data\\camera 73\\2022_12_15_15_51_19_934_rgb.png", "camera_right":"data\\camera 73\\2022_12_15_15_51_19_934_right.png"},
            "camera_74":{"camera_left":"data\\camera 74\\2022_12_15_15_51_19_951_left.png", "camera_rgb":"data\\camera 74\\2022_12_15_15_51_19_951_rgb.png", "camera_right":"data\\camera 74\\2022_12_15_15_51_19_951_right.png"}
        }
        
        return camera_image_paths
    
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------      
    def json_paths(self):
        
        # Define the camera parameter paths (All the paths, will filter through these in specific functions)
        json_paths = {
            "camera_11":{"camera_left":"data\\camera parameters\\zedLeft720p.json", "camera_right":"data\\camera parameters\\zedRight720p.json"},
            "camera_71":{"camera_left":"data\\camera parameters\\realsense71Left.json", "camera_rgb":"data\\camera parameters\\realsense71RGB.json", "camera_right":"data\\camera parameters\\realsense71Right.json"},
            "camera_72":{"camera_left":"data\\camera parameters\\realsense72Left.json", "camera_rgb":"data\\camera parameters\\realsense72RGB.json", "camera_right":"data\\camera parameters\\realsense72Right.json"},
            "camera_73":{"camera_left":"data\\camera parameters\\realsense73Left.json", "camera_rgb":"data\\camera parameters\\realsense73RGB.json", "camera_right":"data\\camera parameters\\realsense73Right.json"},
            "camera_74":{"camera_left":"data\\camera parameters\\realsense74Left.json", "camera_rgb":"data\\camera parameters\\realsense74RGB.json", "camera_right":"data\\camera parameters\\realsense74Right.json"}
        }
        
        return json_paths
        
            
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------      
    # Define Image Paths for the images to be loaded
    
    def image_paths(self):
        
        paths = self.camera_image_paths()
        # Define the image paths
        image_paths = []
        image_paths.append(paths["camera_11"]["camera_left"])
        image_paths.append(paths["camera_11"]["camera_right"])
        image_paths.append(paths["camera_71"]["camera_rgb"])
        image_paths.append(paths["camera_72"]["camera_rgb"])
        image_paths.append(paths["camera_73"]["camera_rgb"])
        image_paths.append(paths["camera_74"]["camera_rgb"])
        
        return image_paths

#---------------------------------------------------------------------------------------------------------------------------------------------------------------------      
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------      

        
# Instantiate a window from the class

if __name__ == "__main__":
    window = tk.Tk()
    window.geometry("1920x1080")
    gui = CalibrateGUI(window)
    window.mainloop()
        
        
