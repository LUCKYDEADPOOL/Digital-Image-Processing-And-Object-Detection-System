import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
from PIL import Image, ImageTk
import cv2, numpy as np
from collections import Counter
import matplotlib.pyplot as plt
from ultralytics import YOLO

input_image = None
output_image = None
draw_image = None
model = YOLO('yolov8n.pt')

frame_w, frame_h = 400, 400

def show_on_canvas_input(img):
    global photo_input
    im = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    im_pil = Image.fromarray(im).resize((frame_w, frame_h))
    photo_input = ImageTk.PhotoImage(im_pil)
    canvas_input.delete("all")
    canvas_input.create_image(0, 0, anchor='nw', image=photo_input)

def show_on_canvas_output(img):
    global photo_output
    im = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    im_pil = Image.fromarray(im).resize((frame_w, frame_h))
    photo_output = ImageTk.PhotoImage(im_pil)
    canvas_output.delete("all")
    canvas_output.create_image(0, 0, anchor='nw', image=photo_output)

def open_image():
    global input_image, output_image
    path = filedialog.askopenfilename(filetypes=[('Images','*.jpg;*.png;*.bmp')])
    
    if path:
        input_image = cv2.imread(path)
        output_image = input_image.copy()
        show_on_canvas_input(input_image)
        show_on_canvas_output(output_image)

def reset():
    global input_image, output_image
    input_image = None
    output_image = None
    canvas_input.delete("all")
    canvas_output.delete("all")

def save_output():
    global output_image
    
    if output_image is None:
        messagebox.showerror('Error', 'No output image to save.')
        return
    
    file_path = filedialog.asksaveasfilename(
        defaultextension=".png",
        filetypes=[("PNG files", "*.png"), ("JPEG files", "*.jpg")]
    )
   
    if file_path:
        cv2.imwrite(file_path, output_image)
        messagebox.showinfo('Saved', f'Image saved to {file_path}')

# FILTERS
def apply_filter(name):
    global output_image
    
    if output_image is None:
        return
    
    img = output_image.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    if name == 'gaussian':
        img = cv2.GaussianBlur(img, (9, 9), 0)
    
    elif name == 'canny':
        img = cv2.cvtColor(cv2.Canny(img, 100, 200), cv2.COLOR_GRAY2BGR)
    
    elif name == 'sobel':
        sx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, 3)
        sy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, 3)
        mag = cv2.convertScaleAbs(cv2.magnitude(sx, sy))
        img = cv2.cvtColor(mag, cv2.COLOR_GRAY2BGR)
    
    elif name == 'prewitt':
        kx = np.array([[-1, 0, 1]] * 3)
        ky = kx.T
        px = cv2.filter2D(gray, -1, kx)
        py = cv2.filter2D(gray, -1, ky)
        mag = cv2.convertScaleAbs(np.hypot(px, py))
        img = cv2.cvtColor(mag, cv2.COLOR_GRAY2BGR)
    
    elif name == 'laplacian':
        lp = cv2.convertScaleAbs(cv2.Laplacian(gray, cv2.CV_64F))
        img = cv2.cvtColor(lp, cv2.COLOR_GRAY2BGR)
   
    elif name == 'threshold':
        _, th = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        img = cv2.cvtColor(th, cv2.COLOR_GRAY2BGR)
    
    output_image = img
    show_on_canvas_output(output_image)

# MORPHOLOCICAL OPERATIONS
def apply_morph(name):
    global output_image
    
    if output_image is None:
        return
    
    k = np.ones((5, 5), np.uint8)
    img = output_image.copy()
    
    if name == 'dilate':
        img = cv2.dilate(img, k)
    
    elif name == 'erode':
        img = cv2.erode(img, k)
    
    elif name == 'open':
        img = cv2.morphologyEx(img, cv2.MORPH_OPEN, k)
   
    elif name == 'close':
        img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, k)
    
    output_image = img
    show_on_canvas_output(output_image)

# MATHEMATICAL OPERATIONS
def compute_math(name):
    if output_image is None:
        return
    
    if name == 'mean':
        m = output_image.mean(axis=(0,1))
        messagebox.showinfo('Mean', f'BGR: {m.round(2)}')
    
    elif name == 'mode':
        gray = cv2.cvtColor(output_image, cv2.COLOR_BGR2GRAY)
        mode = Counter(gray.flatten()).most_common(1)[0][0]
        messagebox.showinfo('Mode', f'{mode}')
    
    elif name == 'median':
        med = np.median(cv2.cvtColor(output_image, cv2.COLOR_BGR2GRAY))
        messagebox.showinfo('Median', f'{med}')
    
    elif name == 'histogram':
        plt.figure()
        for i, col in enumerate(('b', 'g', 'r')):
            h = cv2.calcHist([output_image], [i], None, [256], [0, 256])
            plt.plot(h, color=col, label=f'{col.upper()} channel')
        plt.legend(); plt.grid(); plt.show()
    
    elif name == 'equalize':
        ycr = cv2.cvtColor(output_image, cv2.COLOR_BGR2YCrCb)
        ycr[:, :, 0] = cv2.equalizeHist(ycr[:, :, 0])
        output_image[:] = cv2.cvtColor(ycr, cv2.COLOR_YCrCb2BGR)
        show_on_canvas_output(output_image)

# RESTORE(DeNoise)
def restore():
    global output_image
    
    if output_image is None:
        return
    
    output_image = cv2.fastNlMeansDenoisingColored(output_image, None, 10, 10, 7, 21)
    show_on_canvas_output(output_image)

# K-MEAN
def kmeans():
    global output_image
    
    if output_image is None:
        return
    
    K = simpledialog.askinteger('Clusters', '#clusters:')
    # K = simpledialog.askinteger(10,20)
    Z = output_image.reshape((-1, 3)).astype(np.float32)
    _,lab, cent = cv2.kmeans(Z, K, None,
        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0),
        10, cv2.KMEANS_RANDOM_CENTERS)
    output_image[:] = cent[lab.flatten()].reshape(output_image.shape).astype(np.uint8)
    show_on_canvas_output(output_image)

# YOLO
def yolo_detect():
    global output_image
    
    if output_image is None:
        return
    
    res = model(output_image)
    ann = res[0].plot()
    output_image[:] = cv2.cvtColor(ann, cv2.COLOR_RGB2BGR)
    show_on_canvas_output(output_image)

# DRAWING SHAPES
shape = None
start = None

def set_shape(s): 
    globals()['shape'] = s

def on_down(e): 
    global start
    start = (e.x, e.y)

def on_move(e):
    if not start:
        return
    canvas_output.delete('preview')
    x0, y0 = start
    x1, y1 = e.x, e.y
    if shape == 'line':
        canvas_output.create_line(x0, y0, x1, y1, fill='red', tag='preview')
    elif shape == 'rect':
        canvas_output.create_rectangle(x0, y0, x1, y1, outline='green', tag='preview')
    elif shape == 'circle':
        r = int(((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5)
        canvas_output.create_oval(x0 - r, y0 - r, x0 + r, y0 + r, outline='blue', tag='preview')

def on_up(e): # retaining the updates by GPT
    if not start:
        return
    x0, y0 = start
    x1, y1 = e.x, e.y
    canvas_output.delete('preview')
    if shape == 'line':
        canvas_output.create_line(x0, y0, x1, y1, fill='red')
    elif shape == 'rect':
        canvas_output.create_rectangle(x0, y0, x1, y1, outline='green')
    elif shape == 'circle':
        r = int(((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5)
        canvas_output.create_oval(x0 - r, y0 - r, x0 + r, y0 + r, outline='blue')
    globals()['start'] = None

# EXTRA
def apply_blur():
    global output_image
    if output_image is None:
        messagebox.showerror("Error", "No output image to blur.")
        return

    ksize = simpledialog.askinteger("Kernel Size", "Enter odd kernel size", minvalue=1)
    if ksize is None:
        return
    if ksize % 2 == 0:
        messagebox.showwarning("Invalid Input", "Kernel size must be an odd number.")
        return

    output_image = cv2.GaussianBlur(output_image, (ksize, ksize), 0)
    show_on_canvas_output(output_image)


def blend_images():
    global output_image
    if output_image is None:
        return
    path = filedialog.askopenfilename(title='Select image to blend')
    if not path:
        return
    img2 = cv2.imread(path)
    img2 = cv2.resize(img2, (output_image.shape[1], output_image.shape[0]))
    alpha = simpledialog.askfloat('Alpha', 'Enter blend ratio (0.0 - 1.0):', minvalue=0.0, maxvalue=1.0)
    if alpha is None:
        return
    output_image[:] = cv2.addWeighted(output_image, alpha, img2, 1 - alpha, 0)
    show_on_canvas_output(output_image)

def resize_image():
    global output_image
    if output_image is None:
        return
    w = simpledialog.askinteger('Width', 'New width:', minvalue=1)
    h = simpledialog.askinteger('Height', 'New height:', minvalue=1)
    if w and h:
        output_image = cv2.resize(output_image, (w, h))
        show_on_canvas_output(output_image)

# GUI Setup
root = tk.Tk()
root.title('Enhanced Image Processing GUI')

menu = tk.Menu(root)
root.config(menu=menu)

# File Menu
fm = tk.Menu(menu, tearoff=0)
fm.add_command(label='Open', command=open_image)
fm.add_command(label='Reset', command=reset)
fm.add_command(label='Save Output', command=save_output)
menu.add_cascade(label='File', menu=fm)

# Filters
filt = tk.Menu(menu, tearoff=0)
for f in ['gaussian', 'canny', 'sobel', 'prewitt', 'laplacian', 'threshold']:
    filt.add_command(label=f.title(), command=lambda f=f: apply_filter(f))
menu.add_cascade(label='Filters', menu=filt)

# Morphological
morph = tk.Menu(menu, tearoff=0)
for m in ['dilate', 'erode', 'open', 'close']:
    morph.add_command(label=m.title(), command=lambda m=m: apply_morph(m))
menu.add_cascade(label='Morphology', menu=morph)

# Mathematical
mathm = tk.Menu(menu, tearoff=0)
for m, name in [('mean', 'Mean'), ('mode', 'Mode'), ('median', 'Median'), ('histogram', 'Histogram'), ('equalize', 'Equalize Hist')]:
    mathm.add_command(label=name, command=lambda m=m: compute_math(m))
menu.add_cascade(label='Mathematical', menu=mathm)

# Restore
rm = tk.Menu(menu, tearoff=0)
rm.add_command(label='Denoise (NLMeans)', command=restore)
menu.add_cascade(label='Restore', menu=rm)

# Segment (K-mean)
seg = tk.Menu(menu, tearoff=0)
seg.add_command(label='K-Means Quantize', command=kmeans)
menu.add_cascade(label='Segment', menu=seg)

# Detection
dm = tk.Menu(menu, tearoff=0)
dm.add_command(label='YOLOv8 Detect', command=yolo_detect)
menu.add_cascade(label='Detection', menu=dm)

# Shapes
shapes = tk.Menu(menu, tearoff=0)
shapes.add_command(label='Draw Line', command=lambda: set_shape('line'))
shapes.add_command(label='Draw Rectangle', command=lambda: set_shape('rect'))
shapes.add_command(label='Draw Circle', command=lambda: set_shape('circle'))
menu.add_cascade(label='Shapes', menu=shapes)

# Extra
extra_menu = tk.Menu(menu, tearoff=0)
extra_menu.add_command(label='Blur', command=apply_blur)
extra_menu.add_command(label='Blend Images', command=blend_images)
extra_menu.add_command(label='Resize Image', command=resize_image)
menu.add_cascade(label='Extra', menu=extra_menu)

# Canvas
frame = tk.Frame(root)
frame.pack(padx=20, pady=10)

# Input section
input_frame = tk.Frame(frame)
input_frame.pack(side='left', padx=10)
canvas_input = tk.Canvas(input_frame, width=frame_w, height=frame_h, bg='lightgray')
canvas_input.pack()
tk.Label(input_frame, text="Input Image").pack(pady=(5, 0))

# Output section
output_frame = tk.Frame(frame)
output_frame.pack(side='right', padx=10)
canvas_output = tk.Canvas(output_frame, width=frame_w, height=frame_h, bg='lightgray')
canvas_output.pack()
tk.Label(output_frame, text="Output Image").pack(pady=(5, 0))

# GPT
for ev in ('<ButtonPress-1>', '<B1-Motion>', '<ButtonRelease-1>'):
    canvas_output.bind(ev, {'<ButtonPress-1>': on_down, '<B1-Motion>': on_move, '<ButtonRelease-1>': on_up}[ev])

root.mainloop()
