import tkinter as tk
from tkinter import ttk
from joblib import load
from PIL import Image, ImageOps
import io
import numpy as np

# Load pre-trained models (assuming you've saved them in the same directory)
logistic_model = load('logistic_model.joblib')
tree_model = load('tree_model.joblib')
kmeans_model = load('kmeans_model.joblib')

# Initially set the current model to logistic_model
current_model = logistic_model


# Function to change the model based on dropdown selection
def change_model(*args):
    global current_model  # Declare as global to modify it
    selected_model = model_var.get()
    if selected_model == "Decision Trees":
        current_model = tree_model
    elif selected_model == "Regression":
        current_model = logistic_model
    else:  # Clustering
        current_model = kmeans_model


# Function to make a prediction based on the drawing


def predict_number():
    # Capture drawing from the canvas by saving it to a file
    drawing_canvas.update()
    drawing_canvas.postscript(file="temp_canvas.ps", colormode='color')

    # Convert PostScript to GIF
    img = Image.open("temp_canvas.ps")
    img.save("temp_canvas.gif", "GIF")

    # Read the saved GIF file
    img = Image.open("temp_canvas.gif")
    img = img.resize((8, 8), Image.Resampling.LANCZOS)

    # Convert to grayscale and then to a flat array
    img = img.convert("L")
    img_array = np.array(img).reshape(1, -1)

    # Make prediction using current_model
    prediction = current_model.predict(img_array)

    # Update the label to display the predicted number
    predicted_label.config(text=f"Predicted Number: {prediction[0]}")

# Function to draw on canvas
def draw(event):
    x, y = event.x, event.y
    r = 5  # radius
    drawing_canvas.create_oval(x - r, y - r, x + r, y + r, fill="black")


# Function to clear the canvas
def clear_canvas():
    drawing_canvas.delete("all")


# Initialize Tkinter window
root = tk.Tk()
root.title("Digit Prediction App")
root.geometry("800x400")

# Drawing Canvas (Left)
drawing_canvas = tk.Canvas(root, bg="white", height=280, width=280)
drawing_canvas.pack(side=tk.LEFT, padx=10, pady=10)
drawing_canvas.bind("<B1-Motion>", draw)

# Widgets Frame (Right)
widgets_frame = ttk.Frame(root)
widgets_frame.pack(side=tk.RIGHT, padx=10, pady=10)

# Dropdown for model selection
model_var = tk.StringVar()
model_var.set("Decision Trees")  # default value
model_dropdown = ttk.OptionMenu(widgets_frame, model_var, "Decision Trees", "Regression", "Clustering")
model_dropdown.grid(row=0, column=0, padx=10, pady=10)
model_var.trace("w", change_model)

# Label for predicted number
predicted_label = ttk.Label(widgets_frame, text="Predicted Number: ")
predicted_label.grid(row=1, column=0, padx=10, pady=10)

# "Clear" button
clear_button = ttk.Button(widgets_frame, text="Clear", command=clear_canvas)
clear_button.grid(row=2, column=0, padx=10, pady=10)

# "Predict" button
predict_button = ttk.Button(widgets_frame, text="Predict", command=predict_number)
predict_button.grid(row=3, column=0, padx=10, pady=10)

# Run the Tkinter event loop
root.mainloop()

# %%
