import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import csv
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.linear_model import LinearRegression
import requests
import json


def handle_sign_in():
    username = username_entry.get()
    password = password_entry.get()

    valid_username = "admin"
    valid_password = "123"

    if username == valid_username and password == valid_password:
        show_dashboard()
    else:
        messagebox.showerror("Sign-In Failed", "Invalid username or password.")


def load_csv():
    file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
    if not file_path:
        return

    try:
        data = []
        with open(file_path, newline='', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            for row in reader:
                data.append(row)
    except Exception as e:
        messagebox.showerror("Error", f"Failed to load CSV file: {e}")
        return

    for row in table.get_children():
        table.delete(row)

    for row in data:
        values = [row[col] for col in table["columns"]]
        table.insert("", "end", values=values)

    attendance_rates = [float(row["Attendance_Rate"]) for row in data]
    update_area_chart(attendance_rates)

    send_requests(data)


def send_requests(data):
    url = "http://localhost:5000/predict"
    headers = {'Content-Type': 'application/json'}
    predictions = []

    for row in data:
        try:
            response = requests.post(
                url, headers=headers, data=json.dumps(row))
            response.raise_for_status()
            prediction = response.json().get("predictions", [])
            predictions.append(prediction[0])
        except requests.exceptions.RequestException as e:
            messagebox.showerror("Request Failed", f"An error occurred: {e}")
            return

    messagebox.showinfo("Predictions", f"Predictions: {predictions}")
    update_area_chart_with_predictions(predictions)


def update_area_chart(attendance_rates):
    ax_area.clear()
    x = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday"]
    ax_area.plot(x, attendance_rates, label="Actual Attendance", color="blue")
    ax_area.fill_between(x, attendance_rates, alpha=0.1, color="blue")

    days = np.arange(len(attendance_rates)).reshape(-1, 1)
    model = LinearRegression()
    model.fit(days, attendance_rates)
    predicted_rates = model.predict(days)

    ax_area.plot(x, predicted_rates,
                 label="Predicted Attendance", color="green")
    ax_area.fill_between(x, predicted_rates, alpha=0.1, color="green")

    ax_area.set_ylim(0, 1)
    ax_area.set_title("Attendance Rate")
    ax_area.legend()
    canvas_area.draw()


def update_area_chart_with_predictions(predictions):
    ax_area.clear()
    x = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday"]
    ax_area.plot(x, predictions, label="Predicted Attendance", color="green")
    ax_area.fill_between(x, predictions, alpha=0.1, color="green")

    ax_area.set_ylim(0, 1)
    ax_area.set_title("Attendance Rate")
    ax_area.legend()
    canvas_area.draw()


def show_dashboard():
    for widget in root.winfo_children():
        widget.destroy()

    root.geometry("1200x600")
    root.configure(bg="#f8f9fa")

    sidebar = tk.Frame(root, bg="#343a40", width=200, height=600)
    sidebar.place(x=0, y=0)
    menu_buttons = [
        ("Home", lambda: show_page("Home")),
        ("Profile", lambda: show_page("Profile")),
        ("Settings", lambda: show_page("Settings")),
        ("About", lambda: show_page("About")),
        ("Load CSV", load_csv),
    ]

    for i, (text, command) in enumerate(menu_buttons):
        button = tk.Button(
            sidebar,
            text=text,
            font=("Arial", 14),
            bg="#495057",
            fg="white",
            activebackground="#6c757d",
            activeforeground="white",
            relief="flat",
            command=command,
        )
        button.place(x=10, y=50 + i * 60, width=180, height=40)

    global pages
    pages = {}

    home_page = tk.Frame(root, bg="white")
    pages["Home"] = home_page

    global fig_area, ax_area, canvas_area
    fig_area = Figure(figsize=(5, 3), dpi=100)
    ax_area = fig_area.add_subplot(111)
    x = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday"]
    initial_rates = [0.5, 0.6, 0.7, 0.8, 0.9]
    ax_area.plot(x, initial_rates, label="Actual Attendance", color="blue")
    ax_area.fill_between(x, initial_rates, alpha=0.1, color="blue")
    ax_area.set_ylim(0, 1)
    ax_area.set_title("Attendance Rate")
    ax_area.legend()

    canvas_area = FigureCanvasTkAgg(fig_area, master=home_page)
    canvas_area.get_tk_widget().place(x=20, y=20, width=800, height=300)

    table_frame = tk.Frame(home_page, bg="white", relief="solid", bd=1)
    table_frame.place(x=20, y=340, width=800, height=200)

    columns = ("Day", "Weather", "Classes", "Special_Events",
               "Student_Count", "Student_Count_Last_Week", "Attendance_Rate")
    global table
    table = ttk.Treeview(table_frame, columns=columns,
                         show="headings", height=8)
    for col in columns:
        table.heading(col, text=col)
        table.column(col, width=100, anchor="center")
    table.pack(fill="both", expand=True)

    for page_name in ["Profile", "Settings", "About"]:
        page_frame = tk.Frame(root, bg="white")
        tk.Label(
            page_frame,
            text=f"Welcome to {page_name} Page",
            font=("Arial", 18, "bold"),
            bg="white",
            fg="#495057",
        ).pack(expand=True)
        pages[page_name] = page_frame

    show_page("Home")


def show_page(page_name):
    for frame in pages.values():
        frame.place_forget()  # Hide all pages
    # Show the selected page
    pages[page_name].place(x=200, y=0, width=1000, height=600)


# Sign-In Screen
root = tk.Tk()
root.title("Login")
root.geometry("800x500")
root.configure(bg="white")

canvas = tk.Canvas(root, bg="white", width=300,
                   height=200, highlightthickness=0)
canvas.place(x=30, y=50)

canvas.create_oval(50, 50, 150, 150, fill="#D9EFFF")

title_label = tk.Label(root, text="Sign in", font=(
    "Arial", 24, "bold"), bg="white", fg="blue")
title_label.place(x=350, y=50)

username_label = tk.Label(root, text="Username", font=(
    "Arial", 12), bg="white", fg="black")
username_label.place(x=300, y=120)
username_entry = tk.Entry(root, font=("Arial", 12),
                          width=25, relief="flat", bg="#F0F0F0")
username_entry.place(x=300, y=150)

password_label = tk.Label(root, text="Password", font=(
    "Arial", 12), bg="white", fg="black")
password_label.place(x=300, y=190)
password_entry = tk.Entry(root, font=("Arial", 12),
                          width=25, show="*", relief="flat", bg="#F0F0F0")
password_entry.place(x=300, y=220)

sign_in_button = tk.Button(root, text="Sign in", font=("Arial", 12, "bold"), bg="#4C9AFF", fg="white",
                           width=20, relief="flat", command=handle_sign_in)
sign_in_button.place(x=300, y=260)

signup_label = tk.Label(root, text="Don't have an account?", font=(
    "Arial", 10), bg="white", fg="black")
signup_label.place(x=300, y=310)

signup_link = tk.Label(root, text="Sign up", font=(
    "Arial", 10, "bold underline"), bg="white", fg="blue", cursor="hand2")
signup_link.place(x=430, y=310)

root.mainloop()
