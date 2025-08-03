# app_gui.py

"""
Main GUI application for QuickModeler using Tkinter.
"""

import numpy as np
np.seterr(all='ignore')

import os
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk

from data_loader import load_data
from eda_report import generate_report
from modeling import train_and_evaluate
from visualization import (
    plot_regression_curve,
    plot_confusion_matrix,
    plot_roc_curve,
    plot_feature_importance
)
from prediction import save_predictions

class QuickModelerApp:
    def __init__(self, root):
        self.root = root
        root.title("QuickModeler")
        root.geometry("700x500")

        # --- Variables ---
        self.train_path = tk.StringVar()
        self.test_path = tk.StringVar()
        self.target_col = tk.StringVar()
        self.task_type = tk.StringVar(value="regression")
        self.output_dir = tk.StringVar()

        # --- Layout ---
        frm = ttk.Frame(root, padding=10)
        frm.pack(fill=tk.BOTH, expand=True)

        # Train file
        ttk.Label(frm, text="Training Data (CSV/XLSX):").grid(row=0, column=0, sticky="w")
        ttk.Entry(frm, textvariable=self.train_path, width=50).grid(row=0, column=1, padx=5)
        ttk.Button(frm, text="Browse…", command=self.browse_train).grid(row=0, column=2)

        # Test file
        ttk.Label(frm, text="Test Data (optional):").grid(row=1, column=0, sticky="w")
        ttk.Entry(frm, textvariable=self.test_path, width=50).grid(row=1, column=1, padx=5)
        ttk.Button(frm, text="Browse…", command=self.browse_test).grid(row=1, column=2)

        # Target column
        ttk.Label(frm, text="Target Column:").grid(row=2, column=0, sticky="w")
        ttk.Entry(
            frm,
            textvariable=self.target_col,
            width=50
        ).grid(
            row=2,
            column=1,
            columnspan=1,    # Yatayda sütun 1’in tamamını kaplasın
            padx=5
        )

        # Task type
        ttk.Label(frm, text="Task Type:").grid(row=4, column=0, sticky="w")
        # 1) Task Type için küçük bir Frame oluştur
        task_frame = ttk.Frame(frm)
        task_frame.grid(row=4, column=1, columnspan=1, sticky="w")

        # 2) Radiobutton’ları bu frame içinde pack ile yanyana yerleştir
        ttk.Radiobutton(
            task_frame, text="Regression", variable=self.task_type, value="regression"
        ).pack(side=tk.LEFT)
        ttk.Radiobutton(
            task_frame, text="Classification", variable=self.task_type, value="classification"
        ).pack(side=tk.LEFT, padx=(10,0))

        # Output folder
        ttk.Label(frm, text="Output Folder:").grid(row=3, column=0, sticky="w")
        ttk.Entry(frm, textvariable=self.output_dir, width=50).grid(row=3, column=1, padx=5)
        ttk.Button(frm, text="Browse…", command=self.browse_output).grid(row=3, column=2)

        # Progress bar
        self.progress = ttk.Progressbar(frm, length=400, mode="determinate")
        self.progress.grid(row=5, column=0, columnspan=3, pady=10)

        # Log area
        self.log = scrolledtext.ScrolledText(frm, height=10, state="disabled")
        self.log.grid(row=6, column=0, columnspan=3, sticky="nsew")

        # Buttons
        btn_frame = ttk.Frame(frm)
        btn_frame.grid(row=7, column=0, columnspan=3, pady=10)
        ttk.Button(btn_frame, text="Run", command=self.start).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Quit", command=root.quit).pack(side=tk.LEFT)

        # Make the log box expandable
        frm.rowconfigure(6, weight=1)
        frm.columnconfigure(1, weight=1)

        # Mevcutsa bu satırlar zaten olmalı:
        frm.columnconfigure(1, weight=1)
        #frm.columnconfigure(2, weight=0)

    def browse_train(self):
        path = filedialog.askopenfilename(filetypes=[("CSV/Excel","*.csv;*.xlsx")])
        if path: self.train_path.set(path)

    def browse_test(self):
        path = filedialog.askopenfilename(filetypes=[("CSV/Excel","*.csv;*.xlsx")])
        if path: self.test_path.set(path)

    def browse_output(self):
        path = filedialog.askdirectory()
        if path: self.output_dir.set(path)

    def log_message(self, msg):
        self.log.configure(state="normal")
        self.log.insert(tk.END, msg + "\n")
        self.log.see(tk.END)
        self.log.configure(state="disabled")

    def start(self):
        # Disable Run button to prevent re-entry
        threading.Thread(target=self.run_workflow, daemon=True).start()

    def run_workflow(self):
        try:
            self.progress["value"] = 0
            self.log_message("Loading training data...")
            train_df = load_data(self.train_path.get())

            if self.test_path.get():
                self.log_message("Loading test data...")
                test_df = load_data(self.test_path.get())
            else:
                test_df = None

            self.progress["value"] = 10
            self.log_message("Generating EDA report...")

            report_path = self.output_dir.get() +  "report.html"
            #report_path = os.path.join(self.output_dir.get(), "report.html")
            generate_report(train_df, report_path)

            self.progress["value"] = 30
            self.log_message("Training & evaluating models...")
            models, metrics = train_and_evaluate(
                df=train_df,
                target=self.target_col.get(),
                task=self.task_type.get(),
                output_dir=self.output_dir.get(),
                test_df=test_df
            )

            self.progress["value"] = 60
            self.log_message("Creating visualizations...")
            if self.task_type.get() == "regression":
                plot_regression_curve(models, train_df, self.target_col.get(), self.output_dir.get())
            else:
                
                plot_confusion_matrix(models, train_df, self.target_col.get(), self.output_dir.get())
                plot_roc_curve(models, train_df, self.target_col.get(), self.output_dir.get())

            plot_feature_importance(models, self.output_dir.get())

            self.progress["value"] = 90
            if test_df is not None:
                self.log_message("Saving predictions...")
                save_predictions(models, test_df, self.output_dir.get())

            self.progress["value"] = 100
            self.log_message("All done!")
            messagebox.showinfo("QuickModeler", "Process completed successfully.")
        except Exception as e:
            messagebox.showerror("Error", str(e))
            self.log_message("Error: " + str(e))

if __name__ == "__main__":
    root = tk.Tk()
    QuickModelerApp(root)
    root.mainloop()
