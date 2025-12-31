import os
import sys
from datetime import datetime
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import pygame
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import soundfile as sf
import numpy as np

sys.path.append(os.path.join(os.getcwd(), "backend"))
import data_tools
import config_params
from tensorflow.keras.models import load_model

# ================== APP ==================
class DenoiseApp:
    def __init__(self, root):
        self.root = root
        self.root.title("AI Speech Enhancement System")
        self.root.geometry("1000x950")
        self.root.configure(bg="#f0f3f5", padx=20, pady=20)

        pygame.mixer.init()
        self.input_path = ""
        self.output_path = ""
        self.model = None
        self.current_model_name = ""

        # ===== TITLE =====
        tk.Label(root, text="HỆ THỐNG LỌC NHIỄU ÂM THANH AI", 
                 font=("Helvetica", 22, "bold"), bg="#f0f3f5", fg="#2c3e50").pack(pady=15)

        # ===== CONTROL FRAME =====
        control_frame = tk.Frame(root, bg="#f0f3f5")
        control_frame.pack(fill=tk.X, pady=10)

        self.btn_select = tk.Button(control_frame, text="📁 CHỌN FILE ÂM THANH (.WAV)", 
                                    command=self.select_file, font=("Arial", 14, "bold"),
                                    bg="#34495e", fg="white", height=2, cursor="hand2")
        self.btn_select.pack(fill=tk.X, padx=60, pady=6)

        self.lbl_file = tk.Label(control_frame, text="Chưa chọn tệp", bg="#f0f3f5", font=("Arial", 11))
        self.lbl_file.pack(pady=4)

        # ===== FILTER (Chọn model AI) =====
        filter_frame = tk.Frame(control_frame, bg="#f0f3f5")
        filter_frame.pack(pady=12)

        tk.Label(filter_frame, text="BỘ LỌC:", font=("Arial", 13, "bold"), bg="#f0f3f5").pack(side=tk.LEFT, padx=10)

        self.noise_options = {
            "Gia dụng": "Household_Appliance",
            "TV / Radio": "TVnRadio",
            "Xe cộ": "Vechicles",
            "Tiếng người": "Verbal_Human"
        }
        self.noise_var = tk.StringVar(value="Gia dụng")

        self.combo_noise = ttk.Combobox(filter_frame, textvariable=self.noise_var, 
                                        values=list(self.noise_options.keys()), 
                                        state="readonly", font=("Arial", 13), width=20)
        self.combo_noise.pack(side=tk.LEFT)

        # ===== PROCESS BUTTON =====
        self.btn_process = tk.Button(control_frame, text="⚡ BẮT ĐẦU LỌC NHIỄU", 
                                     command=self.process_audio, font=("Arial", 16, "bold"),
                                     bg="#e67e22", fg="white", height=2, state="disabled")
        self.btn_process.pack(fill=tk.X, padx=60, pady=18)

        # ===== PLOT =====
        plt.rcParams["figure.dpi"] = 100
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(9, 5))
        self.fig.tight_layout(pad=3.5)

        self.canvas = FigureCanvasTkAgg(self.fig, master=root)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, pady=12)
        self.update_plots(np.zeros(1000), np.zeros(1000))

        # ===== PLAYER =====
        play_frame = tk.Frame(root, bg="#f0f3f5")
        play_frame.pack(pady=15)

        btn_style = {"font": ("Arial", 13, "bold"), "width": 18, "height": 2}

        self.btn_play_old = tk.Button(play_frame, text="Nghe bản GỐC", 
                                      command=lambda: self.play_audio(self.input_path), 
                                      state="disabled", **btn_style)
        self.btn_play_old.pack(side=tk.LEFT, padx=12)

        self.btn_play_new = tk.Button(play_frame, text="Nghe bản LỌC", 
                                      command=lambda: self.play_audio(self.output_path), 
                                      state="disabled", bg="#2ecc71", fg="white", **btn_style)
        self.btn_play_new.pack(side=tk.LEFT, padx=12)

        tk.Button(root, text="Dừng nghe", command=self.stop_audio, 
                  bg="#95a5a6", fg="white", font=("Arial", 12), width=15).pack(pady=8)

    # ===== FUNCTIONS =====
    def stop_audio(self):
        pygame.mixer.music.stop()
        pygame.mixer.music.unload()

    def update_plots(self, original, cleaned):
        # Cập nhật biểu đồ sóng âm GỐC
        self.ax1.clear()
        self.ax1.set_title("Sóng âm GỐC (Noisy)", fontweight="bold")
        self.ax1.plot(original, linewidth=0.6, color='#2980b9')
        self.ax1.set_ylim(-1, 1)
        # Thêm lưới cho biểu đồ 1
        self.ax1.grid(True, linestyle='--', alpha=0.6) 

        # Cập nhật biểu đồ sóng âm SAU LỌC
        self.ax2.clear()
        self.ax2.set_title("Sóng âm SAU LỌC (Clean)", fontweight="bold")
        self.ax2.plot(cleaned, linewidth=0.6, color='#27ae60')
        self.ax2.set_ylim(-1, 1)
        # Thêm lưới cho biểu đồ 2
        self.ax2.grid(True, linestyle='--', alpha=0.6) 

        # Vẽ lại canvas
        self.canvas.draw()

    def select_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("WAV files", "*.wav")])
        if file_path:
            self.input_path = file_path
            self.lbl_file.config(text=os.path.basename(file_path))
            self.btn_process.config(state="normal")
            self.btn_play_old.config(state="normal")
            sig, _ = sf.read(file_path)
            self.update_plots(sig, np.zeros_like(sig))

    def play_audio(self, path):
        if path and os.path.exists(path):
            pygame.mixer.music.load(path)
            pygame.mixer.music.play()

    def load_selected_model(self):
        # Lấy class nhiễu từ lựa chọn giao diện
        noise_class = self.noise_options[self.noise_var.get()]
        model_name = f"DDAE_{config_params.MODEL}_{noise_class}.h5"
        model_path = os.path.join("backend", "model_files", model_name)

        if self.model is None or self.current_model_name != model_name:
            if os.path.exists(model_path):
                self.model = load_model(model_path, compile=False)
                self.current_model_name = model_name
            else:
                raise FileNotFoundError(f"Không tìm thấy model: {model_name}")

    def process_audio(self):
        try:
            self.stop_audio()
            # 1. Nạp model AI thực tế
            self.load_selected_model()

            # 2. Xử lý âm thanh đầu vào
            audio = data_tools.audio_files_to_numpy(self.input_path)
            segments = data_tools.split_into_one_second(audio, './temp_split/', 'gui', False)
            segments_array = np.array(segments)

            # 3. Chuyển sang ảnh phổ và thực hiện AI Predict
            mag_db, phase = data_tools.numpy_audio_to_matrix_spectrogram(segments_array, './temp_split/gui_images/')
            X_in = data_tools.scaled_in(mag_db)
            
            X_pred = self.model.predict(X_in)
            inv_sca_X_pred = data_tools.inv_scaled_ou(X_pred)
            X_denoise = mag_db - inv_sca_X_pred

            # 4. Tái tạo âm thanh sau lọc
            audio_reconstruct = data_tools.matrix_spectrogram_to_numpy_audio(
                X_denoise, phase, segments_array.shape[1], './temp_split/gui_images/')

            # Chuẩn hóa âm lượng đầu ra
            audio_flat = audio_reconstruct.flatten()
            peak = np.max(np.abs(audio_flat))
            audio_final = (audio_flat / peak * 0.8) if peak > 0 else audio_flat

            # Tạo tên file theo thời gian để tránh ghi đè
            now = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.output_path = f"./Predictions/Denoised_{now}.wav"
            if not os.path.exists('./Predictions/'): os.makedirs('./Predictions/')
            
            sf.write(self.output_path, audio_final, config_params.SAMPLE_RATE, 'PCM_24')

            # 5. Cập nhật đồ thị và nút nghe
            sig_original, _ = sf.read(self.input_path)
            self.update_plots(sig_original, audio_final)
            self.btn_play_new.config(state="normal")

            messagebox.showinfo("Thành công", f"Đã lọc xong!\nLưu tại: {self.output_path}")
        except Exception as e:
            messagebox.showerror("Lỗi", str(e))

# ================== MAIN ==================
if __name__ == "__main__":
    root = tk.Tk()
    root.tk.call("tk", "scaling", 2.0)

    app = DenoiseApp(root)
    root.mainloop()