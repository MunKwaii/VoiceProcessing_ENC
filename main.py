import os
import sys
sys.path.append(os.path.join(os.getcwd(), "backend"))
import data_tools
import config_params
from tensorflow.keras.models import load_model
import soundfile as sf
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
import pygame

class DenoiseApp:
    def __init__(self, root):
        self.root = root
        self.root.title("AI Speech Enhancement System")
        self.root.geometry("500x400")
        
        # Khởi tạo pygame để nghe nhạc
        pygame.mixer.init()
        
        self.input_path = ""
        self.output_path = "./Predictions/GUI_Result1.wav"
        self.model = None

        # --- Giao diện ---
        tk.Label(root, text="Hệ thống lọc nhiễu AI", font=("Arial", 16, "bold")).pack(pady=10)

        self.btn_select = tk.Button(root, text="1. Chọn file âm thanh (.wav)", command=self.select_file, width=30)
        self.btn_select.pack(pady=5)

        self.lbl_file = tk.Label(root, text="Chưa chọn file", fg="blue", wraplength=400)
        self.lbl_file.pack(pady=5)

        self.btn_process = tk.Button(root, text="2. Bắt đầu lọc nhiễu", command=self.process_audio, width=30, bg="orange", state="disabled")
        self.btn_process.pack(pady=10)

        self.status_label = tk.Label(root, text="Trạng thái: Sẵn sàng", fg="green")
        self.status_label.pack(pady=5)

        # Cụm nút nghe thử
        frame_play = tk.Frame(root)
        frame_play.pack(pady=20)
        
        self.btn_play_old = tk.Button(frame_play, text="Nghe bản GỐC", command=lambda: self.play_audio(self.input_path), state="disabled")
        self.btn_play_old.pack(side=tk.LEFT, padx=10)

        self.btn_play_new = tk.Button(frame_play, text="Nghe bản LỌC", command=lambda: self.play_audio(self.output_path), state="disabled", bg="lightgreen")
        self.btn_play_new.pack(side=tk.LEFT, padx=10)

        tk.Button(root, text="Dừng nghe", command=lambda: pygame.mixer.music.stop()).pack()

    def select_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("WAV files", "*.wav")])
        if file_path:
            self.input_path = file_path
            self.lbl_file.config(text=os.path.basename(file_path))
            self.btn_process.config(state="normal")
            self.btn_play_old.config(state="normal")

    def play_audio(self, path):
        if path and os.path.exists(path):
            pygame.mixer.music.load(path)
            pygame.mixer.music.play()
        else:
            messagebox.showerror("Lỗi", "Không tìm thấy file âm thanh!")

    def process_audio(self):
        try:
            self.status_label.config(text="Đang xử lý... vui lòng đợi", fg="red")
            self.root.update()

            # Tạo thư mục tạm
            temp_split = './temp_split/gui/'
            temp_img = './temp_split/gui_images/'
            for d in [temp_split, temp_img, './Predictions/']:
                if not os.path.exists(d): os.makedirs(d)

            # Nạp model nếu chưa có
            if self.model is None:
                self.model = load_model(config_params.PATH_WEIGHTS, compile=False)

            # Quy trình xử lý (Giống file cũ của bạn)
            audio = data_tools.audio_files_to_numpy(self.input_path)
            segments = data_tools.split_into_one_second(audio, './temp_split/', 'gui', False)
            segments_array = np.array(segments)

            mag_db, phase = data_tools.numpy_audio_to_matrix_spectrogram(segments_array, temp_img)
            X_in = data_tools.scaled_in(mag_db)
            X_pred = self.model.predict(X_in)
            inv_sca_X_pred = data_tools.inv_scaled_ou(X_pred)
            X_denoise = mag_db - inv_sca_X_pred

            audio_reconstruct = data_tools.matrix_spectrogram_to_numpy_audio(
                X_denoise, phase, segments_array.shape[1], temp_img)

            # Normalization âm lượng
            audio_flat = audio_reconstruct.flatten()
            peak = np.max(np.abs(audio_flat))
            audio_final = (audio_flat / peak * 0.8) if peak > 0 else audio_flat

            sf.write(self.output_path, audio_final, config_params.SAMPLE_RATE, 'PCM_24')

            self.status_label.config(text="Đã lọc xong!", fg="green")
            self.btn_play_new.config(state="normal")
            messagebox.showinfo("Thành công", f"File đã được lưu tại:\n{self.output_path}")

        except Exception as e:
            messagebox.showerror("Lỗi hệ thống", str(e))
            self.status_label.config(text="Lỗi xử lý", fg="black")

if __name__ == "__main__":
    root = tk.Tk()
    app = DenoiseApp(root)
    root.mainloop()