import customtkinter as ctk
from tkinter import filedialog, messagebox
import librosa
import librosa.display
import numpy as np
import os
import json
from datetime import datetime
from fpdf import FPDF
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import warnings

warnings.filterwarnings("ignore")

# ---------------- THEME ----------------
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("dark-blue")

# ------------------ REPOSITORY ------------------
class AnalysisRepository:
    def __init__(self, filename="history.json"):
        self.filename = filename
        if not os.path.exists(self.filename):
            with open(self.filename, "w") as f:
                json.dump([], f)

    def save(self, data):
        history = self.get_all()
        history.append(data)
        with open(self.filename, "w") as f:
            json.dump(history, f, indent=4)

    def get_all(self):
        with open(self.filename, "r") as f:
            return json.load(f)

# ------------------ STRATEGY ------------------
class AnalysisStrategy:
    @staticmethod
    def thresholds(engine_type):
        return {
            "Benzinli": {"rms": 0.0003, "hf": 0.35, "impulse": 3},
            "Dizel": {"rms": 0.0005, "hf": 0.40, "impulse": 4},
        }.get(engine_type, {"rms": 0.0003, "hf": 0.35, "impulse": 3})

# ------------------ MAIN APP ------------------
class EngineEarEnterprise(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("EngineEar • Akıllı Motor Ses Analizi")
        self.geometry("1250x820")
        self.minsize(1100, 720)

        self.repo = AnalysisRepository()
        self.current_analysis = None

        self.setup_ui()

    # ---------------- UI ----------------
    def setup_ui(self):
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # ---------- SIDEBAR ----------
        self.sidebar = ctk.CTkFrame(self, width=260, corner_radius=0)
        self.sidebar.grid(row=0, column=0, sticky="nsew")
        self.sidebar.grid_rowconfigure(10, weight=1)

        ctk.CTkLabel(
            self.sidebar,
            text="ENGINE-EAR",
            font=("Inter", 22, "bold")
        ).pack(pady=(25, 5))

        ctk.CTkLabel(
            self.sidebar,
            text="Motor Sesinden Arıza Tespiti",
            font=("Inter", 12),
            text_color="#9ca3af"
        ).pack(pady=(0, 20))

        self.engine_type = ctk.StringVar(value="Benzinli")
        ctk.CTkOptionMenu(
            self.sidebar,
            values=["Benzinli", "Dizel"],
            variable=self.engine_type,
            height=36,
            font=("Inter", 13)
        ).pack(pady=10, padx=20, fill="x")

        ctk.CTkButton(
            self.sidebar,
            text="🔍 Ses Yükle & Analiz Et",
            height=42,
            font=("Inter", 14, "bold"),
            command=self.analyze
        ).pack(pady=(20, 10), padx=20, fill="x")

        ctk.CTkButton(
            self.sidebar,
            text="📄 PDF Rapor Oluştur",
            height=38,
            fg_color="#16a34a",
            hover_color="#15803d",
            font=("Inter", 13, "bold"),
            command=self.export_pdf
        ).pack(pady=(0, 20), padx=20, fill="x")

        ctk.CTkLabel(
            self.sidebar,
            text="Geçmiş Analizler",
            font=("Inter", 14, "bold")
        ).pack(pady=(10, 5))

        self.history_box = ctk.CTkTextbox(
            self.sidebar,
            width=220,
            height=300,
            font=("Inter", 12)
        )
        self.history_box.pack(padx=15, pady=(0, 15), fill="both")

        self.refresh_history()

        # ---------- CONTENT ----------
        self.content = ctk.CTkFrame(self, corner_radius=15)
        self.content.grid(row=0, column=1, sticky="nsew", padx=20, pady=20)
        self.content.grid_rowconfigure(1, weight=1)

        # Chart Card
        chart_card = ctk.CTkFrame(self.content, corner_radius=15)
        chart_card.pack(fill="both", expand=True, padx=10, pady=10)

        self.fig, self.ax = plt.subplots(figsize=(6, 4))
        self.canvas = FigureCanvasTkAgg(self.fig, master=chart_card)
        self.canvas.get_tk_widget().pack(fill="both", expand=True, padx=10, pady=10)

        # Result Card
        result_card = ctk.CTkFrame(self.content, corner_radius=15)
        result_card.pack(fill="x", padx=10, pady=(0, 10))

        ctk.CTkLabel(
            result_card,
            text="Analiz Sonuçları",
            font=("Inter", 15, "bold")
        ).pack(anchor="w", padx=15, pady=(10, 5))

        self.result_box = ctk.CTkTextbox(
            result_card,
            height=180,
            font=("JetBrains Mono", 12)
        )
        self.result_box.pack(fill="x", padx=15, pady=(0, 15))

    # ---------------- ANALYSIS ----------------
    def analyze(self):
        path = filedialog.askopenfilename()
        if not path:
            return

        y, sr = librosa.load(path, mono=True)
        y = librosa.util.normalize(y)

        rms = librosa.feature.rms(y=y)[0]
        rms_var = np.var(rms)

        S = np.abs(librosa.stft(y))
        freqs = librosa.fft_frequencies(sr=sr)
        hf_energy = np.mean(S[(freqs > 2000) & (freqs < 6000)])
        total_energy = np.mean(S)
        hf_ratio = hf_energy / total_energy

        onsets = librosa.onset.onset_detect(y=y, sr=sr)
        impulse_density = len(onsets) / (len(y) / sr)

        t = AnalysisStrategy.thresholds(self.engine_type.get())

        risk = 0
        if hf_ratio > t["hf"]:
            risk += 50
        if rms_var > t["rms"]:
            risk += 25
        if impulse_density > t["impulse"]:
            risk += 25

        if risk >= 70:
            status = "KRİTİK: VURUNTU"
            advice = "Motoru kullanmayın. Mekanik kontrol şart."
        elif risk >= 40:
            status = "UYARI: ANORMAL SES"
            advice = "Kısa süreli kullanım önerilir. Kontrol ettirin."
        else:
            status = "SAĞLIKLI"
            advice = "Motor sesi normal."

        self.current_analysis = {
            "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "file": os.path.basename(path),
            "engine": self.engine_type.get(),
            "status": status,
            "risk_score": risk,
            "rms_variance": float(rms_var),
            "hf_ratio": float(hf_ratio),
            "impulse_density": float(impulse_density),
            "advice": advice
        }

        self.repo.save(self.current_analysis)
        self.refresh_history()
        self.display_result(y, sr, status)

    # ---------------- VISUAL ----------------
    def display_result(self, y, sr, status):
        self.ax.clear()
        S = librosa.feature.melspectrogram(y=y, sr=sr)
        librosa.display.specshow(
            librosa.power_to_db(S, ref=np.max),
            ax=self.ax,
            x_axis="time",
            y_axis="mel"
        )
        self.ax.set_title(status)
        self.canvas.draw()

        self.result_box.delete("0.0", "end")
        for k, v in self.current_analysis.items():
            self.result_box.insert("end", f"{k:<18}: {v}\n")

    # ---------------- HISTORY ----------------
    def refresh_history(self):
        self.history_box.delete("0.0", "end")
        for item in reversed(self.repo.get_all()[-10:]):
            self.history_box.insert(
                "end",
                f"[{item.get('date')}] {item.get('status')} | Risk: {item.get('risk_score','-')}\n"
            )

    # ---------------- PDF ----------------
    def export_pdf(self):
        if not self.current_analysis:
            messagebox.showerror("Hata", "Analiz yok.")
            return

        pdf = FPDF()
        pdf.add_page()

        font = "/System/Library/Fonts/Supplemental/Arial.ttf"
        pdf.add_font("ArialTR", "", font, uni=True)
        pdf.set_font("ArialTR", size=14)

        pdf.cell(0, 10, "ENGINE-EAR TEKNİK ANALİZ RAPORU", new_x="LMARGIN", new_y="NEXT", align="C")
        pdf.ln(5)

        for k, v in self.current_analysis.items():
            pdf.cell(0, 8, f"{k}: {v}", new_x="LMARGIN", new_y="NEXT")

        name = f"EngineAnalysis_{datetime.now().strftime('%H%M%S')}.pdf"
        pdf.output(name)
        messagebox.showinfo("PDF", f"Oluşturuldu: {name}")

if __name__ == "__main__":
    app = EngineEarEnterprise()
    app.mainloop()