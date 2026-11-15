"""
Weapon Detector - YOLOv8 photo checker
Pick one or more photos, the model draws boxes around guns / knives.
"""
import os
import sys
import subprocess
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox

import customtkinter as ctk
from PIL import Image
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "best.pt"
if not MODEL_PATH.exists() and (ROOT / "best.pt").exists():
    MODEL_PATH = ROOT / "best.pt"

OUTPUT_DIR = ROOT / "output_detections"
IMG_TYPES = [("Images", "*.jpg;*.jpeg;*.png;*.bmp;*.webp"), ("All files", "*.*")]

# Soft teal / glass-style theme
ctk.set_appearance_mode("light")
ctk.set_default_color_theme("blue")

WIN_BG = "#bdd3d3"       # window backdrop
SHELL = "#dde9e7"        # big rounded container
HEADER = "#e8f1ef"
CARD = "#e6efed"
CARD_EDGE = "#f3f9f7"
TILE = "#f1f7f6"
SIDEBAR = "#c3dbd8"
SIDE_ICON = "#3b6967"
SIDE_HOVER = "#b0cecb"
INK = "#1c292d"
MUTED = "#7b9093"
LINE = "#c3d4d3"
TEAL = "#16a39a"
TEAL_SOFT = "#d3ebe8"
TEAL_DARK = "#0f7f78"
ALERT = "#d9534f"
ALERT_SOFT = "#f6dcd9"
VIEW_BG = "#cfdfdd"

SANS = "Segoe UI"
SERIF = "Georgia"
MONO = "Consolas"


class DetectorApp(ctk.CTk):
    VIEW_W = 700
    VIEW_H = 370
    RING = 150

    def __init__(self):
        super().__init__()
        self.title("Weapon Detector")
        self.geometry("1300x820")
        self.minsize(1240, 800)
        self.configure(fg_color=WIN_BG)

        if not MODEL_PATH.exists():
            messagebox.showerror(
                "Model not found",
                f"Couldn't find the weights file at:\n{MODEL_PATH}\n\nRun train.bat first to create it.",
            )
            self.destroy()
            sys.exit(1)

        self.model = YOLO(str(MODEL_PATH))
        self.results = []  # (path, yolo_result)
        self.index = 0
        self._img_ref = None

        self._build()
        self.bind("<Left>", lambda e: self.go(-1))
        self.bind("<Right>", lambda e: self.go(1))
        self.bind("<Control-o>", lambda e: self.pick_images())
        self._draw_ring(0, TEAL, "--")

    # ------------------------------------------------------------------ UI
    def _card(self, parent, **kw):
        return ctk.CTkFrame(parent, fg_color=CARD, corner_radius=20, border_width=2, border_color=CARD_EDGE, **kw)

    def _build(self):
        # ---------------- sidebar (square, full height, flush with the window edge)
        side = ctk.CTkFrame(self, fg_color=SIDEBAR, corner_radius=0, width=104)
        side.pack(side="left", fill="y")
        side.pack_propagate(False)

        shell = ctk.CTkFrame(self, fg_color=SHELL, corner_radius=34)
        shell.pack(side="left", fill="both", expand=True, padx=(26, 34), pady=30)
        shell.grid_columnconfigure(0, weight=1)
        shell.grid_rowconfigure(0, weight=1)

        def side_btn(glyph, label, command=None, active=False, top_pad=12):
            box = ctk.CTkFrame(side, fg_color="transparent")
            box.pack(pady=(top_pad, 0), padx=8)
            ctk.CTkButton(
                box, text=glyph, width=48, height=48, corner_radius=24,
                font=ctk.CTkFont(SANS, 19, "bold"), border_width=0,
                fg_color="#f4faf8" if active else "transparent",
                hover_color="#f4faf8" if active else SIDE_HOVER,
                text_color=TEAL_DARK if active else SIDE_ICON,
                command=command,
            ).pack()
            ctk.CTkLabel(
                box, text=label, font=ctk.CTkFont(SANS, 11, "bold" if active else "normal"),
                text_color=TEAL_DARK if active else SIDE_ICON,
            ).pack(pady=(2, 0))

        side_btn("+", "Open photos", self.pick_images, top_pad=60)
        side_btn("❒", "Saved results", self.open_output_dir)
        side_btn("◀", "Previous", lambda: self.go(-1))
        side_btn("▶", "Next", lambda: self.go(1))

        # ---------------- right side of shell
        main = ctk.CTkFrame(shell, fg_color="transparent")
        main.grid(row=0, column=0, sticky="nsew")
        main.grid_columnconfigure(0, weight=1)
        main.grid_rowconfigure(1, weight=1)

        # header
        header = ctk.CTkFrame(main, fg_color="transparent", corner_radius=0, height=84)
        header.grid(row=0, column=0, sticky="ew", pady=(6, 0))
        header.grid_propagate(False)
        # round only the top-right look by overlaying: keep simple, header sits flush

        pill = ctk.CTkFrame(header, fg_color="transparent", border_width=1, border_color=LINE,
                            corner_radius=22, width=330, height=44)
        pill.pack(side="left", padx=(30, 0), pady=16)
        pill.pack_propagate(False)
        self.file_lbl = ctk.CTkLabel(pill, text="No photo loaded", font=ctk.CTkFont(SANS, 13),
                                     text_color=MUTED, anchor="w")
        self.file_lbl.pack(fill="both", expand=True, padx=18)

        self.sound_var = tk.BooleanVar(value=True)
        ctk.CTkSwitch(header, text="Alarm", variable=self.sound_var, font=ctk.CTkFont(SANS, 12),
                      text_color=INK, progress_color=TEAL, button_color="white",
                      button_hover_color="#f0f0f0", fg_color=LINE).pack(side="right", padx=30)

        # body grid: left (summary + image) / right (ring + threshold + list)
        body = ctk.CTkFrame(main, fg_color="transparent")
        body.grid(row=1, column=0, sticky="nsew", padx=22, pady=(16, 20))
        body.grid_columnconfigure(0, weight=1)
        body.grid_columnconfigure(1, weight=0, minsize=320)
        body.grid_rowconfigure(0, weight=1)

        left = ctk.CTkFrame(body, fg_color="transparent")
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 16))
        left.grid_columnconfigure(0, weight=1)
        left.grid_rowconfigure(1, weight=1)

        right = ctk.CTkFrame(body, fg_color="transparent", width=320)
        right.grid(row=0, column=1, sticky="nsew")
        right.grid_propagate(False)
        right.grid_columnconfigure(0, weight=1)
        right.grid_rowconfigure(2, weight=1)

        # ---- summary card with 4 tiles
        summary = self._card(left)
        summary.grid(row=0, column=0, sticky="ew", pady=(0, 14))
        ctk.CTkLabel(summary, text="Scan summary", font=ctk.CTkFont(SANS, 13, "bold"),
                     text_color=INK, anchor="w").pack(fill="x", padx=20, pady=(14, 6))
        ctk.CTkFrame(summary, height=1, fg_color=LINE).pack(fill="x", padx=20)
        tiles = ctk.CTkFrame(summary, fg_color="transparent")
        tiles.pack(fill="x", padx=14, pady=14)
        self.tiles = {}
        for i, (key, label) in enumerate([("frames", "Frames"), ("flagged", "Flagged"),
                                          ("clear", "Clear"), ("peak", "Peak conf.")]):
            tiles.grid_columnconfigure(i, weight=1, uniform="t")
            t = ctk.CTkFrame(tiles, fg_color=TILE, corner_radius=14, border_width=1, border_color=TILE, height=78)
            t.grid(row=0, column=i, sticky="ew", padx=6)
            t.grid_propagate(False)
            num = ctk.CTkLabel(t, text="0", font=ctk.CTkFont(SERIF, 26, "bold"), text_color=INK)
            num.pack(pady=(10, 0))
            ctk.CTkLabel(t, text=label, font=ctk.CTkFont(SANS, 11), text_color=MUTED).pack()
            self.tiles[key] = (t, num)

        # ---- image card
        img_card = self._card(left)
        img_card.grid(row=1, column=0, sticky="nsew")
        img_card.grid_columnconfigure(0, weight=1)
        top = ctk.CTkFrame(img_card, fg_color="transparent")
        top.pack(fill="x", padx=20, pady=(14, 8))
        ctk.CTkLabel(top, text="Current frame", font=ctk.CTkFont(SANS, 13, "bold"), text_color=INK).pack(side="left")
        self.counter = ctk.CTkLabel(top, text="0 / 0", font=ctk.CTkFont(MONO, 12, "bold"), text_color=MUTED)
        self.counter.pack(side="right")

        self.viewport = ctk.CTkFrame(img_card, width=self.VIEW_W, height=self.VIEW_H, corner_radius=16,
                                     fg_color=VIEW_BG)
        self.viewport.pack(padx=20, pady=(0, 18))
        self.viewport.pack_propagate(False)
        self.image_label = ctk.CTkLabel(self.viewport,
                                        text="No photo yet\nPress Ctrl+O or use the + button",
                                        font=ctk.CTkFont(SANS, 14), text_color=MUTED)
        self.image_label.pack(expand=True, fill="both")

        # ---- ring card (right, top)
        ring_card = self._card(right)
        ring_card.grid(row=0, column=0, sticky="ew")
        self.verdict_small = ctk.CTkLabel(ring_card, text="Waiting for photos", font=ctk.CTkFont(SANS, 12),
                                          text_color=MUTED, anchor="w")
        self.verdict_small.pack(fill="x", padx=20, pady=(16, 0))
        self.verdict_big = ctk.CTkLabel(ring_card, text="Ready", font=ctk.CTkFont(SERIF, 24, "bold"),
                                        text_color=INK, anchor="w")
        self.verdict_big.pack(fill="x", padx=20)
        self.canvas = tk.Canvas(ring_card, width=self.RING, height=self.RING, bg=CARD,
                                highlightthickness=0, bd=0)
        self.canvas.pack(pady=(8, 16))

        # ---- threshold card
        conf_card = self._card(right)
        conf_card.grid(row=1, column=0, sticky="ew", pady=14)
        row = ctk.CTkFrame(conf_card, fg_color="transparent")
        row.pack(fill="x", padx=20, pady=(14, 0))
        ctk.CTkLabel(row, text="Min. confidence", font=ctk.CTkFont(SANS, 12, "bold"), text_color=INK).pack(side="left")
        self.conf_lbl = ctk.CTkLabel(row, text="25%", font=ctk.CTkFont(MONO, 13, "bold"), text_color=TEAL_DARK)
        self.conf_lbl.pack(side="right")
        self.conf_slider = ctk.CTkSlider(conf_card, from_=0.05, to=0.95, number_of_steps=90,
                                         button_color=TEAL, button_hover_color=TEAL_DARK,
                                         progress_color=TEAL, fg_color=LINE, command=self._on_conf)
        self.conf_slider.set(0.25)
        self.conf_slider.pack(fill="x", padx=20, pady=(8, 16))

        # ---- detection list card
        list_card = self._card(right)
        list_card.grid(row=2, column=0, sticky="nsew")
        ctk.CTkLabel(list_card, text="Detections in this frame", font=ctk.CTkFont(SANS, 13, "bold"),
                     text_color=INK, anchor="w").pack(fill="x", padx=20, pady=(14, 6))
        ctk.CTkFrame(list_card, height=1, fg_color=LINE).pack(fill="x", padx=20)
        self.det_list = ctk.CTkScrollableFrame(list_card, fg_color="transparent", corner_radius=0,
                                               scrollbar_button_color=TEAL, scrollbar_button_hover_color=TEAL_DARK)
        self.det_list.pack(fill="both", expand=True, padx=8, pady=8)

    # -------------------------------------------------------------- helpers
    def _draw_ring(self, frac, color, center_text):
        c, s, pad = self.canvas, self.RING, 12
        c.delete("all")
        c.create_oval(pad, pad, s - pad, s - pad, outline="#c9d9d8", width=14)
        if frac > 0:
            c.create_arc(pad, pad, s - pad, s - pad, start=90, extent=-min(359.9, 360 * frac),
                         style="arc", outline=color, width=14)
        c.create_text(s / 2, s / 2 - 8, text=center_text, font=(SERIF, 24, "bold"), fill=INK)
        c.create_text(s / 2, s / 2 + 18, text="peak confidence", font=(SANS, 9), fill=MUTED)

    def _stats(self, thresh):
        flagged, peak = 0, 0.0
        for _, r in self.results:
            if r.boxes is not None and len(r.boxes) > 0:
                m = r.boxes.conf >= thresh
                if m.any():
                    flagged += 1
                    peak = max(peak, float(r.boxes.conf[m].max()))
        return len(self.results), flagged, len(self.results) - flagged, peak

    def _update_tiles(self, thresh):
        total, flagged, clear, peak = self._stats(thresh)
        vals = {"frames": str(total), "flagged": str(flagged), "clear": str(clear),
                "peak": f"{peak * 100:.0f}%" if flagged else "--"}
        for k, (tile, lbl) in self.tiles.items():
            lbl.configure(text=vals[k])
            highlight = (k == "flagged" and flagged > 0)
            tile.configure(fg_color=ALERT_SOFT if highlight else TILE,
                           border_color=ALERT if highlight else TILE)

    # -------------------------------------------------------------- actions
    def _on_conf(self, val):
        self.conf_lbl.configure(text=f"{int(float(val) * 100)}%")
        if self.results:
            self.show_current()

    def open_output_dir(self):
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        if sys.platform.startswith("win"):
            os.startfile(OUTPUT_DIR)
        elif sys.platform.startswith("darwin"):
            subprocess.run(["open", str(OUTPUT_DIR)])
        else:
            subprocess.run(["xdg-open", str(OUTPUT_DIR)])

    def pick_images(self):
        paths = filedialog.askopenfilenames(title="Choose photos to check", filetypes=IMG_TYPES)
        if not paths:
            return

        self.verdict_small.configure(text=f"Running model on {len(paths)} photo(s)")
        self.verdict_big.configure(text="Checking...", text_color=INK)
        self.update_idletasks()

        # run once at very low confidence, then filter with the slider afterwards
        self.results = []
        for p in paths:
            res = self.model.predict(p, conf=0.01, verbose=False)[0]
            self.results.append((p, res))

        self.index = 0
        thresh = float(self.conf_slider.get())
        self.show_current()

        flagged = [
            Path(p).name
            for p, r in self.results
            if r.boxes is not None and len(r.boxes) > 0 and (r.boxes.conf >= thresh).any()
        ]

        if flagged:
            if self.sound_var.get():
                self.bell()
                if sys.platform.startswith("win"):
                    try:
                        import winsound
                        winsound.Beep(1200, 500)
                    except Exception:
                        pass
            messagebox.showwarning(
                "Weapon found",
                f"Possible weapon in {len(flagged)} of {len(self.results)} photo(s):\n\n"
                + "\n".join(flagged[:10])
                + (f"\n...and {len(flagged) - 10} more." if len(flagged) > 10 else ""),
            )

    def go(self, step):
        if not self.results:
            return
        self.index = (self.index + step) % len(self.results)
        self.show_current()

    def show_current(self):
        if not self.results:
            return

        path, res = self.results[self.index]
        thresh = float(self.conf_slider.get())

        if res.boxes is not None and len(res.boxes) > 0:
            filtered = res[res.boxes.conf >= thresh]
        else:
            filtered = res

        dets = []
        if filtered.boxes is not None and len(filtered.boxes) > 0:
            for box in filtered.boxes:
                dets.append((filtered.names[int(box.cls[0])], float(box.conf[0])))

        annotated = Image.fromarray(filtered.plot()[..., ::-1])

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        out_file = OUTPUT_DIR / f"detected_{Path(path).name}"
        annotated.save(out_file)

        # fit image inside the viewport
        max_w, max_h = self.VIEW_W - 20, self.VIEW_H - 20
        w, h = annotated.size
        ratio = min(max_w / w, max_h / h)
        size = (max(1, int(w * ratio)), max(1, int(h * ratio)))
        self._img_ref = ctk.CTkImage(light_image=annotated, dark_image=annotated, size=size)
        self.image_label.configure(image=self._img_ref, text="")

        # ring + verdict
        if dets:
            best = max(c for _, c in dets)
            self._draw_ring(best, ALERT, f"{best * 100:.0f}%")
            self.verdict_small.configure(text=f"{len(dets)} object(s) above {int(thresh * 100)}%")
            self.verdict_big.configure(text="Weapon detected", text_color=ALERT)
        else:
            self._draw_ring(0, TEAL, "0%")
            self.verdict_small.configure(text=f"Nothing above {int(thresh * 100)}%")
            self.verdict_big.configure(text="Looks clear", text_color=TEAL_DARK)

        self._update_tiles(thresh)

        # detection rows (styled like list pills)
        for child in self.det_list.winfo_children():
            child.destroy()
        if dets:
            for name, conf in sorted(dets, key=lambda d: -d[1]):
                row = ctk.CTkFrame(self.det_list, fg_color=TILE, corner_radius=12, height=40)
                row.pack(fill="x", pady=4, padx=2)
                row.pack_propagate(False)
                ctk.CTkLabel(row, text=f"{conf * 100:4.0f}%", font=ctk.CTkFont(MONO, 12, "bold"),
                             text_color=ALERT, width=48).pack(side="left", padx=(12, 4))
                ctk.CTkLabel(row, text=name.capitalize(), font=ctk.CTkFont(SANS, 13),
                             text_color=INK, anchor="w").pack(side="left", fill="x", expand=True)
                ctk.CTkLabel(row, text="●", font=ctk.CTkFont(SANS, 10), text_color=ALERT).pack(side="right", padx=12)
        else:
            ctk.CTkLabel(self.det_list, text="Nothing to show.", font=ctk.CTkFont(SANS, 12),
                         text_color=MUTED).pack(pady=18)

        self.counter.configure(text=f"{self.index + 1} / {len(self.results)}")
        self.file_lbl.configure(text=Path(path).name, text_color=INK)


if __name__ == "__main__":
    DetectorApp().mainloop()