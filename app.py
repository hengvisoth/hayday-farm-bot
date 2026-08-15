import customtkinter
import mss
import cv2

from PIL import Image
from bot import Bot
from queue import Empty, Full, Queue
from threading import Thread

customtkinter.set_appearance_mode("dark")
customtkinter.set_default_color_theme("blue")

screen_dim = {
    'left': 0,
    'top': 0,
    'width': 1920,
    'height': 1080
}


class Logger(customtkinter.CTkTextbox):
    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)
        self.messages = Queue()
        self.grid(row=0, column=0, sticky="nsew")

    def log(self, *message):
        self.messages.put(" ".join(map(str, message)))

    def drain(self):
        lines = []
        while True:
            try:
                lines.append(self.messages.get_nowait())
            except Empty:
                break
        if not lines:
            return
        self.configure(state="normal")
        for line in lines:
            self.insert("end", line + "\n")
        self.see("end")
        self.configure(state="disabled")


class WorkerController:
    def __init__(self, bot, thread_factory=Thread):
        self.bot = bot
        self.thread_factory = thread_factory
        self.thread = None

    def is_running(self):
        return self.thread is not None and self.thread.is_alive()

    def start(self):
        if self.is_running():
            return False
        self.bot.reset_stop()
        self.thread = self.thread_factory(target=self.bot.bot_loop, daemon=True)
        self.thread.start()
        return True

    def stop(self):
        self.bot.request_stop()


class App(customtkinter.CTk):
    def __init__(self):
        super().__init__()
        self.sct = mss.MSS()
        self.preview_queue = Queue(maxsize=1)

        # configure window
        self.title("Hay Day Farm Bot")
        self.geometry(f"{800}x{710}")

        # configure grid layout
        self.grid_rowconfigure(1, weight=1)
        self.grid_rowconfigure((0, 2), weight=0)
        self.grid_columnconfigure(0, weight=1)

        # create toolbar
        self.console_frame = customtkinter.CTkFrame(self, height=40, corner_radius=0)
        self.console_frame.grid(row=0, column=0, sticky="nsew")
        self.console_frame.grid_columnconfigure(0, weight=1)
        self.start_button = customtkinter.CTkButton(self.console_frame, command=self.start_button_click, text="Start")
        self.start_button.grid(row=0, column=0, padx=5, pady=10, sticky="w")
        self.stop_button = customtkinter.CTkButton(self.console_frame, command=self.stop_button_click, text="Stop")
        self.stop_button.grid(row=0, column=1, padx=5, pady=10, sticky="w")
        self.stop_button.configure(state="disabled")

        # create tracking frame
        self.tracking_frame = customtkinter.CTkFrame(self, corner_radius=0)
        self.tracking_frame.grid(row=1, column=0, padx=5, pady=5, sticky="nsew")
        self.tracking_image_label = customtkinter.CTkLabel(self.tracking_frame, text="")
        self.tracking_image_label.grid(row=0, column=0, sticky="nsew")
        self.update_screen()

        # create console frame
        self.console_frame = customtkinter.CTkFrame(self, height=100, corner_radius=0)
        self.console_frame.grid(row=2, column=0, sticky="nsew")
        self.console_frame.grid_columnconfigure(0, weight=1)

        self.logger = Logger(master=self.console_frame)
        self.logger.grid(row=0, column=0, sticky="nsew")
        self.logger.log("Initialized Bot UI")

        # bot
        self.bot = Bot(self.logger, self.set_tracking_img)
        self.worker = WorkerController(self.bot)
        self.protocol("WM_DELETE_WINDOW", self.close_app)
        self.after(50, self._drain_ui_queues)

    def update_screen(self):
        data = self.sct.grab(screen_dim)
        tracking_image = customtkinter.CTkImage(Image.frombytes('RGB', data.size, data.bgra, 'raw', 'BGRX'), size=(790, 450))
        self.tracking_image_label.configure(image=tracking_image)
        self.tracking_image_label.image = tracking_image

    def set_tracking_img(self, cv2_data):
        try:
            self.preview_queue.put_nowait(cv2_data.copy())
        except Full:
            try:
                self.preview_queue.get_nowait()
            except Empty:
                pass
            self.preview_queue.put_nowait(cv2_data.copy())

    def _apply_tracking_img(self, cv2_data):
        if len(cv2_data.shape) == 3 and cv2_data.shape[2] == 4:
            data = cv2.cvtColor(cv2_data, cv2.COLOR_BGRA2RGB)
        else:
            data = cv2.cvtColor(cv2_data, cv2.COLOR_BGR2RGB)
        tracking_image = customtkinter.CTkImage(
            Image.fromarray(data),
            size=(790, 450),
        )
        self.tracking_image_label.configure(image=tracking_image)
        self.tracking_image_label.image = tracking_image

    def _drain_ui_queues(self):
        self.logger.drain()
        latest_image = None
        while True:
            try:
                latest_image = self.preview_queue.get_nowait()
            except Empty:
                break
        if latest_image is not None:
            self._apply_tracking_img(latest_image)
        if not self.worker.is_running():
            self.start_button.configure(state="normal")
            self.stop_button.configure(state="disabled")
        self.after(50, self._drain_ui_queues)

    def start_button_click(self):
        if self.start_bot():
            self.logger.log("Start")
            self.start_button.configure(state="disabled")
            self.stop_button.configure(state="normal")
        else:
            self.logger.log("Start ignored: bot is already running")

    def stop_button_click(self):
        self.logger.log("Stop")
        self.stop_button.configure(state="disabled")
        self.stop_bot()

    def start_bot(self):
        return self.worker.start()

    def stop_bot(self):
        self.worker.stop()

    def close_app(self):
        self.worker.stop()
        self.sct.close()
        self.destroy()
