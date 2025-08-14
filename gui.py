# gui.py
from PyQt5.QtWidgets import QApplication, QLabel, QWidget, QVBoxLayout, QPushButton
from PyQt5.QtGui import QPixmap, QImage
import sys, cv2

class App(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Sign2Text GUI Demo")

        self.image_label = QLabel(self)
        self.text_label = QLabel("Detected text: ")

        self.clear_button = QPushButton("Clear")
        self.clear_button.clicked.connect(self.clear_text)

        layout = QVBoxLayout()
        layout.addWidget(self.image_label)
        layout.addWidget(self.text_label)
        layout.addWidget(self.clear_button)
        self.setLayout(layout)

        self.cap = cv2.VideoCapture(0)
        self.timer = self.startTimer(30)
        self.current_text = ""

    def timerEvent(self, event):
        ret, frame = self.cap.read()
        if ret:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb.shape
            bytes_per_line = ch * w
            qt_image = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
            self.image_label.setPixmap(QPixmap.fromImage(qt_image))
            self.text_label.setText(f"Detected text: {self.current_text}")

    def clear_text(self):
        self.current_text = ""

    def closeEvent(self, event):
        self.cap.release()
        event.accept()

app = QApplication(sys.argv)
window = App()
window.show()
sys.exit(app.exec_())
