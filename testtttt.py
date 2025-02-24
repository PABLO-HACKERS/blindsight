import sys
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout,
    QScrollArea, QLabel, QLineEdit, QPushButton, QSizePolicy
)
from PyQt5.QtCore import Qt

class ChatBubble(QLabel):
    def __init__(self, text, is_sender=False):
        super().__init__(text)
        self.setWordWrap(True)
        self.setMaximumWidth(250)
        self.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Preferred)
        # Set different background colors for sender and receiver
        if is_sender:
            self.setStyleSheet("""
                QLabel {
                    background-color: #DCF8C6;
                    border-radius: 10px;
                    padding: 8px;
                }
            """)
        else:
            self.setStyleSheet("""
                QLabel {
                    background-color: #FFFFFF;
                    border: 1px solid #DDD;
                    border-radius: 10px;
                    padding: 8px;
                }
            """)

class ChatWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Chat App with Bubbles")
        self.resize(400, 500)

        # Main layout for the window
        main_layout = QVBoxLayout(self)
        main_layout.setSpacing(10)

        # Scroll area to hold chat messages
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        main_layout.addWidget(self.scroll_area)

        # Container widget inside the scroll area
        self.chat_container = QWidget()
        self.chat_layout = QVBoxLayout(self.chat_container)
        self.chat_layout.setAlignment(Qt.AlignTop)
        self.chat_layout.setSpacing(10)
        self.scroll_area.setWidget(self.chat_container)

        # Input area with QLineEdit and Send button
        input_layout = QHBoxLayout()
        self.message_input = QLineEdit()
        self.send_button = QPushButton("Send")
        input_layout.addWidget(self.message_input)
        input_layout.addWidget(self.send_button)
        main_layout.addLayout(input_layout)

        # Connect signals to functions
        self.send_button.clicked.connect(self.send_message)
        self.message_input.returnPressed.connect(self.send_message)

    def add_chat_bubble(self, text, is_sender=False):
        """Add a chat bubble to the chat container."""
        bubble = ChatBubble(text, is_sender)
        
        # Create a horizontal layout to align bubble left or right
        bubble_layout = QHBoxLayout()
        if is_sender:
            bubble_layout.addStretch()
            bubble_layout.addWidget(bubble)
        else:
            bubble_layout.addWidget(bubble)
            bubble_layout.addStretch()

        self.chat_layout.addLayout(bubble_layout)
        # Auto-scroll to the bottom
        self.scroll_area.verticalScrollBar().setValue(
            self.scroll_area.verticalScrollBar().maximum()
        )

    def send_message(self):
        """Handle sending a message."""
        message = self.message_input.text().strip()
        if message:
            # Display the sent message (right-aligned)
            self.add_chat_bubble("You: " + message, is_sender=True)
            self.message_input.clear()
            # Simulate a received response
            self.receive_message("Echo: " + message)

    def receive_message(self, message):
        """Handle receiving a message (left-aligned)."""
        self.add_chat_bubble("Friend: " + message, is_sender=False)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ChatWindow()
    window.show()
    sys.exit(app.exec_())
