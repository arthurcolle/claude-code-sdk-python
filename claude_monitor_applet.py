#!/usr/bin/env python3
"""
Claude State Monitor Applet
A system tray application that continuously monitors and visualizes Claude's internal state
"""

import sys
import os
import time
import threading
import json
from datetime import datetime
from pathlib import Path
import subprocess
from queue import Queue
import asyncio

# Try to import GUI libraries
try:
    from PyQt6.QtWidgets import (
        QApplication, QSystemTrayIcon, QMenu, QMainWindow, 
        QVBoxLayout, QWidget, QLabel, QPushButton, QTextEdit,
        QHBoxLayout, QSplitter, QScrollArea
    )
    from PyQt6.QtCore import QTimer, pyqtSignal, QObject, Qt, QThread
    from PyQt6.QtGui import QIcon, QPixmap, QAction, QFont, QTextCursor
    HAS_QT = True
except ImportError:
    HAS_QT = False
    print("PyQt6 not found. Installing fallback tkinter version...")

# Import our visualizer
from claude_visualizer_standalone import ClaudeVisualizer

class ClaudeStateMonitor(QObject):
    """Monitors Claude's state and emits signals when it changes"""
    state_changed = pyqtSignal(dict)
    new_message = pyqtSignal(str)
    
    def __init__(self):
        super().__init__()
        self.visualizer = ClaudeVisualizer()
        self.current_state = "idle"
        self.is_monitoring = False
        self.message_queue = Queue()
        
    def start_monitoring(self):
        """Start monitoring Claude's state"""
        self.is_monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        
    def stop_monitoring(self):
        """Stop monitoring"""
        self.is_monitoring = False
        
    def _monitor_loop(self):
        """Main monitoring loop"""
        while self.is_monitoring:
            try:
                # Get current visualizer state
                state_data = {
                    'state': self.visualizer.current_state,
                    'parameters': self.visualizer.parameters.copy(),
                    'timestamp': datetime.now().isoformat(),
                    'signature': self.visualizer.get_state_signature()
                }
                
                # Check for state changes
                if state_data['state'] != self.current_state:
                    self.current_state = state_data['state']
                    self.new_message.emit(f"State changed to: {self.current_state}")
                
                # Emit state update
                self.state_changed.emit(state_data)
                
                # Simulate natural state evolution
                self.visualizer.evolve_state()
                
                time.sleep(1)  # Update every second
                
            except Exception as e:
                self.new_message.emit(f"Monitor error: {str(e)}")
                time.sleep(5)

class VisualizationWidget(QWidget):
    """Widget to display the current visualization"""
    def __init__(self):
        super().__init__()
        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout()
        
        # Image display
        self.image_label = QLabel("Waiting for visualization...")
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setMinimumSize(400, 400)
        self.image_label.setStyleSheet("border: 1px solid #ccc; background: #000;")
        
        # State info
        self.state_label = QLabel("State: Initializing...")
        self.state_label.setFont(QFont("Monospace", 12))
        
        layout.addWidget(self.state_label)
        layout.addWidget(self.image_label)
        self.setLayout(layout)
        
    def update_visualization(self, image_path, state_data):
        """Update the displayed visualization"""
        if os.path.exists(image_path):
            pixmap = QPixmap(image_path)
            scaled = pixmap.scaled(400, 400, Qt.AspectRatioMode.KeepAspectRatio, 
                                 Qt.TransformationMode.SmoothTransformation)
            self.image_label.setPixmap(scaled)
            
        state_text = f"State: {state_data['state']} | {state_data['signature']}"
        self.state_label.setText(state_text)

class ClaudeMonitorWindow(QMainWindow):
    """Main window for the Claude monitor"""
    def __init__(self):
        super().__init__()
        self.monitor = ClaudeStateMonitor()
        self.visualizer = ClaudeVisualizer()
        self.init_ui()
        self.setup_connections()
        
    def init_ui(self):
        self.setWindowTitle("Claude State Monitor")
        self.setGeometry(100, 100, 900, 600)
        
        # Central widget
        central = QWidget()
        self.setCentralWidget(central)
        
        # Main layout
        main_layout = QHBoxLayout()
        central.setLayout(main_layout)
        
        # Left panel - visualization
        self.viz_widget = VisualizationWidget()
        
        # Right panel - controls and logs
        right_panel = QWidget()
        right_layout = QVBoxLayout()
        right_panel.setLayout(right_layout)
        
        # Control buttons
        button_layout = QHBoxLayout()
        self.start_btn = QPushButton("Start Monitoring")
        self.stop_btn = QPushButton("Stop Monitoring")
        self.stop_btn.setEnabled(False)
        self.snapshot_btn = QPushButton("Take Snapshot")
        
        button_layout.addWidget(self.start_btn)
        button_layout.addWidget(self.stop_btn)
        button_layout.addWidget(self.snapshot_btn)
        
        # Parameter display
        self.param_display = QTextEdit()
        self.param_display.setReadOnly(True)
        self.param_display.setMaximumHeight(150)
        self.param_display.setFont(QFont("Monospace", 10))
        
        # Log display
        self.log_display = QTextEdit()
        self.log_display.setReadOnly(True)
        self.log_display.setFont(QFont("Monospace", 9))
        
        # Add to right panel
        right_layout.addLayout(button_layout)
        right_layout.addWidget(QLabel("Parameters:"))
        right_layout.addWidget(self.param_display)
        right_layout.addWidget(QLabel("Activity Log:"))
        right_layout.addWidget(self.log_display)
        
        # Add panels to main layout
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self.viz_widget)
        splitter.addWidget(right_panel)
        splitter.setSizes([500, 400])
        
        main_layout.addWidget(splitter)
        
    def setup_connections(self):
        """Connect signals and slots"""
        self.start_btn.clicked.connect(self.start_monitoring)
        self.stop_btn.clicked.connect(self.stop_monitoring)
        self.snapshot_btn.clicked.connect(self.take_snapshot)
        
        self.monitor.state_changed.connect(self.update_display)
        self.monitor.new_message.connect(self.add_log_message)
        
    def start_monitoring(self):
        """Start the monitoring process"""
        self.monitor.start_monitoring()
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.add_log_message("Monitoring started")
        
    def stop_monitoring(self):
        """Stop the monitoring process"""
        self.monitor.stop_monitoring()
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.add_log_message("Monitoring stopped")
        
    def take_snapshot(self):
        """Take a snapshot of current state"""
        try:
            image_path = self.visualizer.generate_visualization()
            self.add_log_message(f"Snapshot saved: {image_path}")
            
            # Update display with snapshot
            state_data = {
                'state': self.visualizer.current_state,
                'signature': self.visualizer.get_state_signature()
            }
            self.viz_widget.update_visualization(image_path, state_data)
        except Exception as e:
            self.add_log_message(f"Snapshot error: {str(e)}")
            
    def update_display(self, state_data):
        """Update the display with new state data"""
        # Update parameters
        param_text = json.dumps(state_data['parameters'], indent=2)
        self.param_display.setPlainText(param_text)
        
        # Generate and display visualization
        try:
            image_path = self.visualizer.generate_visualization()
            self.viz_widget.update_visualization(image_path, state_data)
        except Exception as e:
            self.add_log_message(f"Visualization error: {str(e)}")
            
    def add_log_message(self, message):
        """Add a message to the log"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_display.append(f"[{timestamp}] {message}")
        # Auto-scroll to bottom
        cursor = self.log_display.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        self.log_display.setTextCursor(cursor)

class ClaudeMonitorApp(QApplication):
    """Main application class"""
    def __init__(self, argv):
        super().__init__(argv)
        self.setQuitOnLastWindowClosed(False)
        
        # Create system tray icon
        self.tray_icon = QSystemTrayIcon(self)
        self.tray_icon.setToolTip("Claude State Monitor")
        
        # Try to load an icon (create a simple one if not found)
        icon_path = Path(__file__).parent / "visualizations" / "icon.png"
        if icon_path.exists():
            self.tray_icon.setIcon(QIcon(str(icon_path)))
        else:
            # Use a default icon
            self.tray_icon.setIcon(self.style().standardIcon(
                self.style().StandardPixmap.SP_ComputerIcon))
        
        # Create tray menu
        self.tray_menu = QMenu()
        
        show_action = QAction("Show Monitor", self)
        show_action.triggered.connect(self.show_window)
        self.tray_menu.addAction(show_action)
        
        self.tray_menu.addSeparator()
        
        quit_action = QAction("Quit", self)
        quit_action.triggered.connect(self.quit)
        self.tray_menu.addAction(quit_action)
        
        self.tray_icon.setContextMenu(self.tray_menu)
        self.tray_icon.activated.connect(self.tray_activated)
        
        # Create main window
        self.window = ClaudeMonitorWindow()
        
        # Show tray icon
        self.tray_icon.show()
        
    def show_window(self):
        """Show the main window"""
        self.window.show()
        self.window.raise_()
        self.window.activateWindow()
        
    def tray_activated(self, reason):
        """Handle tray icon activation"""
        if reason == QSystemTrayIcon.ActivationReason.DoubleClick:
            self.show_window()

# Fallback Tkinter implementation
if not HAS_QT:
    import tkinter as tk
    from tkinter import ttk, scrolledtext
    from PIL import Image, ImageTk
    
    class ClaudeMonitorTk:
        """Tkinter fallback version"""
        def __init__(self):
            self.root = tk.Tk()
            self.root.title("Claude State Monitor")
            self.root.geometry("900x600")
            
            self.visualizer = ClaudeVisualizer()
            self.monitor = None
            self.is_monitoring = False
            
            self.setup_ui()
            
        def setup_ui(self):
            # Main container
            main_frame = ttk.Frame(self.root, padding="10")
            main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
            
            # Left panel - visualization
            viz_frame = ttk.LabelFrame(main_frame, text="Visualization", padding="10")
            viz_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 5))
            
            self.image_label = ttk.Label(viz_frame, text="Waiting for visualization...")
            self.image_label.pack()
            
            self.state_label = ttk.Label(viz_frame, text="State: Initializing...")
            self.state_label.pack(pady=(10, 0))
            
            # Right panel - controls
            control_frame = ttk.LabelFrame(main_frame, text="Controls", padding="10")
            control_frame.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))
            
            # Buttons
            self.start_btn = ttk.Button(control_frame, text="Start Monitoring", 
                                      command=self.start_monitoring)
            self.start_btn.pack(pady=5)
            
            self.stop_btn = ttk.Button(control_frame, text="Stop Monitoring", 
                                     command=self.stop_monitoring, state="disabled")
            self.stop_btn.pack(pady=5)
            
            self.snapshot_btn = ttk.Button(control_frame, text="Take Snapshot", 
                                         command=self.take_snapshot)
            self.snapshot_btn.pack(pady=5)
            
            # Parameters
            ttk.Label(control_frame, text="Parameters:").pack(pady=(20, 5))
            self.param_text = scrolledtext.ScrolledText(control_frame, height=8, width=40)
            self.param_text.pack()
            
            # Log
            ttk.Label(control_frame, text="Activity Log:").pack(pady=(20, 5))
            self.log_text = scrolledtext.ScrolledText(control_frame, height=10, width=40)
            self.log_text.pack()
            
            # Configure grid weights
            self.root.columnconfigure(0, weight=1)
            self.root.rowconfigure(0, weight=1)
            main_frame.columnconfigure(0, weight=2)
            main_frame.columnconfigure(1, weight=1)
            main_frame.rowconfigure(0, weight=1)
            
        def start_monitoring(self):
            self.is_monitoring = True
            self.start_btn.config(state="disabled")
            self.stop_btn.config(state="normal")
            self.add_log("Monitoring started")
            self.monitor_loop()
            
        def stop_monitoring(self):
            self.is_monitoring = False
            self.start_btn.config(state="normal")
            self.stop_btn.config(state="disabled")
            self.add_log("Monitoring stopped")
            
        def monitor_loop(self):
            if self.is_monitoring:
                try:
                    # Update visualization
                    self.visualizer.evolve_state()
                    image_path = self.visualizer.generate_visualization()
                    self.update_display(image_path)
                    
                    # Schedule next update
                    self.root.after(1000, self.monitor_loop)
                except Exception as e:
                    self.add_log(f"Error: {str(e)}")
                    
        def take_snapshot(self):
            try:
                image_path = self.visualizer.generate_visualization()
                self.update_display(image_path)
                self.add_log(f"Snapshot saved: {image_path}")
            except Exception as e:
                self.add_log(f"Snapshot error: {str(e)}")
                
        def update_display(self, image_path):
            # Update image
            if os.path.exists(image_path):
                image = Image.open(image_path)
                image = image.resize((400, 400), Image.Resampling.LANCZOS)
                photo = ImageTk.PhotoImage(image)
                self.image_label.config(image=photo, text="")
                self.image_label.image = photo  # Keep reference
                
            # Update state label
            state_text = f"State: {self.visualizer.current_state} | {self.visualizer.get_state_signature()}"
            self.state_label.config(text=state_text)
            
            # Update parameters
            params = json.dumps(self.visualizer.parameters, indent=2)
            self.param_text.delete(1.0, tk.END)
            self.param_text.insert(1.0, params)
            
        def add_log(self, message):
            timestamp = datetime.now().strftime("%H:%M:%S")
            self.log_text.insert(tk.END, f"[{timestamp}] {message}\n")
            self.log_text.see(tk.END)
            
        def run(self):
            self.root.mainloop()

def main():
    """Main entry point"""
    if HAS_QT:
        app = ClaudeMonitorApp(sys.argv)
        app.show_window()
        sys.exit(app.exec())
    else:
        # Fallback to Tkinter
        print("Using Tkinter interface (install PyQt6 for better experience)")
        app = ClaudeMonitorTk()
        app.run()

if __name__ == "__main__":
    main()