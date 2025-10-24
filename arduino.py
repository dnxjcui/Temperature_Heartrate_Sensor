import serial
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets
import numpy as np
import time
from collections import deque
import sys

# Use Qt components from pyqtgraph.Qt for compatibility
from pyqtgraph.Qt.QtCore import Signal as pyqtSignal, Slot as pyqtSlot, QObject

# ============== CONFIGURATION ==============
# Configure serial port (change COM port to match your device)
PORT = 'COM6'  # Windows: 'COM3', Mac/Linux: '/dev/ttyUSB0' or '/dev/cu.usbserial-*'
BAUD_RATE = 115200
BAUD_RATE = 500000

# Time window settings
WINDOW_SIZE = 5  # Show last N seconds of data (change this to 10 for 10-second window)

# Y-axis limits for Celsius plot
Y_AXIS_LOW_C = 0    # Lower limit for Celsius plot
Y_AXIS_HIGH_C = 50  # Upper limit for Celsius plot

# Y-axis limits for Fahrenheit plot  
Y_AXIS_LOW_F = 32   # Lower limit for Fahrenheit plot
Y_AXIS_HIGH_F = 120 # Upper limit for Fahrenheit plot

# Buffer settings for performance
# Calculate buffer size based on expected data rate (adjust if needed)
# Assuming ~100 samples/second, we keep a small buffer
EXPECTED_SAMPLE_RATE = 10000  # samples per second (adjust based on your sensor)
BUFFER_SIZE = int(WINDOW_SIZE * EXPECTED_SAMPLE_RATE * 1.1)  # 10% extra buffer
# ==========================================

class SerialWorker(QObject):
    """
    Worker thread for reading serial data without blocking the UI
    """
    data_received = pyqtSignal(float, float, float, float)  # timestamp, temp_c, temp_f, avg_temp_f
    error_occurred = pyqtSignal(str)
    
    def __init__(self, port, baud_rate):
        super().__init__()
        self.port = port
        self.baud_rate = baud_rate
        self.ser = None
        self.running = False
        self.line_buffer = ""  # Buffer for incomplete lines
        
    @pyqtSlot()
    def start_reading(self):
        """Start the serial reading - called when thread starts"""
        try:
            self.ser = serial.Serial(self.port, self.baud_rate, timeout=0.001)
            time.sleep(2)  # Wait for Arduino to reset
            self.ser.flushInput()
            self.running = True
            print(f"Worker: Connected to {self.port} at {self.baud_rate} baud")
            
            # Create timer in the worker thread
            self.timer = QtCore.QTimer()
            self.timer.timeout.connect(self.read_data)
            self.timer.start(10)  # Read every 10ms
            
        except serial.SerialException as e:
            self.error_occurred.emit(f"Error opening serial port: {e}")
            return
            
    @pyqtSlot()
    def stop_reading(self):
        """Stop the serial reading thread"""
        self.running = False
        if hasattr(self, 'timer'):
            self.timer.stop()
        if self.ser and self.ser.is_open:
            self.ser.close()
            print("Worker: Serial port closed")
            
    @pyqtSlot()
    def read_data(self):
        """Read data from serial port - called by timer in worker thread"""
        if not self.running or not self.ser or not self.ser.is_open:
            return
            
        try:
            bytes_to_read = self.ser.in_waiting
            if bytes_to_read > 0:
                # Read new data and add to buffer
                new_data = self.ser.read(bytes_to_read).decode('utf-8', errors='ignore')
                self.line_buffer += new_data
                
                # Process complete lines (split on newline)
                while '\n' in self.line_buffer:
                    line, self.line_buffer = self.line_buffer.split('\n', 1)
                    line = line.strip()
                    
                    # Skip empty lines and header lines
                    if not line or line.startswith('Board') or line.startswith('Time'):
                        continue
                        
                    if ',' in line:
                        parts = line.split(',')
                        if len(parts) == 4:
                            try:
                                timestamp = float(parts[0]) / 1000.0
                                temp_c = float(parts[1])
                                temp_f = float(parts[2])
                                avg_temp_f = float(parts[3])
                                
                                # Validate data ranges - reject obviously bad data
                                # Reasonable temperature range: -20°C to 100°C
                                if not (-20 <= temp_c <= 100):
                                    print(f"Rejected bad temp_c: {temp_c}")
                                    continue
                                if not (-4 <= temp_f <= 212):
                                    print(f"Rejected bad temp_f: {temp_f}")
                                    continue
                                if not (-4 <= avg_temp_f <= 212):
                                    print(f"Rejected bad avg_temp_f: {avg_temp_f}")
                                    continue
                                    
                                # Check for NaN or Inf
                                if not (np.isfinite(temp_c) and np.isfinite(temp_f) and np.isfinite(avg_temp_f)):
                                    print(f"Rejected non-finite values")
                                    continue
                                
                                # Emit data to main thread
                                self.data_received.emit(timestamp, temp_c, temp_f, avg_temp_f)
                                
                            except (ValueError, IndexError) as e:
                                print(f"Parse error on line '{line}': {e}")
                                continue
                
                # Prevent buffer from growing too large (keep last 1000 chars max)
                if len(self.line_buffer) > 1000:
                    self.line_buffer = self.line_buffer[-1000:]
                                
        except Exception as e:
            self.error_occurred.emit(f"Error reading data: {e}")

class TemperatureMonitor(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        
        # Data storage with deques for efficiency - auto-discard old data
        self.times = deque(maxlen=BUFFER_SIZE)
        self.temps_c = deque(maxlen=BUFFER_SIZE)
        self.temps_f = deque(maxlen=BUFFER_SIZE)
        self.avg_temps_f = deque(maxlen=BUFFER_SIZE)
        self.start_time = None
        
        # Setup UI
        self.setup_ui()
        
        # Setup worker thread for serial communication
        self.setup_worker_thread()
        
        # Setup timer for plot updates (main thread only)
        self.plot_timer = QtCore.QTimer()
        self.plot_timer.timeout.connect(self.update_plots)
        self.plot_timer.start(50)  # Update plots every 50ms
        
    def setup_ui(self):
        # Create main layout
        layout = QtWidgets.QVBoxLayout()
        self.setLayout(layout)
        self.setWindowTitle('Real-time Temperature Monitor')
        
        # Create status label
        self.status_label = QtWidgets.QLabel('Connecting...')
        layout.addWidget(self.status_label)
        
        # Configure PyQtGraph settings for performance
        pg.setConfigOptions(antialias=False)  # Disable antialiasing for speed
        pg.setConfigOption('background', 'w')
        pg.setConfigOption('foreground', 'k')
        
        # Create plot widget for Celsius
        self.plot_c = pg.PlotWidget(title="Object Temperature (Celsius)")
        self.plot_c.setLabel('left', 'Temperature', units='°C')
        self.plot_c.setLabel('bottom', 'Time', units='s')
        self.plot_c.showGrid(x=True, y=True, alpha=0.3)
        self.curve_c = self.plot_c.plot(pen=pg.mkPen('b', width=2), name='Temp (°C)')
        layout.addWidget(self.plot_c)
        
        # Create plot widget for Fahrenheit
        self.plot_f = pg.PlotWidget(title="Object Temperature (Fahrenheit)")
        self.plot_f.setLabel('left', 'Temperature', units='°F')
        self.plot_f.setLabel('bottom', 'Time', units='s')
        self.plot_f.showGrid(x=True, y=True, alpha=0.3)
        self.curve_f = self.plot_f.plot(pen=pg.mkPen('r', width=2), name='Temp (°F)')
        self.curve_avg_f = self.plot_f.plot(pen=pg.mkPen('g', width=3), name='Avg Temp (°F)')
        
        # Add legend to Fahrenheit plot
        self.plot_f.addLegend()
        layout.addWidget(self.plot_f)
        
        # Set initial plot ranges
        self.plot_c.setXRange(0, WINDOW_SIZE)
        self.plot_c.setYRange(Y_AXIS_LOW_C, Y_AXIS_HIGH_C)
        self.plot_f.setXRange(0, WINDOW_SIZE)
        self.plot_f.setYRange(Y_AXIS_LOW_F, Y_AXIS_HIGH_F)
        
        # Disable auto-ranging to use fixed window and y-axis
        self.plot_c.disableAutoRange()
        self.plot_f.disableAutoRange()
        
    def setup_worker_thread(self):
        """Setup worker thread for serial communication"""
        # Create worker and thread
        self.worker = SerialWorker(PORT, BAUD_RATE)
        self.worker_thread = QtCore.QThread()
        
        # Move worker to thread
        self.worker.moveToThread(self.worker_thread)
        
        # Connect signals - these will be called in the main thread
        self.worker.data_received.connect(self.on_data_received)
        self.worker.error_occurred.connect(self.on_error_occurred)
        
        # Connect thread lifecycle signals
        self.worker_thread.started.connect(self.worker.start_reading)
        self.worker_thread.finished.connect(self.worker.stop_reading)
        
        # Start the thread
        self.worker_thread.start()
        
    @pyqtSlot(float, float, float, float)
    def on_data_received(self, timestamp, temp_c, temp_f, avg_temp_f):
        """Handle data received from worker thread (runs in main thread)"""
        # Initialize start time
        if self.start_time is None:
            self.start_time = timestamp
            self.last_print_time = time.time()
            self.data_count = 0
            self.last_timestamp = -1
        
        # Store data with relative time
        relative_time = timestamp - self.start_time
        
        # Skip duplicate or out-of-order data
        if len(self.times) > 0 and relative_time <= self.times[-1]:
            return
            
        self.times.append(relative_time)
        self.temps_c.append(temp_c)
        self.temps_f.append(temp_f)
        self.avg_temps_f.append(avg_temp_f)
        
        self.data_count += 1
        
        # Print data rate every 2 seconds for debugging
        current_print_time = time.time()
        if current_print_time - self.last_print_time >= 2.0:
            rate = self.data_count / (current_print_time - self.last_print_time)
            print(f"Data rate: {rate:.1f} samples/sec, Buffer size: {len(self.times)}")
            self.last_print_time = current_print_time
            self.data_count = 0
        
        # Update status
        self.status_label.setText(
            f'Time: {relative_time:.2f}s | '
            f'C: {temp_c:.2f}°C | '
            f'F: {temp_f:.2f}°F | '
            f'Avg: {avg_temp_f:.2f}°F'
        )
        
    @pyqtSlot(str)
    def on_error_occurred(self, error_msg):
        """Handle errors from worker thread (runs in main thread)"""
        print(f"Worker error: {error_msg}")
        self.status_label.setText(f'Error: {error_msg}')
        
    def update_plots(self):
        """Update plots - runs in main thread only"""
        if len(self.times) < 2:  # Need at least 2 points to draw a line
            return
            
        # Get current time (latest timestamp)
        current_time = self.times[-1]
        cutoff_time = current_time - WINDOW_SIZE
        
        # Remove old data that's outside the window
        while len(self.times) > 1 and self.times[0] < cutoff_time:
            self.times.popleft()
            self.temps_c.popleft()
            self.temps_f.popleft()
            self.avg_temps_f.popleft()
        
        # Convert to numpy arrays for plotting
        t_array = np.array(self.times, dtype=np.float64)
        c_array = np.array(self.temps_c, dtype=np.float64)
        f_array = np.array(self.temps_f, dtype=np.float64)
        avg_f_array = np.array(self.avg_temps_f, dtype=np.float64)
        
        # Update curves - this draws ONE continuous line through all points
        self.curve_c.setData(t_array, c_array)
        self.curve_f.setData(t_array, f_array)
        self.curve_avg_f.setData(t_array, avg_f_array)
        
        # Update X-axis range to show rolling window
        x_min = max(0, current_time - WINDOW_SIZE)
        x_max = current_time + 0.5  # Add small buffer on right
        self.plot_c.setXRange(x_min, x_max, padding=0)
        self.plot_f.setXRange(x_min, x_max, padding=0)
        
        # Keep Celsius Y-axis fixed to configured values
        self.plot_c.setYRange(Y_AXIS_LOW_C, Y_AXIS_HIGH_C, padding=0)

        # Auto-scale Fahrenheit Y-axis with padding
        if len(f_array) > 0 and len(avg_f_array) > 0:
            f_min = min(np.min(f_array), np.min(avg_f_array))
            f_max = max(np.max(f_array), np.max(avg_f_array))
            
            if np.isfinite(f_min) and np.isfinite(f_max):
                span = f_max - f_min
                if span < 3.0:
                    # If range is too small, use fixed window around midpoint
                    mid = (f_max + f_min) / 2.0
                    self.plot_f.setYRange(mid - 1.5, mid + 1.5, padding=0)
                else:
                    # Add 10% padding to the range
                    padding = span * 0.1
                    self.plot_f.setYRange(f_min - padding, f_max + padding, padding=0)
            
    def closeEvent(self, event):
        """Clean up worker thread on close"""
        print("Closing application...")
        
        # Stop the worker
        if hasattr(self, 'worker'):
            self.worker.stop_reading()
            
        # Stop and wait for thread to finish
        if hasattr(self, 'worker_thread'):
            self.worker_thread.quit()
            self.worker_thread.wait(3000)  # Wait up to 3 seconds
            print("Worker thread stopped")
            
        event.accept()

def main():
    app = QtWidgets.QApplication(sys.argv)
    
    # Set application style for better appearance
    app.setStyle('Fusion')
    
    # Create and show main window
    monitor = TemperatureMonitor()
    monitor.resize(1000, 700)
    monitor.show()
    
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()