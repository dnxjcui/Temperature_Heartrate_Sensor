import serial
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets
import numpy as np
import time
from collections import deque
import sys

# ============== CONFIGURATION ==============
# Configure serial port (change COM port to match your device)
PORT = 'COM6'  # Windows: 'COM3', Mac/Linux: '/dev/ttyUSB0' or '/dev/cu.usbserial-*'
BAUD_RATE = 115200

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
EXPECTED_SAMPLE_RATE = 100  # samples per second (adjust based on your sensor)
BUFFER_SIZE = int(WINDOW_SIZE * EXPECTED_SAMPLE_RATE * 1.2)  # 20% extra buffer
# ==========================================

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
        
        # Setup serial connection
        self.setup_serial()
        
        # Setup timer for reading serial data
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_data)
        self.timer.start(20)  # Update every 10ms
        
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
        
    def setup_serial(self):
        try:
            self.ser = serial.Serial(PORT, BAUD_RATE, timeout=0.001)  # Very short timeout for non-blocking
            time.sleep(2)  # Wait for Arduino to reset
            self.ser.flushInput()  # Clear any buffered data
            print(f"Connected to {PORT} at {BAUD_RATE} baud")
            self.status_label.setText(f'Connected to {PORT} - Streaming data...')
        except serial.SerialException as e:
            print(f"Error opening serial port: {e}")
            self.status_label.setText(f'Error: {e}')
            self.list_available_ports()
            self.ser = None
            
    def list_available_ports(self):
        print("Available ports:")
        import serial.tools.list_ports
        ports = serial.tools.list_ports.comports()
        for port in ports:
            print(f"  {port.device}")
            
    def update_data(self):
        if not hasattr(self, 'ser') or self.ser is None or not self.ser.is_open:
            return
            
        try:
            # Read all available data
            bytes_to_read = self.ser.in_waiting
            if bytes_to_read > 0:
                # Read multiple lines if available
                data = self.ser.read(bytes_to_read).decode('utf-8', errors='ignore')
                lines = data.strip().split('\n')
                
                for line in lines:
                    # Skip setup messages, only process CSV data
                    if ',' in line and not line.startswith('Board'):
                        parts = line.strip().split(',')
                        if len(parts) == 4:
                            try:
                                timestamp = float(parts[0]) / 1000.0  # Convert ms to seconds
                                temp_c = float(parts[1])
                                temp_f = float(parts[2])
                                avg_temp_f = float(parts[3])
                                
                                # Initialize start time
                                if self.start_time is None:
                                    self.start_time = timestamp
                                
                                # Store data with relative time
                                relative_time = timestamp - self.start_time
                                self.times.append(relative_time)
                                self.temps_c.append(temp_c)
                                self.temps_f.append(temp_f)
                                self.avg_temps_f.append(avg_temp_f)
                                
                                # Update status
                                self.status_label.setText(
                                    f'Time: {relative_time:.2f}s | '
                                    f'C: {temp_c:.2f}°C | '
                                    f'F: {temp_f:.2f}°F | '
                                    f'Avg: {avg_temp_f:.2f}°F'
                                )
                                
                            except ValueError:
                                continue
                
                # Update plots if we have data
                if len(self.times) > 0:
                    self.update_plots()
                    
        except Exception as e:
            print(f"Error reading data: {e}")
            
    def update_plots(self):
        # Only keep and plot data within the window
        if len(self.times) > 0:
            # Get current time (latest timestamp)
            current_time = self.times[-1]
            cutoff_time = current_time - WINDOW_SIZE
            
            # Remove old data that's outside the window
            while len(self.times) > 0 and self.times[0] < cutoff_time:
                self.times.popleft()
                self.temps_c.popleft()
                self.temps_f.popleft()
                self.avg_temps_f.popleft()
            
            # Convert remaining data to arrays for plotting
            if len(self.times) > 0:
                t_array = np.array(self.times)
                c_array = np.array(self.temps_c)
                f_array = np.array(self.temps_f)
                avg_f_array = np.array(self.avg_temps_f)
                
                # Update curves
                self.curve_c.setData(t_array, c_array)
                self.curve_f.setData(t_array, f_array)
                self.curve_avg_f.setData(t_array, avg_f_array)
                
                # Update X-axis range to show rolling window
                self.plot_c.setXRange(max(0, current_time - WINDOW_SIZE), current_time)
                self.plot_f.setXRange(max(0, current_time - WINDOW_SIZE), current_time)
                
                # Keep Y-axis fixed to configured values
                self.plot_c.setYRange(Y_AXIS_LOW_C, Y_AXIS_HIGH_C)
                self.plot_f.setYRange(Y_AXIS_LOW_F, Y_AXIS_HIGH_F)
            
    def closeEvent(self, event):
        # Clean up serial connection on close
        if hasattr(self, 'ser') and self.ser and self.ser.is_open:
            self.ser.close()
            print("Serial port closed")
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