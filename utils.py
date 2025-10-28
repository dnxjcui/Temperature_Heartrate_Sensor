"""
utils.py: Utility functions for the project

Contains functions for:
 - Serial communication with the Arduino
 - STFT (Short-Time Fourier Transform) for frequency analysis

"""
import serial
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets
import numpy as np
import time
from collections import deque
import sys

# Use Qt components from pyqtgraph.Qt for compatibility
from pyqtgraph.Qt.QtCore import Signal as pyqtSignal, Slot as pyqtSlot, QObject


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


def stft(data, sample_rate, window_size=1024, overlap=0.5):
    """
    Compute Short-Time Fourier Transform for frequency analysis
    
    Args:
        data: Input signal data (1D numpy array)
        sample_rate: Sampling rate in Hz
        window_size: FFT window size (default 1024)
        overlap: Overlap ratio between windows (0.0 to 1.0, default 0.5)
    
    Returns:
        frequencies: Frequency bins in Hz
        times: Time bins in seconds  
        magnitude: STFT magnitude (dB)
    """
    from scipy import signal
    import numpy as np
    
    # Ensure data is numpy array
    data = np.array(data, dtype=np.float64)
    
    # Remove any NaN or infinite values
    data = data[np.isfinite(data)]
    
    if len(data) < window_size:
        # Not enough data for STFT
        return np.array([]), np.array([]), np.array([])
    
    # Compute STFT using scipy
    frequencies, times, Zxx = signal.stft(
        data, 
        fs=sample_rate, 
        window='hann', 
        nperseg=window_size, 
        noverlap=int(window_size * overlap),
        boundary='zeros'
    )
    
    # Convert to magnitude in dB
    magnitude = 20 * np.log10(np.abs(Zxx) + 1e-10)  # Add small value to avoid log(0)
    
    return frequencies, times, magnitude


def compute_heart_rate_stft(temperature_data, sample_rate, max_freq=200):
    """
    Compute STFT specifically for heart rate analysis (0-200 Hz)
    
    Args:
        temperature_data: Temperature signal data
        sample_rate: Current sampling rate in Hz
        max_freq: Maximum frequency to analyze (default 200 Hz for heart rate)
    
    Returns:
        frequencies: Frequency bins up to max_freq
        times: Time bins
        magnitude: STFT magnitude in dB
    """
    # Use smaller window for better time resolution with heart rate
    # window_size = min(512, len(temperature_data) // 4)
    window_size = len(temperature_data)
    if window_size < 64:
        return np.array([]), np.array([]), np.array([])
    
    # Compute STFT
    frequencies, times, magnitude = stft(temperature_data, sample_rate, window_size, overlap=0.75)
    
    if len(frequencies) == 0:
        return np.array([]), np.array([]), np.array([])
    
    # Limit to frequency range of interest (0 to max_freq Hz)
    freq_mask = frequencies <= max_freq
    frequencies = frequencies[freq_mask]
    magnitude = magnitude[freq_mask, :]
    
    return frequencies, times, magnitude