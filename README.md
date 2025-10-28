# Real-time Temperature Monitor

A Python-based real-time temperature monitoring system that interfaces with an Arduino/ESP32 equipped with an MLX90614 infrared temperature sensor. The application provides live temperature visualization and frequency analysis using PyQtGraph.

## Features

- **Real-time Temperature Monitoring**: Live temperature readings in both Celsius and Fahrenheit
- **Rolling Average Calculation**: Smoothed temperature data with configurable buffer size
- **Frequency Analysis**: STFT (Short-Time Fourier Transform) analysis for heart rate detection
- **Multi-threaded Architecture**: Non-blocking UI with separate worker thread for serial communication
- **Auto-scaling Plots**: Dynamic Y-axis scaling for optimal data visualization
- **High Performance**: Optimized for high sample rates (up to 10,000 samples/second)

## Hardware Requirements

- Arduino/ESP32 microcontroller
- Adafruit MLX90614 infrared temperature sensor
- USB cable for serial communication
- Breadboard and jumper wires (for connections)

## Software Requirements

- Python 3.7+
- Arduino IDE (for uploading firmware)
- Required Python packages (see Installation section)

## Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd labb
```

### 2. Install Python Dependencies

```bash
pip install -r requirements.txt
```

Or install manually:
```bash
pip install pyqtgraph numpy scipy pyserial
```

### 3. Upload Arduino Firmware

1. Open `get_temp/get_temp.ino` in Arduino IDE
2. Install the Adafruit MLX90614 library:
   - Go to Tools → Manage Libraries
   - Search for "Adafruit MLX90614" and install
3. Select your board (Arduino/ESP32) and COM port
4. Upload the firmware to your device

### 4. Configure Serial Port

Edit the `PORT` variable in `arduino.py` to match your device:

```python
# Windows
PORT = 'COM6'  # Change to your COM port

# Mac/Linux
PORT = '/dev/ttyUSB0'  # or '/dev/cu.usbserial-*'
```

## Usage

### Running the Application

```bash
python arduino.py
```

### Understanding the Interface

The application displays two main plots:

1. **Frequency Spectrum**: Shows power vs frequency analysis (0.5-20 Hz range)
   - Useful for detecting periodic patterns in temperature data
   - Can be used for heart rate analysis from temperature variations

2. **Temperature Plot**: Shows real-time temperature data
   - Red line: Current temperature in Fahrenheit
   - Green line: Rolling average temperature
   - Auto-scaling Y-axis for optimal viewing

### Configuration Options

You can modify these parameters in `arduino.py`:

```python
# Serial communication
PORT = 'COM6'           # Serial port
BAUD_RATE = 500000      # Baud rate

# Display settings
WINDOW_SIZE = 15        # Time window in seconds
Y_AXIS_LOW_F = 32       # Y-axis minimum (°F)
Y_AXIS_HIGH_F = 120     # Y-axis maximum (°F)

# Performance settings
EXPECTED_SAMPLE_RATE = 10000  # Expected samples per second
BUFFER_SIZE = int(WINDOW_SIZE * EXPECTED_SAMPLE_RATE * 1.1)
```

## Project Structure

```
labb/
├── arduino.py              # Main application file
├── utils.py                # Utility functions (serial worker, STFT)
├── get_temp/
│   └── get_temp.ino        # Arduino firmware
├── README.md               # This file
└── .gitignore              # Git ignore file
```

## Technical Details

### Serial Communication

- Uses PySerial for communication with Arduino
- Implements a worker thread to prevent UI blocking
- Handles data validation and error recovery
- Supports high baud rates (500,000 bps) for high-frequency sampling

### Data Processing

- **Rolling Average**: 100-sample buffer for temperature smoothing
- **STFT Analysis**: Short-Time Fourier Transform for frequency domain analysis
- **Data Validation**: Filters out invalid readings and handles edge cases

### Performance Optimizations

- Deque-based circular buffers for efficient memory usage
- Non-blocking UI updates with Qt timers
- Configurable buffer sizes based on expected sample rates
- Optimized PyQtGraph settings for real-time plotting

## Troubleshooting

### Common Issues

1. **Serial Port Not Found**
   - Check device manager (Windows) or `ls /dev/tty*` (Mac/Linux)
   - Ensure correct COM port in configuration
   - Try different USB cables or ports

2. **No Data Received**
   - Verify Arduino firmware is uploaded correctly
   - Check baud rate settings match between Arduino and Python
   - Ensure MLX90614 sensor is properly connected

3. **Performance Issues**
   - Reduce `EXPECTED_SAMPLE_RATE` if experiencing lag
   - Increase `WINDOW_SIZE` to reduce update frequency
   - Close other applications to free system resources

### Data Format

The Arduino sends data in CSV format:
```
timestamp_ms,temp_celsius,temp_fahrenheit,avg_temp_fahrenheit
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This project is part of an instrumentation laboratory course. Please check with your institution for usage rights.

## Acknowledgments

- Adafruit for the MLX90614 sensor library
- PyQtGraph for excellent real-time plotting capabilities
- Arduino community for hardware support
