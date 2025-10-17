import serial
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import time

# Configure serial port (change COM port to match your device)
PORT = 'COM6'  # Windows: 'COM3', Mac/Linux: '/dev/ttyUSB0' or '/dev/cu.usbserial-*'
BAUD_RATE = 115200

# Data storage
times = []
temps_c = []
temps_f = []
avg_temps_f = []

# Initialize plot
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
line1, = ax1.plot([], [], 'b-', label='Temp (°C)')
line2, = ax2.plot([], [], 'r-', label='Temp (°F)')
line3, = ax2.plot([], [], 'g-', linewidth=2, label='Avg Temp (°F)')

ax1.set_xlabel('Time (s)')
ax1.set_ylabel('Temperature (°C)')
ax1.set_title('Object Temperature (Celsius)')
ax1.legend()
ax1.grid(True)

ax2.set_xlabel('Time (s)')
ax2.set_ylabel('Temperature (°F)')
ax2.set_title('Object Temperature (Fahrenheit)')
ax2.legend()
ax2.grid(True)

def init():
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 50)
    ax2.set_xlim(0, 10)
    ax2.set_ylim(32, 120)
    return line1, line2, line3

def update(frame):
    try:
        if ser.in_waiting:
            line = ser.readline().decode('utf-8').strip()
            
            # Skip setup messages, only process CSV data
            if ',' in line and not line.startswith('Board'):
                parts = line.split(',')
                if len(parts) == 4:
                    timestamp = float(parts[0]) / 1000.0  # Convert ms to seconds
                    temp_c = float(parts[1])
                    temp_f = float(parts[2])
                    avg_temp_f = float(parts[3])
                    
                    # Store data
                    times.append(timestamp)
                    temps_c.append(temp_c)
                    temps_f.append(temp_f)
                    avg_temps_f.append(avg_temp_f)
                    
                    # Keep only last 100 points for display
                    if len(times) > 100:
                        times.pop(0)
                        temps_c.pop(0)
                        temps_f.pop(0)
                        avg_temps_f.pop(0)
                    
                    # Update plot data
                    if len(times) > 0:
                        # Normalize time to start at 0
                        t_normalized = [t - times[0] for t in times]
                        line1.set_data(t_normalized, temps_c)
                        line2.set_data(t_normalized, temps_f)
                        line3.set_data(t_normalized, avg_temps_f)
                        
                        # Auto-scale axes
                        ax1.set_xlim(0, max(10, t_normalized[-1]))
                        ax1.set_ylim(min(temps_c) - 5, max(temps_c) + 5)
                        ax2.set_xlim(0, max(10, t_normalized[-1]))
                        ax2.set_ylim(min(temps_f) - 5, max(max(temps_f), max(avg_temps_f)) + 5)
                    
                    print(f"Time: {timestamp:.2f}s | C: {temp_c:.2f}°C | F: {temp_f:.2f}°F | Avg: {avg_temp_f:.2f}°F")
    
    except Exception as e:
        print(f"Error: {e}")
    
    return line1, line2, line3

# Connect to serial port
try:
    ser = serial.Serial(PORT, BAUD_RATE, timeout=1)
    time.sleep(2)  # Wait for Arduino to reset
    
    print("Waiting for Arduino to be ready...")
    # Wait for READY signal
    # while True:
        # line = ser.readline().decode('utf-8').strip()
        # print(line)
        # if line == "READY":
        #     print("\nStreaming data...\n")
        #     break
    
    # Start animation
    ani = FuncAnimation(fig, update, init_func=init, blit=False, interval=50)
    plt.tight_layout()
    plt.show()
    
except serial.SerialException as e:
    print(f"Error opening serial port: {e}")
    print("Available ports:")
    import serial.tools.list_ports
    ports = serial.tools.list_ports.comports()
    for port in ports:
        print(f"  {port.device}")
finally:
    if 'ser' in locals() and ser.is_open:
        ser.close()