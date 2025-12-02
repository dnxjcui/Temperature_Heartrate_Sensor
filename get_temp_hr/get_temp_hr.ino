// Get IP - Address from the ESP32... 
// 

#include <math.h>
#include <Adafruit_MLX90614.h>
#include <Wire.h>
#include "MAX30105.h"

#include "heartRate.h"

Adafruit_MLX90614 mlx = Adafruit_MLX90614();

const int ANALOG_TEMP_PIN = 4;
float voltage = 1.65;
float thermistor_resistance = 10000.0;
float R0 = 10000.0;
float B = 3950.0;
float temp = 25.0;
float objTempC = 0.0;
float objTempF = 0.0;
// float emissivity = 0.0;

// **DROP-IN 1: Variables for rolling average**
static int buffer = 100;
float tempBuffer[100];
int bufferIndex = 0;
int sampleCount = 0;
float avgTempF = 0.0;

TwoWire I2Cone = TwoWire(0);   // Bus 0: accelerometers on 4/5
TwoWire I2Ctwo = TwoWire(1);   // Bus 1: pulse ox on 6/7

// HEART RATE STUFF
MAX30105 heartRateSensor;
const byte RATE_SIZE = 4;
byte rates[RATE_SIZE];
byte rateSpot = 0;
long lastBeat = 0;
float beatsPerMinute = 0;
int beatAvg = 0;

void setup(){

  Serial.begin(115200);
  // Serial.begin(500000);
  Serial.println("Board initialized!");
  
  while (!Serial);

  I2Cone.begin(8, 9);
   if (!mlx.begin(0x5A, &I2Cone)) {  // Specify the bus
  // if (!mlx.begin()) {
    Serial.println("Error connecting to MLX sensor. Check wiring.");
    while (1);
  };
  mlx.writeEmissivity(0.98);

  I2Ctwo.begin(6, 7); // for the pulse ox, sda, scl
  // Initialize sensor
  if (!heartRateSensor.begin(I2Ctwo, I2C_SPEED_FAST, 0x57)) {
    Serial.println("MAX30102 was not found. Please check wiring/power. ");
    while (1);
  }
  
  Serial.print("Emissivity = "); Serial.println(mlx.readEmissivity());
  Serial.println("================================================");
  
  Serial.println("Place your index finger on the sensor with steady pressure.");

  // Configure sensor for heart rate detection
  byte ledBrightness = 0x1F; // Options: 0=Off to 255=50mA
  byte sampleAverage = 4;    // Options: 1, 2, 4, 8, 16, 32
  byte ledMode = 2;          // Options: 1 = Red only, 2 = Red + IR
  int sampleRate = 100;      // Options: 50, 100, 200, 400, 800, 1000, 1600, 3200
  int pulseWidth = 411;      // Options: 69, 118, 215, 411
  int adcRange = 4096;       // Options: 2048, 4096, 8192, 16384
  heartRateSensor.setup(); //Configure sensor with default settings
  heartRateSensor.setPulseAmplitudeRed(0x0A); //Turn Red LED to low to indicate sensor is running
  heartRateSensor.setPulseAmplitudeGreen(0); //Turn off Green LED
}
 
void loop(){
  objTempC = mlx.readObjectTempC();
  objTempF = mlx.readObjectTempF();

  long irValue = heartRateSensor.getIR();
  
  // Check if finger is detected
  if (irValue < 50000) {
    // No finger detected
    beatsPerMinute = 0;
    beatAvg = 0;
  } else {
    // Finger is detected, check for beat
    if (checkForBeat(irValue) == true) {
      long delta = millis() - lastBeat;
      lastBeat = millis();

      beatsPerMinute = 60 / (delta / 1000.0);

      // Filter out unrealistic heart rates
      if (beatsPerMinute < 255 && beatsPerMinute > 20) {
        rates[rateSpot++] = (byte)beatsPerMinute;
        rateSpot %= RATE_SIZE;

        // Calculate average of readings
        beatAvg = 0;
        for (byte x = 0; x < RATE_SIZE; x++) {
          beatAvg += rates[x];
        }
        beatAvg /= RATE_SIZE;
      }
    }
  }
  
    // **DROP-IN 2: Update rolling average**
  tempBuffer[bufferIndex] = objTempF;
  bufferIndex = (bufferIndex + 1) % buffer;
  
  if (sampleCount < buffer) {
    sampleCount++;
  }
  
  // Calculate average every 100 iterations
  if (bufferIndex == 0 && sampleCount == buffer) {
    float sum = 0.0;
    for (int i = 0; i < buffer; i++) {
      sum += tempBuffer[i];
    }
    avgTempF = sum / (buffer + 0.0);
  }
  Serial.print(millis());
  Serial.print(",");
  Serial.print(objTempC);
  Serial.print(",");
  Serial.print(objTempF);
  Serial.print(",");
  Serial.println(beatsPerMinute);
  delay(1);  
}

