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


// HEART RATE STUFF
MAX30105 heartRateSensor;
const byte RATE_SIZE = 4; //Increase this for more averaging. 4 is good.
byte rates[RATE_SIZE]; //Array of heart rates
byte rateSpot = 0;
long lastBeat = 0; //Time at which the last beat occurred
float beatsPerMinute;
int beatAvg;

void setup(){
  Serial.begin(115200);
  // Serial.begin(500000);
  Serial.println("Board initialized!");
  
  while (!Serial);

  Serial.println("Adafruit MLX90614 test");

  if (!mlx.begin()) {
    Serial.println("Error connecting to MLX sensor. Check wiring.");
    while (1);
  };
  mlx.writeEmissivity(0.98);

  // Initialize sensor
  if (!heartRateSensor.begin(Wire, I2C_SPEED_FAST)) {
    Serial.println("MAX30102 was not found. Please check wiring/power. ");
    while (1);
  }
  
  Serial.print("Emissivity = "); Serial.println(mlx.readEmissivity());
  Serial.println("================================================");
  
  Serial.println("Place your index finger on the sensor with steady pressure.");

  heartRateSensor.setup(); //Configure sensor with default settings
  heartRateSensor.setPulseAmplitudeRed(0x0A); //Turn Red LED to low to indicate sensor is running
  heartRateSensor.setPulseAmplitudeGreen(0); //Turn off Green LED
}
 
void loop(){
  objTempC = mlx.readObjectTempC();
  objTempF = mlx.readObjectTempF();

  long irValue = heartRateSensor.getIR();
  
  if (checkForBeat(irValue) == true) {
    //We sensed a beat!
    long delta = millis() - lastBeat;
    lastBeat = millis();

    beatsPerMinute = 60 / (delta / 1000.0);

    if (beatsPerMinute < 255 && beatsPerMinute > 20) {
      rates[rateSpot++] = (byte)beatsPerMinute; //Store this reading in the array
      rateSpot %= RATE_SIZE; //Wrap variable

      //Take average of readings
      beatAvg = 0;
      for (byte x = 0 ; x < RATE_SIZE ; x++)
        beatAvg += rates[x];
      beatAvg /= RATE_SIZE;
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
  Serial.print(beatsPerMinute);
  delay(1);  
}

