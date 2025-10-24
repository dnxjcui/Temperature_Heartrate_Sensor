// Get IP - Address from the ESP32... 
// 

#include <math.h>
#include <Adafruit_MLX90614.h>

Adafruit_MLX90614 mlx = Adafruit_MLX90614();

const int ANALOG_TEMP_PIN = 4;
float voltage = 1.65;
float thermistor_resistance = 10000.0;
float R0 = 10000.0;
float B = 3950.0;
float temp = 25.0;
float objTempC = 0.0;
float objTempF = 0.0;

// **DROP-IN 1: Variables for rolling average**
static int buffer = 100;
float tempBuffer[100];
int bufferIndex = 0;
int sampleCount = 0;
float avgTempF = 0.0;

void setup(){
  // Serial.begin(115200);
  Serial.begin(500000);
  Serial.println("Board initialized!");
  
  while (!Serial);

  Serial.println("Adafruit MLX90614 test");

  if (!mlx.begin()) {
    Serial.println("Error connecting to MLX sensor. Check wiring.");
    while (1);
  };

  Serial.print("Emissivity = "); Serial.println(mlx.readEmissivity());
  Serial.println("================================================");
}
 
void loop(){
  // voltage = analogRead(ANALOG_TEMP_PIN);
  // // Serial.print("Voltage (raw sensor signal): ");
  // // Serial.println(voltage);

  // voltage = voltage / 4095 * 3.3;
  
  // // Serial.print("Voltage (V): ");
  // // Serial.println(voltage);
  // thermistor_resistance = R0 / ((1.0/(voltage/3.3) - 1));
  
  // // Serial.print("Resistance (Ohms): ");
  // // Serial.println(thermistor_resistance);

  // temp = 1.0 / (log(thermistor_resistance/10000.0) / B + 1.0/298.0);
  // temp = temp - 273.15;
  // Serial.print("Temperature (C): ");
  // Serial.println(temp);

  // temp = temp * 9/5 + 32;
  // Serial.print("Temperature (F): ");
  // Serial.println(temp);

 // Serial.print("Ambient = "); Serial.print(mlx.readAmbientTempC());

  objTempC = mlx.readObjectTempC();
  objTempF = mlx.readObjectTempF();
  
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
    
    // Serial.print("Average Temp (F) = ");
    // Serial.println(avgTempF);
  }
  Serial.print(millis());
  Serial.print(",");
  Serial.print(objTempC);
  Serial.print(",");
  Serial.print(objTempF);
  Serial.print(",");
  Serial.println(avgTempF);

  // Serial.println();
  delayMicroseconds(10);
  // delay(1);  
}

