#include <ESP8266WiFi.h>

const char* ssid = "Pratik";
const char* password = "papa9833";

const char* host = "192.168.1.100"; // Your PC's IP
const int port = 9002;

// AD8232 Pin definitions
const int PIN_LOPLUS = D5;    // GPIO14
const int PIN_LOMINUS = D6;   // GPIO12
const int PIN_SDN = D7;       // GPIO13

WiFiClient client;

void setup() {
  Serial.begin(9600);
  
  // Initialize AD8232 pins
  pinMode(PIN_LOPLUS, INPUT);
  pinMode(PIN_LOMINUS, INPUT);
  pinMode(PIN_SDN, OUTPUT);
  digitalWrite(PIN_SDN, HIGH);
  
  // Connect to WiFi
  WiFi.begin(ssid, password);
  Serial.print("Connecting to WiFi");
  
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }

  Serial.println("");
  Serial.println("WiFi connected!");
  Serial.print("IP address: ");
  Serial.println(WiFi.localIP());
  
  // Connect to server
  connectToServer();
}

void connectToServer() {
  Serial.print("Connecting to server...");
  if (client.connect(host, port)) {
    Serial.println("Connected to server!");
    // Send identification message
    client.println("# ESP8266 ECG connected");
  } else {
    Serial.println("Connection failed!");
  }
}

void loop() {
  if (!client.connected()) {
    Serial.println("Server disconnected, attempting to reconnect...");
    connectToServer();
    delay(1000);
    return;
  }
  
  int ecgValue = analogRead(A0);
  int loPlus = digitalRead(PIN_LOPLUS);
  int loMinus = digitalRead(PIN_LOMINUS);
  unsigned long timestamp = micros();
  
  // Create complete line and send at once
  String dataLine = String(timestamp) + "," + String(ecgValue) + "," + String(loPlus) + "," + String(loMinus) + "\n";
  client.print(dataLine);
  client.flush();  // Force send immediately
  
  // Debug print
  Serial.print("Sent: ");
  Serial.print(timestamp);
  Serial.print(",");
  Serial.print(ecgValue);
  Serial.print(",");
  Serial.print(loPlus);
  Serial.print(",");
  Serial.println(loMinus);
  
  delay(4);
}