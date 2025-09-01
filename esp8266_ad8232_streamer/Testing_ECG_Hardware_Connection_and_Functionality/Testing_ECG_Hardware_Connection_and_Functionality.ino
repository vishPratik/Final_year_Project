// ESP8266_ECG_UDP_Test.ino
#include <ESP8266WiFi.h>
#include <WiFiUdp.h>

const char* ssid = "Pratik";
const char* password = "papa9833";

// Your PC's IP address (run ipconfig to find it)
const char* pcIP = "192.168.1.100"; 
const int udpPort = 5000;

WiFiUDP udp;

const int PIN_LOPLUS = D5;
const int PIN_LOMINUS = D6;
const int PIN_SDN = D7;

void setup() {
  Serial.begin(9600);
  
  pinMode(PIN_LOPLUS, INPUT);
  pinMode(PIN_LOMINUS, INPUT);
  pinMode(PIN_SDN, OUTPUT);
  digitalWrite(PIN_SDN, HIGH);
  
  WiFi.begin(ssid, password);
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  
  Serial.println("");
  Serial.println("WiFi connected");
  Serial.println("IP address: ");
  Serial.println(WiFi.localIP());
}

void loop() {
  int ecgValue = analogRead(A0);
  
  // Send via UDP
  char buffer[10];
  sprintf(buffer, "%d", ecgValue);
  udp.beginPacket(pcIP, udpPort);
  udp.write(buffer);
  udp.endPacket();
  
  delay(4); // ~250Hz sampling
}