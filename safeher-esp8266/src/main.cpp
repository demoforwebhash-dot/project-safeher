#include <Arduino.h>
#include <ESP8266WiFi.h>
#include <ESP8266HTTPClient.h>

// --- WiFi ---
const char *WIFI_SSID = "HOME-ACT";
const char *WIFI_PASS = "Charankumar-ACT";

// --- Backend ---
// Use your computer's LAN IP so the ESP8266 can reach it.
// 127.0.0.1 only works on the computer itself, not from the ESP8266.
const char *ALERT_URL = "http://192.168.0.5:8000/v1/device/alert";

// --- Device Info ---
const char *USER_ID = "user-1";
const char *DEVICE_ID = "esp8266-01";

// --- Button ---
const uint8_t BUTTON_PIN = D7; // button to GND, active-low input
bool lastReading = HIGH;
bool stableButtonState = HIGH;
unsigned long lastDebounceMs = 0;
const unsigned long debounceMs = 50;

// --- Status LED ---
const uint8_t STATUS_LED_PIN = LED_BUILTIN;
const bool STATUS_LED_ON = LOW;   // built-in LED is active-low on ESP8266 NodeMCU boards
const bool STATUS_LED_OFF = HIGH;
bool statusLedOn = false;
unsigned long statusLedOffAt = 0;
const unsigned long statusLedPulseMs = 1000;

void setStatusLed(bool on) {
  digitalWrite(STATUS_LED_PIN, on ? STATUS_LED_ON : STATUS_LED_OFF);
  statusLedOn = on;
  statusLedOffAt = on ? millis() + statusLedPulseMs : 0;
}

void updateStatusLed() {
  if (statusLedOn && (long)(millis() - statusLedOffAt) >= 0) {
    setStatusLed(false);
  }
}

void connectWiFi() {
  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASS);
  Serial.print("WiFi");
  while (WiFi.status() != WL_CONNECTED) {
    delay(300);
    Serial.print(".");
  }
  Serial.println();
  Serial.print("Connected. IP: ");
  Serial.println(WiFi.localIP());
}

void sendAlert() {
  if (WiFi.status() != WL_CONNECTED) {
    Serial.println("WiFi not connected.");
    return;
  }

  WiFiClient client;
  HTTPClient http;

  if (!http.begin(client, ALERT_URL)) {
    Serial.println("HTTP begin failed.");
    return;
  }

  http.addHeader("Content-Type", "application/json");

  String payload = String("{\"user_id\":\"") + USER_ID +
                   "\",\"device_id\":\"" + DEVICE_ID +
                   "\",\"kind\":\"sos\",\"message\":\"SOS button pressed\"}";

  int code = http.POST(payload);
  Serial.printf("Alert POST code: %d\n", code);
  Serial.println(http.getString());

  http.end();
}

void setup() {
  Serial.begin(115200);
  pinMode(BUTTON_PIN, INPUT_PULLUP);
  pinMode(STATUS_LED_PIN, OUTPUT);
  setStatusLed(false);
  connectWiFi();
}

void loop() {
  bool reading = digitalRead(BUTTON_PIN);

  if (reading != lastReading) {
    lastDebounceMs = millis();
  }

  if ((millis() - lastDebounceMs) > debounceMs && reading != stableButtonState) {
    stableButtonState = reading;

    if (stableButtonState == LOW) {
      setStatusLed(true);
      sendAlert();
    }
  }

  lastReading = reading;
  updateStatusLed();
  delay(10);
}
