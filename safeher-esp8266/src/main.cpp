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
const unsigned long statusLedPulseMs = 1500;
unsigned long lastWiFiRetryMs = 0;
const unsigned long wifiRetryMs = 10000;

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

bool connectWiFi(unsigned long timeoutMs = 15000) {
  WiFi.persistent(false);
  WiFi.setAutoReconnect(true);
  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASS);
  Serial.printf("Connecting to WiFi SSID \"%s\"", WIFI_SSID);

  unsigned long startMs = millis();
  while (WiFi.status() != WL_CONNECTED && (millis() - startMs) < timeoutMs) {
    delay(300);
    Serial.print(".");
  }

  Serial.println();

  if (WiFi.status() != WL_CONNECTED) {
    Serial.println("WiFi connection failed.");
    return false;
  }

  Serial.println("WiFi connected.");
  Serial.print("SSID: ");
  Serial.println(WiFi.SSID());
  Serial.print("IP: ");
  Serial.println(WiFi.localIP());
  Serial.print("Gateway: ");
  Serial.println(WiFi.gatewayIP());
  Serial.print("RSSI: ");
  Serial.print(WiFi.RSSI());
  Serial.println(" dBm");
  return true;
}

void retryWiFiIfNeeded() {
  if (WiFi.status() == WL_CONNECTED) {
    return;
  }

  if ((millis() - lastWiFiRetryMs) < wifiRetryMs) {
    return;
  }

  lastWiFiRetryMs = millis();
  Serial.println("WiFi disconnected. Retrying connection...");
  connectWiFi(5000);
}

void sendAlert() {
  if (WiFi.status() != WL_CONNECTED && !connectWiFi(5000)) {
    Serial.println("WiFi not connected.");
    return;
  }

  WiFiClient client;
  HTTPClient http;

  if (!http.begin(client, ALERT_URL)) {
    Serial.println("HTTP begin failed.");
    return;
  }

  Serial.println("Posting SOS to backend...");
  http.addHeader("Content-Type", "application/json");

  String payload = String("{\"user_id\":\"") + USER_ID +
                   "\",\"device_id\":\"" + DEVICE_ID +
                   "\",\"kind\":\"sos\",\"message\":\"SOS button pressed\"}";

  int code = http.POST(payload);
  Serial.printf("Alert POST code: %d\n", code);
  Serial.println(http.getString());

  if (code >= 200 && code < 300) {
    Serial.println("SOS acknowledged by backend.");
    statusLedOffAt = millis() + statusLedPulseMs;
  } else {
    Serial.println("SOS delivery failed or returned an error.");
    statusLedOffAt = millis() + 500;
  }

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
  retryWiFiIfNeeded();

  bool reading = digitalRead(BUTTON_PIN);

  if (reading != lastReading) {
    Serial.printf("Button raw state: %s\n", reading == LOW ? "LOW" : "HIGH");
    lastDebounceMs = millis();
  }

  if ((millis() - lastDebounceMs) > debounceMs && reading != stableButtonState) {
    stableButtonState = reading;

    if (stableButtonState == LOW) {
      Serial.println("SOS button detected.");
      setStatusLed(true);
      sendAlert();
    }
  }

  lastReading = reading;
  updateStatusLed();
  delay(10);
}
