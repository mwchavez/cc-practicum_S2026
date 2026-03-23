#include <WiFi.h>
#include <WiFiClientSecure.h>
#include <PubSubClient.h>

// Your AWS IoT endpoint
const char* AWS_ENDPOINT = "a1b2c3d4e5f6g7-ats.iot.us-east-1.amazonaws.com";

// Paste certificate contents here (from your downloaded files)
const char* ROOT_CA = R"(-----BEGIN CERTIFICATE-----
...your AmazonRootCA1.pem content...
-----END CERTIFICATE-----)";

const char* DEVICE_CERT = R"(-----BEGIN CERTIFICATE-----
...your device.pem.crt content...
-----END CERTIFICATE-----)";

const char* PRIVATE_KEY = R"(-----BEGIN RSA PRIVATE KEY-----
...your private.pem.key content...
-----END RSA PRIVATE KEY-----)";

WiFiClientSecure net;
PubSubClient client(net);

void connectAWS() {
  // Connect to WiFi first
  WiFi.begin("YourSSID", "YourPassword");
  while (WiFi.status() != WL_CONNECTED) delay(500);

  // Load the certificates
  net.setCACert(ROOT_CA);
  net.setCertificate(DEVICE_CERT);
  net.setPrivateKey(PRIVATE_KEY);

  // Point to your IoT Core endpoint, port 8883
  client.setServer(AWS_ENDPOINT, 8883);

  // Connect using your Thing name as the client ID
  while (!client.connect("node-01")) delay(1000);
}

void publishData() {
  String payload = "{\"node\":\"node-01\",\"moisture\":45,\"temp\":72.3}";
  client.publish("leaksensor/node-01/data", payload.c_str());
}
```

---

/*## How the Chain Works End-to-End
```
ESP32 (node-01)
    │
    │  WiFi → Internet
    │
    ▼
AWS IoT Core Endpoint (your unique URL)
    │
    │  Your IoT Rule listens on "leaksensor/node-01/data"
    │
    ▼
DynamoDB Table*/

