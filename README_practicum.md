# Cloud Backend for Hybrid Leak Detection System
**CSEC 4390 – Senior Design Practicum | Spring 2026**  
**Faculty Advisor:** Dr. Okan Caglayan  
**Team:** Engineering & Computer Information Systems (Cybersecurity Focus)

---

## 📌 Project Overview

This repository contains the **cloud infrastructure** component of a hybrid, multi-modal water leak detection system. The full system is a joint effort between an Engineering team (responsible for physical sensors and edge AI) and a CIS/Cybersecurity team (responsible for cloud backend and data pipeline).

Our role is to build the cloud-side of the pipeline: receiving sensor data from distributed IoT nodes, storing and processing it in real time, triggering alerts when anomalies are detected, and visualizing results on a live dashboard.

> **In plain terms:** the Engineering team builds the physical sensors and puts them near pipes. Our team builds the cloud system that receives those sensor readings, figures out if a leak is happening, and sends an alert — all within 5 seconds.

---

## 🏗️ System Architecture

```
ESP32 Sensors (Edge Devices)
        │
        │  MQTT over TLS (port 8883)
        ▼
  AWS IoT Core  ──────────────────────────────────────┐
  (MQTT Broker + Rules Engine)                        │
        │                                             │
        │  IoT Rule → DynamoDB                        │ 
        ▼                                             |
   DynamoDB                                           |  
 (Raw sensor storage)                                 |
                                                      │
                                          ┌───────────┴───────────┐
                                          ▼                       ▼
                                   SNS / Email Alert       CloudWatch
                                   (< 5 sec latency)       (Monitoring
                                                            & Logs)
                                                            │
                                                            ▼
                                                    Web Dashboard
                                                  (Live Visualization)
```

### Why Each Service Was Chosen

| AWS Service | Role | Why It's Used |
|---|---|---|
| **IoT Core** | MQTT broker & device gateway | Managed service; handles device auth with certificates so we don't have to build our own broker |
| **DynamoDB** | Raw sensor data storage | NoSQL, scales automatically, fast writes for high-frequency sensor data |
| **SNS** | Alert notifications | Delivers email/SMS alerts when thresholds are exceeded |
| **CloudWatch** | Monitoring & logging | Tracks system health, Lambda errors, pipeline performance |

---

## 📡 Sensor Data Schema

Each IoT device publishes JSON payloads to AWS IoT Core on the topic `leaksense/data`:

```json
{
  "device_id": "node-001",
  "timestamp": "2026-03-01T14:23:00Z",
  "moisture": 72.4,
  "temperature": 21.3,
  "vibration": 0.12,
  "acoustic": 340.5,
  "location": "bathroom-sink"
}
```

**Field Descriptions:**
- `moisture` – relative humidity/moisture percentage near the sensor node
- `temperature` – ambient temperature in Celsius
- `vibration` – vibration intensity (g-force units)
- `acoustic` – sound frequency reading in Hz (pipe resonance detection)
- `location` – human-readable label for the sensor's physical placement

---

## 🚨 Alert Logic

The Lambda processing function evaluates each incoming event against threshold rules:

| Condition | Threshold | Action |
|---|---|---|
| High moisture | `moisture > 80%` | Trigger alert |
| Temperature spike | `temp change > 5°C in 60s` | Trigger alert |
| Acoustic anomaly | `acoustic > 500 Hz sustained` | Trigger alert |
| Combined signal | Any 2+ thresholds crossed | High-priority alert |

> **Note:** The system uses rule-based threshold detection rather than ML for this controlled environment. This keeps the pipeline reliable and debuggable for a single-semester project. ML-based detection is scoped as future work for more complex, multi-building deployments.

---

## 📁 Repository Structure

```
practicum-leak-detection/
├── iot-core/
│   ├── device-certs/          # Certificate files for ESP32 auth (not committed — see .gitignore)
│   ├── thing-policy.json      # AWS IoT Core device policy definition
│   └── iot-rules.json         # Rules Engine configuration (DynamoDB + Lambda routing)
│
├── dynamodb/
│   └── table-schema.json      # DynamoDB table definition and key structure
│
├── esp32/
│   ├── sensor_publisher.ino   # Arduino sketch for MQTT publishing from ESP32
│   └── config.h               # WiFi/broker config (not committed — see .gitignore)
│
├── dashboard/
│   └── [visualization files]  # Web dashboard source (in development)
│
├── docs/
│   └── architecture.md        # Detailed architecture decision log
│
└── README.md
```

---

## ✅ Current Progress

- [x] AWS IoT Core provisioned with device certificates
- [x] DynamoDB table configured with correct key schema
- [x] IoT Rules Engine routing MQTT → DynamoDB
- [x] End-to-end MQTT → IoT Core → DynamoDB pipeline tested
- [ ] SNS alert delivery integration
- [ ] CloudWatch monitoring and dashboards
- [ ] Web visualization dashboard
- [ ] Integration with Engineering team's physical sensor dataset

---

## 🔐 Security Practices

- IAM user with least-privilege permissions (no root account usage)
- Certificate-based mutual TLS authentication for all IoT devices
- Device policies scoped to specific MQTT topics only
- Secrets (certificates, WiFi credentials) excluded from version control via `.gitignore`

---

## 🔗 Related Resources

- [Project Wiki](../../wiki) — Full technical report, architecture decisions, and results
- [GitHub Project Board](../../projects) — Milestone tracking and task assignments
- [Foundation Project Team](../foundation-project/) — Engineering team's sensor and edge AI repository

---

*Spring 2026 — Engineering & Cybersecurity Systems | CSEC 4390*
