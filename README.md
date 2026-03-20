# Enki — Hybrid Multi-Modal Leak Detection System
**ENGR 4380 - Senior Design 2 / CSEC 4390 – Practicum | Spring 2026**  
**Faculty Advisor:** Dr. Okan Caglayan & Dr. Gonzalo Parra  
**Course:** Engineering & Cybersecurity Systems

> A leak detection alert system using cross-integration of microcontrollers for data collection, cloud computing for alerts and monitoring, and machine learning for standalone edge inference on a handheld device.

---

## 👥 Team

| Name | Role | GitHub |
|---|---|---|
| Moses Chavez | Cloud Infrastructure & DevOps | [@mwchavez](https://github.com/mwchavez) |
| Andres Varela | Hardware & Sensor Integration | [@Avarela314](https://github.com/Avarela314) |
| Ethan Garcia | ML Model Development & Training | [@shamumonky](https://github.com/shamumonky) |
| Carolina Flores | Hardware Design & Documentation | [@CarolinaFUIW26](https://github.com/CarolinaFUIW26) |

---

## 🏗️ System Overview

Enki is a hybrid leak detection system with **two independent detection paths** designed to work together:

**Path 1 — Distributed Monitoring (Cloud)**  
Fixed sensor nodes attached to pipes collect environmental readings and transmit them to AWS via MQTT. The cloud pipeline stores the data, evaluates it against threshold rules, and sends email alerts when anomalies are detected.

**Path 2 — Handheld Inspection (Edge ML)**  
A portable handheld device with onboard sensors and a TensorFlow Lite model runs inference directly on the ESP32. A technician holds the device against a pipe and receives an immediate "Leak" or "No Leak" result on the LCD screen — no cloud connection required.

---

## 🔀 Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    PATH 1: DISTRIBUTED MONITORING               │
│                                                                 │
│   Fixed Sensor Nodes (ESP32-S3)                                 │
│   [SHTC3] [MPU6050] [INMP441] [HC-SR04]                        │
│           │                                                     │
│           │  MQTT over TLS (port 8883)                          │
│           ▼                                                     │
│     AWS IoT Core                                                │
│     (MQTT Broker + Rules Engine)                                │
│           │                                                     │
│           │  IoT Topic Rule                                     │
│           ▼                                                     │
│      DynamoDB                                                   │
│      (Raw sensor storage)                                       │
│           │                                                     │
│           │  CloudWatch Metric Alarms                           │
│           ▼                                                     │
│     CloudWatch ──────► SNS Email Alert                          │
│     (Threshold monitoring)                                      │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                    PATH 2: HANDHELD INSPECTION                  │
│                                                                 │
│   Handheld Device (ESP32-S3)                                    │
│   [SHTC3] [MPU6050] [INMP441] [HC-SR04]                        │
│           │                                                     │
│           │  Onboard Inference                                  │
│           ▼                                                     │
│     TensorFlow Lite Model                                       │
│     (leak_fusion_model.tflite)                                  │
│           │                                                     │
│           ▼                                                     │
│     LCD Display: "Leak" / "No Leak"                             │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## ☁️ AWS Cloud Services

| AWS Service | Role | Why It's Used |
|---|---|---|
| **IoT Core** | MQTT broker & device gateway | Managed service; handles device authentication with X.509 certificates — no custom broker needed |
| **DynamoDB** | Raw sensor data storage | NoSQL with on-demand scaling; handles high-frequency sensor writes with single-digit millisecond latency |
| **CloudWatch** | Threshold alerting & monitoring | Native AWS monitoring; configurable metric alarms eliminate the need for custom processing code |
| **SNS** | Alert notifications | Delivers email alerts when CloudWatch alarm thresholds are exceeded |
| **IAM** | Access control | Least-privilege roles scoped per service; IoT Rules Engine assumes a role to write to DynamoDB |
| **AWS CDK** | Infrastructure as Code | Entire cloud stack defined in Python; parameterized and portable across AWS accounts |

---

## 🔧 Edge & ML Stack

| Component | Role | Details |
|---|---|---|
| **ESP32-S3** | Microcontroller | Drives both fixed nodes and handheld device |
| **SHTC3** | Temperature & humidity sensor | Detects environmental changes near pipes |
| **MPU6050** | Vibration/motion sensor | Detects pipe vibration anomalies |
| **INMP441** | I2S digital microphone | Acoustic sensing for pipe resonance detection |
| **HC-SR04** | Ultrasonic sensor | Proximity-based moisture/leak detection |
| **TensorFlow Lite** | Edge ML framework | Runs leak classification model on the handheld ESP32 |

---

## 📡 Sensor Data Schema

Each fixed node publishes JSON payloads via MQTT to the topic pattern `leaksensor/node-XX/data`:

```json
{
  "device_id": "node-01",
  "timestamp": 1709312580,
  "moisture": 72.4,
  "temperature": 21.3,
  "vibration": 0.12,
  "acoustic": 340.5
}
```

| Field | Type | Description |
|---|---|---|
| `device_id` | String | Unique identifier for the sensor node (partition key) |
| `timestamp` | Number | Unix epoch in seconds (sort key) |
| `moisture` | Number | Relative humidity/moisture percentage near the node |
| `temperature` | Number | Ambient temperature in Celsius |
| `vibration` | Number | Vibration intensity (g-force units) |
| `acoustic` | Number | Sound frequency in Hz (pipe resonance detection) |

---

## 🚨 Alert Logic

CloudWatch Metric Alarms evaluate incoming sensor data against threshold rules and trigger SNS notifications:

| Condition | Threshold | Action |
|---|---|---|
| High moisture | `moisture > 80%` | Trigger alert |
| Temperature spike | `temp change > 5°C in 60s` | Trigger alert |
| Acoustic anomaly | `acoustic > 500 Hz sustained` | Trigger alert |
| Combined signal | Any 2+ thresholds crossed | High-priority alert |

> **Design Decision:** The cloud pipeline uses rule-based threshold detection rather than ML. In a controlled environment with known baselines, CloudWatch alarms provide reliable, debuggable alerting. ML-based cloud detection is scoped as future work for larger-scale, multi-building deployments.

---

## 📁 Repository Structure

```
enki-leak-detection/
├── cloud_infrastructure/        # AWS CDK stack (Python)
│   ├── leak_detection_practicum_stack.py
│   ├── docs/
│        ├── Architecture.md 
├── ml_training/                 # ML model training & dataset generation
│   ├── train_fusion_model.py
│   ├── generate_fake_dataset.py
│   ├── leak_fusion_model.tflite
│   ├── norm_mu.csv
│   └── norm_sigma.csv
├── pipe_nodes/                  # (Planned) ESP32 firmware for fixed sensor nodes
├── handheld_device/             # (Planned) ESP32 firmware + TFLite for handheld
└── README.md
```

---

## ✅ Progress

### Cloud Infrastructure (Moses)
- [x] AWS account setup with IAM least-privilege
- [x] AWS IoT Core provisioned with X.509 device certificates
- [x] DynamoDB table configured (partition key: `device_id`, sort key: `timestamp`)
- [x] IoT Rules Engine routing MQTT → DynamoDB
- [x] IAM Role for IoT → DynamoDB write permissions
- [x] End-to-end MQTT → IoT Core → DynamoDB pipeline tested
- [x] SNS Topic with email subscription
- [ ] Infrastructure defined as code with AWS CDK (Python)
- [ ] CloudWatch metric alarms for threshold detection
- [ ] CloudWatch dashboard for monitoring

### Hardware & Sensors (Andres, Carolina)
- [x] Sensor data sheets compiled
- [x] Bill of materials finalized
- [x] ESP32 WiFi connection and MQTT data transfer tested
- [ ] Fixed pipe node prototype assembly
- [ ] Handheld device CAD design (in progress)
- [ ] Handheld device assembly

### ML Model (Ethan)
- [x] Fake dataset generator for training
- [x] TFLite model trained and exported
- [x] Normalization parameters saved (norm_mu.csv, norm_sigma.csv)
- [ ] Embedded AI deployment on ESP32-S3
- [ ] Handheld integration and field testing

---

## 🔐 Security Practices

- **No root account usage** — IAM user with scoped permissions only
- **Certificate-based mutual TLS** — X.509 certificates authenticate every IoT device
- **Least-privilege IoT policies** — device policies scoped to specific MQTT topics (`leaksensor/node-*/data`)
- **IAM Roles over IAM Users** — services assume temporary roles; IoT Rules Engine has only `dynamodb:PutItem` permission on the sensor table
- **Secrets excluded from version control** — certificates, WiFi credentials, and ARNs kept out of the repo via `.gitignore`
- **CDK parameterization** — certificate ARNs passed as `CfnParameter` at deploy time, not hardcoded

---

## 🔗 Resources

- [Project Wiki](../../wiki) — Full technical report, architecture decisions, and results
- [GitHub Project Board](../../projects) — Milestone tracking and task assignments

---

*Spring 2026 — University of the Incarnate Word | ENGR 4380 - Senior Design 2 / CSEC 4390 – Practicum*
