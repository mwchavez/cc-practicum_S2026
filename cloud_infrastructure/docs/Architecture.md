# Architecture Decision Log — Hybrid Leak Detection System

## System Overview

This document records the architectural decisions made during the design and implementation of the cloud infrastructure for the Hybrid Leak Detection System. The system ingests multi-modal sensor data from ESP32-based nodes, stores it in a central repository on AWS, and triggers alerts when readings exceed predefined thresholds.

The cloud infrastructure is defined entirely as Infrastructure as Code (IaC) using **AWS CDK (Python)** and deployed as a single CloudFormation stack.

---

## Data Flow

```
ESP32 Nodes (MQTT over TLS)
        │
        ▼
   AWS IoT Core
   (Message Broker)
        │
        ▼
   IoT Rules Engine
   (SQL: SELECT * FROM 'leaksensor/+/data')
        │
        ▼
   DynamoDB Table
   (PK: device_id / SK: timestamp)
        │
        ▼
   CloudWatch Metric Alarms
        │
        ▼
   SNS → Email Alert
```

Each ESP32 node publishes JSON payloads over MQTT to the topic pattern `leaksensor/node-XX/data`. AWS IoT Core authenticates the device via X.509 certificates and mTLS, then the Rules Engine evaluates incoming messages and routes them into DynamoDB using a `DynamoDBv2` action. CloudWatch monitors metric thresholds and fires SNS email notifications when anomalous readings are detected.

---

## Architecture Decisions

### ADR-1: AWS CDK over Terraform or Console

**Decision:** Use AWS CDK (Python) as the IaC tool.

**Rationale:** CDK allows defining cloud resources in a general-purpose programming language, which provides variable reuse, loops, and conditional logic that declarative tools like raw CloudFormation JSON/YAML do not. Python was chosen because it is the team's strongest language. CDK also generates CloudFormation under the hood, so the deployment is fully reproducible and version-controlled.

**Trade-off:** CDK has a steeper initial learning curve than console-based provisioning, but produces a portable, repeatable deployment.

---

### ADR-2: DynamoDB with On-Demand Billing

**Decision:** Use DynamoDB in `ON_DEMAND` billing mode with `device_id` (String) as the partition key and `timestamp` (Number) as the sort key.

**Rationale:** On-demand billing eliminates the need to estimate read/write capacity units, which is appropriate for a prototype with unpredictable or bursty traffic patterns. The composite key structure supports efficient queries: retrieving all readings from a specific device sorted by time, or querying a time range for a single device.

**Trade-off:** On-demand is more expensive per-request at high volume compared to provisioned capacity, but cost is negligible at prototype scale and avoids throttling risk.

---

### ADR-3: IoT Core Certificate-Based Authentication (mTLS)

**Decision:** Authenticate ESP32 devices using X.509 certificates attached to IoT Things, with certificate ARNs passed into the CDK stack as `CfnParameter` values.

**Rationale:** Certificate-based mutual TLS is the AWS-recommended authentication mechanism for IoT devices. It provides strong device identity without embedding IAM credentials on constrained hardware. Using `CfnParameter` for certificate ARNs keeps the stack portable — anyone cloning the repo can supply their own certificate ARNs at deploy time without modifying source code.

**Security note:** Certificate `.pem` files are excluded from version control via `.gitignore`. Only the ARN references are used in the stack.

---

### ADR-4: IoT Policy — Least-Privilege with Wildcard Scoping

**Decision:** A single IoT policy (`leak_detection_nodes_policy`) grants `iot:Connect`, `iot:Publish`, and `iot:Subscribe` permissions scoped to `node*` client IDs and `leaksensor/node-*/data` topic patterns.

**Rationale:** Least-privilege access control limits each node to connecting with a `node*`-prefixed client ID and publishing/subscribing only to the leak sensor data topics. The wildcard pattern allows the policy to cover both nodes (and future nodes) without per-device policies.

**Scalability consideration:** In a production deployment with many nodes across different physical zones, the policy could be segmented per node group (e.g., per building or floor) to further restrict the blast radius.

---

### ADR-5: IAM Role for IoT Rules Engine → DynamoDB

**Decision:** Create a dedicated IAM role (`Table_Write_Role`) assumed by `iot.amazonaws.com` with a policy granting only `dynamodb:PutItem` on the specific table ARN.

**Rationale:** AWS services should authenticate via IAM roles, not IAM users. The IoT Rules Engine assumes this role temporarily when writing to DynamoDB. The policy is scoped to a single action (`PutItem`) on a single resource (the table ARN), following the principle of least privilege.

---

### ADR-6: IoT Topic Rule — SQL-Based Message Routing

**Decision:** Use the IoT Rules Engine with the SQL statement `SELECT * FROM 'leaksensor/+/data'` to route all sensor messages into DynamoDB via a `DynamoDBv2` action.

**Rationale:** The `+` wildcard in the MQTT topic filter matches any single topic level, capturing messages from all nodes (e.g., `leaksensor/node-01/data`, `leaksensor/node-02/data`) with a single rule. The `DynamoDBv2` action writes the full JSON payload as a DynamoDB item, preserving all sensor fields without manual column mapping.

**Important distinction:** IoT Core acts as a message router, not a data store. The MQTT Test Client is a debugging tool for verifying message flow — it is not part of the production data pipeline.

---

### ADR-7: Threshold-Based Alerting over Machine Learning

**Decision:** Use CloudWatch metric alarms with static thresholds for leak detection, rather than ML-based anomaly detection (e.g., AWS Random Cut Forest).

**Rationale:** The controlled test environment has a known, fixed baseline for "normal" sensor readings. When normal is predictable — consistent pipe pressure, stable ambient temperature, no occupancy variation — simple threshold comparisons (e.g., moisture > X for Y consecutive periods) are sufficient and far simpler to implement, debug, and explain.

ML-based detection becomes justified when "normal" is contextual: variable building occupancy, seasonal temperature shifts, cross-sensor interdependencies, or when the system scales to hundreds of nodes where manual threshold tuning is impractical.

**Future work:** ML is scoped as future work for larger-scale or real-world deployments where contextual baselines are necessary.

---

### ADR-8: SNS Email Alerts with Parameterized Recipient

**Decision:** Use an SNS topic with an email subscription for alert delivery. The recipient email is passed as a `CfnParameter`.

**Rationale:** SNS provides a managed, scalable notification service that integrates directly with CloudWatch alarms. Email is sufficient for a prototype alerting mechanism. Parameterizing the email address maintains stack portability.

**Future extension:** Additional subscription protocols (SMS, Lambda, HTTP webhook) can be added without modifying the core alarm logic.

---

## Security Posture

| Layer | Measure |
|-------|---------|
| Device ↔ Cloud | mTLS via X.509 certificates; encrypted MQTT over TLS 1.2 |
| Device Identity | IoT Things with certificate-based principal attachments |
| Device Permissions | IoT Policy with least-privilege Connect/Publish/Subscribe |
| Service-to-Service | IAM Role assumed by IoT Rules Engine; scoped to `PutItem` only |
| Data at Rest | DynamoDB encryption enabled by default (AWS-managed keys) |
| Stack Portability | Certificate ARNs and email as `CfnParameter` — no secrets in code |
| Version Control | `.gitignore` excludes certificate files, credentials, and `cdk.out` |

---

## AWS Services Used

| Service | Role in System |
|---------|---------------|
| AWS IoT Core | MQTT message broker, device registry, certificate-based auth |
| IoT Rules Engine | SQL-based message routing from MQTT topics to DynamoDB |
| DynamoDB | Time-series storage for sensor readings |
| IAM | Role-based access control for service-to-service permissions |
| CloudWatch | Metric monitoring and threshold-based alarms |
| SNS | Alert notification delivery (email) |
| CloudFormation (via CDK) | Infrastructure as Code deployment and stack management |

---

## Status

- [x] DynamoDB table (on-demand, composite key)
- [x] IoT Core Things (node01, node02)
- [x] IoT Policy (least-privilege)
- [x] Certificate attachments (parameterized ARNs)
- [x] IAM Role for Rules Engine → DynamoDB
- [x] IoT Topic Rule (SQL routing, DynamoDBv2 action)
- [x] SNS Topic with email subscription (parameterized)
- [ ] CloudWatch metric alarms
- [ ] CloudWatch dashboard
- [ ] End-to-end integration test with ESP32 hardware
- [ ] Live demonstration
