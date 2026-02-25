# Cloud Infrastructure Projects - Spring 2026

This repository contains cloud computing work from two courses, demonstrating 
the application of a unified AWS architecture to different problem domains.

## Repository Structure

This repo houses cloud infrastructure implementations for:

### 📊 CSEC 4390 - Practicum (Senior Design)
**Project:** Cloud Backend for Hybrid Leak Detection System  
**Faculty:** Dr. Okan Caglayan & Dr. Parra  
**Focus:** IoT data pipeline for distributed water leak sensors and detection of said leaks.

### ☁️ CIS 4355 - Cloud Computing  
**Project:** Cloud HoneyPot Monitoring Platform  
**Faculty:** Dr. Parra  
**Focus:** Create a cloud-native Honeypot and securiyt event monitoring platform deployed on GCP.

## Why One Repository?

Rather than duplicating AWS infrastructure code across two separate repos, this 
unified repository demonstrates understanding of reusable cloud architecture 
patterns. The core components—IoT data ingestion, storage, real-time analytics, 
and alerting—form a general-purpose framework applicable to multiple IoT and 
data-driven use cases.

### Shared Architecture Components:
- AWS IoT Core (or equivalent data ingestion service)
- Database storage (DynamoDB/RDS)
- Real-time data processing and analytics
- Alert/notification system
- Web-based dashboard for visualization
- CloudWatch monitoring

### Different Applications:
- **Practicum:** Receives multi-modal sensor data from leak detection devices
- **Cloud Computing:** [Applies same pipeline to different data/problem]

## Project Directories
```
/practicum-leak-detection/
    └── [Leak detection specific implementation]

/cloud-computing-final/
    └── [Final project specific implementation]

/shared/
    └── [Reusable infrastructure templates, configs, documentation]
```

## Learning Objectives

This unified approach allows me to:
1. Master both AWS & GCP services deeply rather than just one
2. Understand cloud architecture as reusable patterns
3. Focus on problem-specific implementation details
4. Demonstrate infrastructure-as-code best practices

---
*Spring 2026 - Engineering & Computer Information Systems*
