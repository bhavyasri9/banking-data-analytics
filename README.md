# banking-data-analytics
capgemini project
[![View Video Demonstartion]([https://img.shields.io/badge/View-Project-blue?style=for-the-badge](https://youtu.be/h6x7DUQ4TQI))]

# Banking Data Analytics Pipeline using Delta Live Tables

## Project Description
This mini project demonstrates an **end-to-end banking data analytics pipeline** built on Databricks. It processes raw transactional data using the **Medallion Architecture (Bronze, Silver, Gold)** and transforms it into actionable business insights. The project includes data ingestion, cleaning, KPI generation, and dashboard visualization for analyzing customer behavior, revenue trends, and financial risk.

## Business Problem
Banks generate large volumes of transactional data but often lack structured insights to:
- Identify high-value and high-risk customers  
- Monitor revenue and transaction trends  
- Detect unusual or suspicious activities  

This project addresses these challenges by building a **data-driven analytics solution** that supports better decision-making and risk management.

## Objectives
Build a scalable data pipeline using Delta Live Tables
Clean and standardize raw banking data
Design a star schema for analytics
Generate business KPIs
Create an interactive dashboard
Identify high-risk customers using transaction behavior

## Dataset Description

The dataset simulates a banking system and contains multiple tables:

Customers → Customer details (ID, name, city, etc.)
Accounts → Account information (type, balance, branch)
Transactions → Transaction data (amount, type, date)
Branches → Branch details (location)
Loans → Loan-related information

Dataset Size: ~1000 records across multiple tables
Purpose: Analyze customer behavior, transactions, and financial risk

## Architecture
Bronze Layer → Raw Data Ingestion (CSV files)
       ↓
Silver Layer → Data Cleaning & Transformation
       ↓
Gold Layer → Business Insights + KPIs + Dashboard

### Bronze Layer
- Ingest raw CSV data  
- Store data in Delta format  
- Simulate real-world data issues (nulls, duplicates)  

### Silver Layer
- Clean and preprocess data  
- Handle null values and duplicates  
- Standardize formats and data types  

### Gold Layer
- Design star schema (fact & dimension tables)  
- Generate KPIs  
- Enable dashboard analytics  

## Tech Stack
- Databricks  
- Delta Live Tables (DLT)  
- PySpark  
- SQL  
- Delta Lake  
- Databricks SQL (Dashboard)  

## Key Features
- End-to-end automated data pipeline  
- Data cleaning and transformation  
- Star schema modeling  
- KPI generation (revenue, transactions, etc.)  
- Risk analysis for high-value customers  
- Interactive dashboard  

## Key KPIs
- Total Revenue  
- Total Transactions  
- Average Transaction Value  
- Top Customers  
- Branch Performance  
- Monthly Revenue Trends  

## Risk Analysis
A risk scoring model is implemented using:
- Transaction count  
- Total transaction amount  
- Average transaction value  

### Insights:
- Risk is concentrated among a few customers  
- High-value transactions indicate potential anomalies  
- Helps in early detection of risky behavior  

## Dashboard
The dashboard is built using Databricks SQL and includes:
- KPI cards for quick insights  
- Area charts for trend analysis  
- Bar charts for customer and branch performance  
- Pie charts for transaction distribution  
- Risk analysis section  

## Outcome
This project transforms raw data into **actionable insights**, enabling:
- Better financial decision-making  
- Identification of high-risk customers  
- Improved understanding of customer behavior  

## Future Enhancements
- Real-time data ingestion using Auto Loader  
- Machine learning-based fraud detection  
- Alert system for high-risk transactions  
- CI/CD pipeline integration  
- Integration with Power BI or Tableau  

## 👥 Team
- **Bhavya Sri Lakshmi Palla** – Gold Layer, KPIs, Dashboard, Pipeline  
- **Narra Lakshmi Hanuma** – Bronze Layer (Data Ingestion)  
- **Munagala Sai Manikanta Yaswanth** – Silver Layer (Data Cleaning & Transformation)  
