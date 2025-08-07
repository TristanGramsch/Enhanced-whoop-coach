# Security Test Report for Raspberry Pi Web App

**Date:** August 7, 2025  
**Target:** dias-sin-accidentes.optoelectronica.cl (130.44.115.94)  
**Tester:** CursorBot  
**Assessment Type:** Ethical Security Testing  

## Executive Summary

This security assessment revealed **critical vulnerabilities** in the target Node.js web application. The most significant finding is exposed source code containing hardcoded credentials, along with publicly accessible sensitive data files. While the TLS configuration is secure, the application-level security has multiple high-risk exposures.

## 1. Scan Summary

**Open Ports Detected:**
- Port 22/tcp: SSH (OpenSSH 8.4p1 Debian)
- Port 443/tcp: HTTPS (Node.js Express framework)
- Port 53/tcp: DNS (tcpwrapped)
- Filtered ports: 25, 80, 135, 139, 445 (typical Windows/mail ports)

**Key Findings:**
- No unexpected services exposed
- SSH service available but not tested (out of scope)
- Web application running on Express.js framework

## 2. SSL/TLS Security Analysis

**Certificate Information:**
- Issuer: Let's Encrypt (R10)
- Valid from: Aug 4, 2025 19:18:13 GMT
- Valid until: Nov 2, 2025 19:18:12 GMT
- Subject: dias-sin-accidentes.optoelectronica.cl

**Security Status:**
- ✅ Strong cipher suites supported (TLS 1.3 preferred)
- ✅ No weak ciphers detected
- ✅ Certificate is valid and properly configured
- ✅ HTTPS properly enforced
- ✅ No heartbleed vulnerability
- ✅ TLS compression disabled

**Note:** SSL certificate verification initially failed due to local certificate store, but the certificate itself is valid.

## 3. Web Application Enumeration

**Discovered Endpoints:**
- `/` - Main application page
- `/api/counter` - Returns current counter data
- `/api/export` - **Publicly accessible data export**
- `/api/counter/reset` - Admin-only reset function
- `/api/counter/update` - Admin-only update function
- `/api/import` - Admin-only data import function

**Exposed Files (CRITICAL):**
- `/server.js` - **Complete source code exposed**
- `/package.json` - Application dependencies and metadata
- `/data.json` - **Application data file exposed**
- `/node_modules/` - Node.js dependencies accessible

## 4. Critical Vulnerability Findings

### 🔴 CRITICAL: Source Code Exposure
**Risk Level:** Critical  
**Description:** The entire Node.js server source code is publicly accessible.  
**Evidence:** `curl -k -s https://dias-sin-accidentes.optoelectronica.cl/server.js` returns full source code  
**Impact:** Complete application logic exposed, including authentication mechanisms  

### 🔴 CRITICAL: Hardcoded Credentials
**Risk Level:** Critical  
**Description:** Admin password hardcoded in exposed source code.  
**Evidence:** `const ADMIN_PASSWORD = 'jefecito';` found in server.js  
**Impact:** Full administrative access to application functions  

### 🔴 HIGH: Sensitive Data Exposure
**Risk Level:** High  
**Description:** Application data file directly accessible via web.  
**Evidence:** `/data.json` returns current application state and metadata  
**Impact:** Information disclosure of application data  

### 🔴 HIGH: Dependency Information Disclosure
**Risk Level:** High  
**Description:** Node.js modules and package information exposed.  
**Evidence:** `/node_modules/express/package.json` accessible  
**Impact:** Reveals dependency versions for targeted attacks  

### 🟡 MEDIUM: Static File Serving Misconfiguration
**Risk Level:** Medium  
**Description:** Express.js configured to serve all files from current directory.  
**Evidence:** `app.use(express.static('./'));` in source code  
**Impact:** Potential exposure of additional sensitive files  

## 5. Vulnerability Testing Results

### SQL Injection
**Status:** ✅ Not Applicable  
**Details:** Application uses file-based storage (JSON), no database detected  

### Cross-Site Scripting (XSS)
**Status:** ✅ No Stored XSS Found  
**Details:** Application primarily serves static content and JSON APIs  
**Note:** Limited input vectors as most functionality requires authentication  

### Command Injection
**Status:** ✅ No Direct Vectors Found  
**Details:** No command execution endpoints identified in exposed source code  

### Authentication Bypass
**Status:** 🔴 Credentials Compromised  
**Details:** Admin password exposed in source code enables full system access  

## 6. Docker Security Assessment

**Docker API Exposure:**
- ✅ Port 2375 (HTTP): Not exposed
- ✅ Port 2376 (HTTPS): Not exposed
- ✅ No Docker API accessible from public network

**Container Information:**
- Application running in Docker container (evident from package.json scripts)
- Container properly isolated from Docker daemon

## 7. Application Logic Analysis

**Functionality Discovered:**
- Days without accidents counter application
- Daily auto-increment functionality
- Admin functions for manual updates and resets
- Data export/import capabilities
- Static file serving for web interface

**Business Logic Issues:**
- No rate limiting on API endpoints
- No session management or multi-user support
- Simple password-based authentication only

## 8. Recommendations

### 🔥 Immediate Actions Required:

1. **Remove Source Code Exposure**
   - Configure web server to exclude .js files from static serving
   - Implement proper file access controls

2. **Secure Credentials**
   - Remove hardcoded password from source code
   - Implement environment variable-based configuration
   - Use strong, randomly generated passwords

3. **Fix Static File Serving**
   - Restrict static file serving to only necessary directories (e.g., `/public`)
   - Exclude sensitive files (.json, .js, node_modules)

### 🛡️ Security Improvements:

1. **Authentication & Authorization**
   - Implement proper session management
   - Add rate limiting to prevent brute force attacks
   - Consider multi-factor authentication for admin functions

2. **Application Security**
   - Add input validation and sanitization
   - Implement proper error handling
   - Add security headers (CSP, HSTS, etc.)

3. **Infrastructure Security**
   - Regularly update dependencies
   - Implement proper logging and monitoring
   - Consider Web Application Firewall (WAF)

## 9. Technical Evidence

### Key Commands Used:
```bash
# Port scanning
nmap -sS -sV --top-ports 1000 dias-sin-accidentes.optoelectronica.cl

# SSL analysis
sslscan --show-certificate dias-sin-accidentes.optoelectronica.cl:443

# Directory enumeration
gobuster dir -u https://dias-sin-accidentes.optoelectronica.cl/ -w wordlist -k

# Source code retrieval
curl -k -s https://dias-sin-accidentes.optoelectronica.cl/server.js
```

### Compromised Credentials:
- **Admin Password:** `jefecito`
- **Access Level:** Full administrative control
- **Affected Endpoints:** `/api/counter/reset`, `/api/counter/update`, `/api/import`

## 10. Terms Explained

- **SQL Injection**: Technique to manipulate database queries via malicious inputs to access unauthorized data
- **XSS (Cross-Site Scripting)**: Injecting malicious scripts that execute in other users' browsers to steal data or perform actions
- **Gobuster**: A tool to brute-force and discover hidden web paths and directories
- **Source Code Exposure**: Unintentional public access to application source code, revealing business logic and potential vulnerabilities
- **Hardcoded Credentials**: Passwords or keys directly embedded in source code instead of secure configuration
- **Static File Serving**: Web server functionality that serves files directly from the filesystem without processing

## Conclusion

The target application demonstrates **critical security vulnerabilities** primarily due to misconfigured static file serving that exposes sensitive source code and data files. While the underlying infrastructure (TLS, Docker isolation) shows good security practices, the application-level security requires immediate attention.

**Risk Assessment:** HIGH - Immediate remediation required  
**Compliance Impact:** Significant security policy violations  
**Business Impact:** Complete compromise of application administrative functions

The exposed credentials provide full administrative access to the application, allowing unauthorized users to reset counters, modify data, and access sensitive information. Immediate action is required to secure the application.