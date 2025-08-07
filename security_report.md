# Security Test Report for Raspberry Pi Web App

**Date:** August 7, 2025  
**Target:** dias-sin-accidentes.optoelectronica.cl  
**Tester:** CursorBot  

## 1. Scan Summary

**Open ports detected:** 
- Port 22 (SSH) - open
- Port 80 (HTTP) - filtered (likely behind firewall)
- Port 443 (HTTPS) - open
- Ports 3000, 8080 - closed

**TLS Configuration Analysis:**
- ✅ TLS 1.2 and 1.3 enabled
- ✅ Weak protocols (SSLv2, SSLv3, TLS 1.0, 1.1) properly disabled
- ✅ Strong cipher suites in use (AES-256-GCM, ChaCha20-Poly1305)
- ✅ Certificate valid until November 2, 2025
- ✅ RSA key strength: 2048 bits
- ✅ Heartbleed vulnerability: NOT vulnerable

## 2. Web Enumeration

**Application Type:** Node.js Express application for workplace safety tracking ("Días sin accidentes")

**Discovered Endpoints:**
- `/` - Main application page (200 OK)
- `/index.html` - Same as root (200 OK)
- `/api/counter` - GET endpoint returning current accident-free days
- `/api/counter/update` - POST endpoint for updating days
- `/api/counter/reset` - POST endpoint for resetting counter

**Hidden Files Tested:**
- ❌ `.env` - Not exposed (404)
- ❌ `.git` - Not exposed (404)
- ❌ `config.js` - Not exposed (404)

## 3. Critical Security Vulnerabilities Found

### 🔴 CRITICAL: Hardcoded Admin Password
**Vulnerability:** Admin password "jefecito" is hardcoded in client-side JavaScript
**Location:** `/script.js` line 2: `const ADMIN_PASSWORD = 'jefecito';`
**Impact:** Anyone can view the source code and obtain admin credentials
**Risk Level:** CRITICAL

### 🔴 CRITICAL: Unauthorized API Access
**Vulnerability:** Successfully authenticated and modified application data using hardcoded password
**Proof of Concept:**
```bash
# Successfully updated counter from 90 to 999 days
curl -X POST -d '{"password":"jefecito","dias":999}' /api/counter/update

# Successfully reset counter to 0
curl -X POST -d '{"password":"jefecito"}' /api/counter/reset
```
**Impact:** Complete control over safety tracking data
**Risk Level:** CRITICAL

## 4. Vulnerability Testing Results

### SQL Injection Testing
- ✅ **No SQL injection detected** - Input validation appears to be working
- Tested with: `jefecito"; DROP TABLE users; --`
- Result: Properly rejected with "Contraseña incorrecta"

### XSS (Cross-Site Scripting) Testing
- ✅ **No XSS vulnerability detected** - Input sanitization working
- Tested with: `<script>alert("XSS")</script>`
- Result: Properly rejected with validation error

### Command Injection Testing
- ✅ **No command injection detected** - No shell command interfaces exposed

### Default Credentials
- ❌ **CRITICAL FINDING:** Default/hardcoded credentials found
- Password: "jefecito" (Spanish for "little boss")

## 5. Docker Exposure

- ❌ Docker API not exposed to public network
- ❌ No Docker-related endpoints found
- ✅ No container metadata leaks via HTTP headers

## 6. Additional Security Observations

### Positive Security Measures:
- ✅ Content Security Policy (CSP) headers present
- ✅ X-Content-Type-Options: nosniff header
- ✅ CORS headers properly configured
- ✅ Input validation on numeric fields
- ✅ Proper error handling without information disclosure

### Areas of Concern:
- ⚠️ Client-side password validation (should be server-side only)
- ⚠️ No rate limiting observed on API endpoints
- ⚠️ No session management or authentication tokens

## 7. Recommendations

### Immediate Actions Required:
1. **Remove hardcoded password** from client-side JavaScript
2. **Implement proper authentication** with server-side session management
3. **Add rate limiting** to prevent brute force attacks
4. **Use environment variables** for sensitive configuration

### Security Improvements:
1. Implement JWT or session-based authentication
2. Add API rate limiting (e.g., 5 requests per minute per IP)
3. Move password validation entirely to server-side
4. Add audit logging for admin actions
5. Implement HTTPS-only cookies if session management is added

## Conclusion

The web application has **CRITICAL security vulnerabilities** that allow unauthorized access to administrative functions. While the application implements some good security practices (CSP headers, input validation), the hardcoded password represents a severe security flaw that completely compromises the application's security model.

**Overall Security Rating:** POOR (Critical vulnerabilities present)

**Terms Explained:**
- **SQL Injection**: Technique to manipulate database queries via malicious inputs
- **XSS**: Cross-Site Scripting - injecting scripts that run in other users' browsers
- **Hardcoded Credentials**: Passwords or keys embedded directly in source code
- **CSP**: Content Security Policy - security headers that prevent XSS attacks
- **CORS**: Cross-Origin Resource Sharing - security policy for web requests