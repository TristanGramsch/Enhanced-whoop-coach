# Security Test Report for Raspberry Pi Web App

**Date:** August 7, 2025  
**Target:** dias-sin-accidentes.optoelectronica.cl  
**Tester:** CursorBot  

## 1. Scan Summary

Open ports detected:
- **Port 22 (SSH)**: OpenSSH 8.4p1 Debian
- **Port 443 (HTTPS)**: Node.js Express framework

The server is running on a Debian-based system with minimal exposed services, which is good security practice.

## 2. SSL/TLS Configuration

TLS check revealed:
- **Strong Configuration**: TLS 1.2 and 1.3 supported, older protocols disabled
- **Strong Ciphers**: Modern cipher suites with forward secrecy
- **Valid Certificate**: Let's Encrypt certificate valid until November 2, 2025
- **Grade B**: Certificate chain incomplete (missing intermediate certificate)
- **Missing Security Headers**: No HSTS (Strict-Transport-Security) configured

## 3. Web Enumeration

Directory enumeration discovered these exposed files:
- `/package.json` - Reveals application dependencies and structure
- `/Dockerfile` - Exposes deployment configuration
- `/package-lock.json` - Contains detailed dependency tree
- `/node_modules/` - Directory listing enabled (potential information disclosure)

No sensitive configuration files (`.env`, `.git`) were found exposed.

## 4. Vulnerability Testing

### Critical Finding: Hardcoded Admin Credentials
- **Admin password found in client-side JavaScript**: `jefecito`
- Located in `/script.js` line 2: `const ADMIN_PASSWORD = 'jefecito';`
- This allows unauthorized access to the admin panel

### Other Security Tests:
- **SQL Injection**: No injectable fields detected
- **XSS (Cross-Site Scripting)**: Input appears to be properly sanitized
- **Command Injection**: No command execution interfaces found
- **CSRF Protection**: Not implemented but low risk due to simple functionality

## 5. Docker Exposure

Docker security assessment:
- **Docker API ports (2375-2377)**: Not exposed to public network ✓
- **Container metadata**: No Docker-specific headers leaked ✓
- **Dockerfile exposed**: Yes, but contains no sensitive information

## 6. Additional Findings

- **Missing Security Headers**:
  - No Content-Security-Policy (except on 404 pages)
  - No X-Frame-Options
  - No X-XSS-Protection
  - No Strict-Transport-Security (HSTS)
  
- **Information Disclosure**:
  - Server identifies as Express via X-Powered-By header
  - Application structure revealed through exposed files

## Recommendations

1. **Critical**: Remove hardcoded password from client-side code
2. **High**: Complete the SSL certificate chain
3. **High**: Implement proper authentication with server-side validation
4. **Medium**: Add security headers (HSTS, CSP, X-Frame-Options)
5. **Medium**: Disable directory listing on `/node_modules/`
6. **Low**: Remove or restrict access to `/package.json` and `/Dockerfile`
7. **Low**: Remove X-Powered-By header to reduce fingerprinting

## Terms Explained

- **SQL Injection**: Technique to manipulate database queries via user inputs
- **XSS**: Injecting malicious scripts that execute in users' browsers
- **HSTS**: Forces browsers to use HTTPS, preventing downgrade attacks
- **Certificate Chain**: The trust path from server certificate to root CA
- **Forward Secrecy**: Ensures past communications remain secure even if keys are compromised

## Conclusion

The web application has a **critical security vulnerability** with hardcoded admin credentials exposed in client-side code. While the infrastructure security is reasonable (strong TLS, minimal exposed services), the application-level security needs immediate attention. The hardcoded password allows anyone to access the admin panel and modify the accident counter, compromising the integrity of the safety tracking system.