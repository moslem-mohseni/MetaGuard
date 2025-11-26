# Security Policy

**Author:** Moslem Mohseni

## Supported Versions

We release patches for security vulnerabilities in the following versions:

| Version | Supported          |
| ------- | ------------------ |
| 1.1.x   | :white_check_mark: |
| 1.0.x   | :white_check_mark: |
| < 1.0   | :x:                |

## Reporting a Vulnerability

We take security seriously. If you discover a security vulnerability within MetaGuard, please follow these steps:

### 1. Do NOT Open a Public Issue

Security vulnerabilities should not be disclosed publicly until we've had a chance to address them.

### 2. Report Privately

Send a detailed report to: **security@metaguard.dev** (or create a private security advisory on GitHub)

Include the following information:
- Description of the vulnerability
- Steps to reproduce the issue
- Potential impact assessment
- Suggested fix (if you have one)

### 3. Response Timeline

- **Initial Response:** Within 48 hours
- **Status Update:** Within 7 days
- **Fix Timeline:** Depends on severity (Critical: 24-48 hours, High: 7 days, Medium: 30 days)

## Security Best Practices

### API Deployment

When deploying the MetaGuard API in production:

1. **Use HTTPS Only**
   ```nginx
   server {
       listen 443 ssl;
       # ... SSL configuration
   }
   ```

2. **Configure CORS Properly**
   ```python
   app.add_middleware(
       CORSMiddleware,
       allow_origins=["https://yourdomain.com"],  # Don't use "*" in production
       allow_credentials=True,
       allow_methods=["GET", "POST"],
       allow_headers=["*"],
   )
   ```

3. **Rate Limiting**
   Consider using a reverse proxy or middleware for rate limiting.

4. **Authentication**
   The API doesn't include authentication by default. For production:
   - Implement API key authentication
   - Use OAuth2/JWT for user authentication
   - Consider using an API gateway

### Environment Variables

Never commit sensitive data. Use environment variables:

```bash
# Required for production
METAGUARD_LOG_LEVEL=INFO
METAGUARD_RISK_THRESHOLD=0.5

# If using custom model
METAGUARD_MODEL_PATH=/secure/path/to/model.pkl
```

### Docker Security

The production Dockerfile includes these security measures:

1. **Non-root user**: Runs as `metaguard` user
2. **Minimal base image**: Uses `python:3.11-slim`
3. **Multi-stage build**: Reduces attack surface
4. **No shell in production**: Consider using `distroless` for maximum security

### Dependency Security

We regularly scan dependencies for vulnerabilities:

```bash
# Check for vulnerable packages
pip-audit

# Check code for security issues
bandit -r src/metaguard
```

## Security Features

### Input Validation

All transaction inputs are validated:

- `amount`: Must be positive number
- `hour`: Must be 0-23
- `user_age_days`: Must be >= 1
- `transaction_count`: Must be >= 0

Invalid inputs are rejected with appropriate error messages.

### Model Security

- Models are loaded from trusted paths only
- Pickle files should only come from trusted sources
- Consider using model signing for additional security

## Known Security Considerations

1. **Pickle Security**: We use pickle for model serialization. Only load models from trusted sources.

2. **No Built-in Auth**: The API doesn't include authentication. Implement appropriate auth for your use case.

3. **Logging**: Be careful not to log sensitive transaction data in production.

## Security Updates

Subscribe to our security advisories:
- Watch this repository on GitHub
- Check the [CHANGELOG.md](CHANGELOG.md) for security-related updates

## Acknowledgments

We appreciate responsible disclosure of security vulnerabilities. Contributors who help improve our security will be acknowledged in our release notes (with permission).
