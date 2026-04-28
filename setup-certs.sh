#!/bin/bash
set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"

echo "=== Generating SSL certificates for api.openai.com ==="

# Generate CA private key
openssl genrsa -out ca-key.pem 2048 2>/dev/null

# Generate CA certificate
openssl req -x509 -new -nodes -key ca-key.pem -sha256 -days 3650 \
  -subj "/C=US/O=LocalProxy/CN=LocalProxy CA" \
  -out ca-cert.pem

# Generate server private key
openssl genrsa -out server-key.pem 2048 2>/dev/null

# Generate server CSR
openssl req -new -key server-key.pem \
  -subj "/C=US/O=LocalProxy/CN=api.openai.com" \
  -out server.csr

# Create SAN extension file
cat > server-ext.cnf <<EOF
basicConstraints=CA:FALSE
keyUsage = digitalSignature, keyEncipherment
extendedKeyUsage = serverAuth
subjectAltName = @alt_names

[alt_names]
DNS.1 = api.openai.com
DNS.2 = *.api.openai.com
EOF

# Sign server certificate with CA
openssl x509 -req -in server.csr -CA ca-cert.pem -CAkey ca-key.pem \
  -CAcreateserial -out server-cert.pem -days 365 -sha256 \
  -extfile server-ext.cnf

# Clean up temp files
rm -f server.csr server-ext.cnf

echo ""
echo "=== Trusting CA in macOS System Keychain ==="
echo "This requires your admin password..."

sudo security add-trusted-cert -d -r trustRoot \
  -k /Library/Keychains/System.keychain ca-cert.pem

echo ""
echo "=== Certificates ready ==="
echo "CA:      $DIR/ca-cert.pem"
echo "Server:  $DIR/server-cert.pem"
echo "Key:     $DIR/server-key.pem"
