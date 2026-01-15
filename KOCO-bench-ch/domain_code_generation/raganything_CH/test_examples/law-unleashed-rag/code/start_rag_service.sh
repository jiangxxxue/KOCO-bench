#!/bin/bash

# RAG-Anything Service Startup Script
# This script handles virtual environment activation, Google Cloud authentication, and starts the app

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
HOST="0.0.0.0"
PORT="8000"

# Detect script location and set paths accordingly
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_NAME="$(basename "${BASH_SOURCE[0]}")"

# Debug information
print_debug() {
    if [ "${DEBUG:-0}" = "1" ]; then
        echo -e "${YELLOW}[DEBUG]${NC} Script directory: $SCRIPT_DIR"
        echo -e "${YELLOW}[DEBUG]${NC} Current directory: $(pwd)"
    fi
}

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if .env file exists
check_env_file() {
    if [ ! -f .env ]; then
        print_error ".env file not found"
        print_status "Please copy env.example to .env and configure it:"
        echo "  cp env.example .env"
        echo "  # Then edit .env with your configuration"
        exit 1
    fi
    print_success ".env file found"
}

# Function to load and validate environment variables
load_env() {
    print_status "Loading environment variables..."
    
    # Load environment variables safely, ignoring comments and empty lines
    while IFS= read -r line; do
        # Skip empty lines and comments
        if [[ -z "$line" || "$line" =~ ^[[:space:]]*# ]]; then
            continue
        fi
        # Export valid environment variables
        if [[ "$line" =~ ^[A-Za-z_][A-Za-z0-9_]*= ]]; then
            export "$line"
        fi
    done < .env
    
    # Check required environment variables
    required_vars=("FIREBASE_PROJECT_ID" "GCS_BUCKET_NAME" "OPENAI_API_KEY")
    for var in "${required_vars[@]}"; do
        if [ -z "${!var}" ]; then
            print_error "Required environment variable $var is not set"
            exit 1
        fi
    done
    print_success "Environment variables loaded and validated"
}

# Function to check if virtual environment exists
check_venv() {
    if [ ! -d "venv" ]; then
        print_status "Creating virtual environment..."
        python3 -m venv venv
        if [ $? -eq 0 ]; then
            print_success "Virtual environment created"
        else
            print_error "Failed to create virtual environment"
            exit 1
        fi
    else
        print_success "Virtual environment found"
    fi
}

# Function to activate virtual environment
activate_venv() {
    print_status "Activating virtual environment..."
    source venv/bin/activate
    if [ $? -eq 0 ]; then
        print_success "Virtual environment activated"
    else
        print_error "Failed to activate virtual environment"
        exit 1
    fi
}

# Function to install dependencies
install_dependencies() {
    print_status "Installing core dependencies..."
    
    # Install core dependencies first
    if pip install -r requirements.txt; then
        print_success "Core dependencies installed"
    else
        print_error "Failed to install core dependencies"
        exit 1
    fi
    
    # Try to install RAG-Anything separately
    print_status "Installing RAG-Anything..."
    if pip install raganything; then
        print_success "RAG-Anything installed"
    else
        print_warning "Failed to install RAG-Anything"
        print_warning "You may need to install it manually:"
        echo "  pip install raganything"
        print_warning "Continuing anyway - RAG processing features may not work"
    fi
}

# Function to check Google Cloud authentication
check_gcloud_auth() {
    print_status "Checking Google Cloud authentication..."
    
    # Check if gcloud is installed
    if ! command -v gcloud &> /dev/null; then
        print_warning "gcloud CLI not found. Skipping authentication check."
        print_warning "Make sure you have GOOGLE_APPLICATION_CREDENTIALS set or run:"
        echo "  gcloud auth application-default login"
        return 0
    fi
    
    # Check if user is authenticated
    if gcloud auth list --filter=status:ACTIVE --format="value(account)" | grep -q .; then
        print_success "Google Cloud authentication is active"
        
        # Check application default credentials
        if gcloud auth application-default print-access-token &> /dev/null; then
            print_success "Application default credentials are valid"
        else
            print_warning "Application default credentials need refresh"
            refresh_gcloud_auth
        fi
    else
        print_warning "No active Google Cloud authentication found"
        refresh_gcloud_auth
    fi
}

# Function to refresh Google Cloud authentication
refresh_gcloud_auth() {
    print_status "Refreshing Google Cloud authentication..."
    echo -e "${YELLOW}This will open a browser window for authentication.${NC}"
    read -p "Do you want to proceed? (y/N): " -n 1 -r
    echo
    
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        gcloud auth application-default login
        if [ $? -eq 0 ]; then
            print_success "Google Cloud authentication completed"
        else
            print_error "Google Cloud authentication failed"
            print_warning "Continuing without authentication - some features may not work"
        fi
    else
        print_warning "Skipping Google Cloud authentication"
        print_warning "Make sure you have GOOGLE_APPLICATION_CREDENTIALS set or run:"
        echo "  gcloud auth application-default login"
    fi
}


# Function to create storage directories
create_directories() {
    print_status "Creating storage directories..."
    mkdir -p storage/kv storage/vector storage/graph storage/status temp
    print_success "Storage directories created"
}

# Function to check if port is available
check_port() {
    if lsof -Pi :$PORT -sTCP:LISTEN -t >/dev/null 2>&1; then
        print_warning "Port $PORT is already in use"
        print_status "Attempting to kill existing process..."
        lsof -ti:$PORT | xargs kill -9 2>/dev/null
        sleep 2
        
        # Check again
        if lsof -Pi :$PORT -sTCP:LISTEN -t >/dev/null 2>&1; then
            print_error "Port $PORT is still in use. Please free the port manually."
            exit 1
        else
            print_success "Port $PORT is now available"
        fi
    else
        print_success "Port $PORT is available"
    fi
}

# Function to start the application
start_app() {
    print_status "Starting RAG-Anything service..."
    print_status "Server will be available at: http://$HOST:$PORT"
    print_status "Health check: http://$HOST:$PORT/health"
    print_status "API docs: http://$HOST:$PORT/docs"
    print_status "Press Ctrl+C to stop the server"
    echo ""
    
    uvicorn src.main:app --host "$HOST" --port "$PORT" --reload
}

# Function to handle cleanup on exit
cleanup() {
    print_status "Shutting down..."
    # Kill any background processes if needed
    exit 0
}

# Set up signal handlers
trap cleanup SIGINT SIGTERM

# Main execution
main() {
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}    RAG-Anything Service Startup${NC}"
    echo -e "${BLUE}========================================${NC}"
    echo ""
    
    # Show debug information if DEBUG=1
    print_debug
    
    # Run all checks and setup
    check_env_file
    load_env
    check_venv
    activate_venv
    install_dependencies
    check_gcloud_auth
    create_directories
    check_port
    
    echo ""
    print_success "All checks passed. Starting application..."
    echo ""
    
    # Start the application
    start_app
}

# Run main function
main "$@"
