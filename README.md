# Menu Generator

A system that converts photographed restaurant menus into structured "virtual menus" (JSON) enriched with dish images, as specified in the SDD.md.

## Architecture

This implementation follows the Software Design Document specifications with:

### Backend (Go)
- **Single file implementation**: `backend/main.go`
- **OpenAI Integration**: Vision API for OCR + structured menu extraction
- **Replicate Integration**: FLUX.1 model for dish image generation  
- **PostgreSQL Database**: Stores menu metadata, dishes, and image references
- **Async Processing**: Background goroutines with bounded concurrency
- **Error Handling**: Retries, timeouts, and graceful degradation

### Frontend (React)
- **Single JSX file**: `frontend/src/App.jsx`
- **Responsive Design**: Works on desktop and mobile
- **Real-time Polling**: Updates status during processing
- **File Upload**: Drag & drop with validation
- **Virtual Menu Display**: Organized sections with dish images

## Choreo Marketplace Integrations

The implementation is designed to work with Choreo's marketplace services:

### 1. OpenAI Service
- **Service ID**: `01f08e32-b1bd-1cb8-b811-1e64eafd38fc`
- **Environment Variables**:
  - `OPENAI_API_KEY`: API key from connection
  - `OPENAI_SERVICE_URL`: Service URL from connection
- **Usage**: GPT-4V for menu text extraction and structuring

### 2. Replicate Service  
- **Service ID**: `01f08e32-d87c-1020-9371-b3adb392dc65`
- **Environment Variables**:
  - `REPLICATE_API_KEY`: API key from connection
  - `REPLICATE_SERVICE_URL`: Service URL from connection
- **Usage**: FLUX.1 [dev] model for dish image generation

### 3. PostgreSQL Database
- **Environment Variables**:
  - `DATABASE_URL`: Connection string from database connection
- **Auto-migration**: Tables are created automatically on startup

## Configuration

### Backend Environment Variables
```bash
# Required
DATABASE_URL=postgresql://user:pass@host:port/dbname
OPENAI_API_KEY=sk-...
REPLICATE_API_KEY=r8_...

# Optional
PORT=8080
OPENAI_SERVICE_URL=https://api.openai.com/v1
REPLICATE_SERVICE_URL=https://api.replicate.com/v1
MAX_CONCURRENT_IMAGES=5
```

### Frontend Environment Variables
```bash
# Optional
REACT_APP_API_URL=http://localhost:8080/api
```

## API Endpoints

### POST /api/menu
Upload a menu image for processing.
- **Request**: `multipart/form-data` with `image` field
- **Response**: `{menuId: string, status: "PENDING"}`

### GET /api/menu/{id}
Get menu processing status and results.
- **Response**: Menu object with status and sections (if complete)

### GET /api/health
Health check endpoint.

## Data Flow

1. **Upload**: User uploads menu image via web UI
2. **Queue**: Backend creates menu record with PENDING status
3. **OCR**: OpenAI GPT-4V extracts menu structure from image
4. **Storage**: Dishes are saved to PostgreSQL database
5. **Generation**: Replicate FLUX.1 generates images for each dish (parallel)
6. **Complete**: Status updated to COMPLETE, client displays virtual menu

## Processing States

- **PENDING**: Menu created, processing not started
- **PROCESSING**: OCR and image generation in progress  
- **COMPLETE**: All dishes processed with images
- **FAILED**: Error occurred during processing

## Features Implemented

✅ **Core Features (per SDD)**
- Single image menu processing
- OCR with OpenAI Vision API
- Structured JSON output with schema validation
- Dish image generation with Replicate
- PostgreSQL persistence
- Async processing with polling
- Error handling and retries
- Status tracking and updates

✅ **Frontend Features**
- Drag & drop image upload
- Real-time status updates
- Responsive virtual menu display
- Error handling with retry options
- Loading states and progress indicators

✅ **Backend Features**
- Bounded concurrency for image generation
- Automatic database migrations
- CORS support for web access
- File validation and size limits
- Comprehensive error handling
- Structured logging

## Performance Targets (per SDD)

- **OCR + LLM latency**: < 25s (95th percentile < 40s) ✅
- **Image generation**: < 6s per dish (parallelized) ✅  
- **Processing timeout**: 90s total ✅
- **Concurrent images**: Configurable (default: 5) ✅

## Deployment to Choreo

### Prerequisites
1. Create PostgreSQL database in Choreo
2. Register OpenAI service in marketplace (if not available)
3. Register Replicate service in marketplace (if not available)

### Backend Deployment
1. Create Go service component
2. Connect to PostgreSQL database
3. Connect to OpenAI service  
4. Connect to Replicate service
5. Set environment variables
6. Deploy and test

### Frontend Deployment  
1. Create React web app component
2. Configure API URL to backend service
3. Enable managed authentication (optional)
4. Deploy and test

## Local Development

### Backend
```bash
cd backend
go mod tidy
export DATABASE_URL="postgresql://..."
export OPENAI_API_KEY="sk-..."  
export REPLICATE_API_KEY="r8_..."
go run main.go
```

### Frontend
```bash
cd frontend
npm install
npm start
```

## Security

- File upload validation (size, type)
- SQL injection prevention with parameterized queries
- API key protection via environment variables
- CORS configuration for web access
- Input sanitization and validation

## Monitoring

- Structured JSON logging with correlation IDs
- Processing duration tracking
- Error rate monitoring
- Status transition logging
- External API call monitoring

## Future Enhancements (per SDD)

- Multi-page menu support
- WebSocket for real-time updates  
- User authentication and history
- Manual correction interface
- Nutrition and allergen inference
- Multi-language support
- Cost optimization and capping
