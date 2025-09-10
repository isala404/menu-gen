import React, { useState, useEffect, useCallback } from 'react';
import './App.css';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8080/api';

// Types
const MenuStatus = {
  PENDING: 'PENDING',
  PROCESSING: 'PROCESSING',
  COMPLETE: 'COMPLETE',
  FAILED: 'FAILED'
};

// Custom hooks
const usePolling = (menuId, status) => {
  const [menu, setMenu] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const fetchMenu = useCallback(async () => {
    if (!menuId) return;
    
    try {
      setLoading(true);
      const response = await fetch(`${API_BASE_URL}/menu/${menuId}`);
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      
      const data = await response.json();
      setMenu(data);
      setError(null);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }, [menuId]);

  useEffect(() => {
    if (!menuId || status === MenuStatus.COMPLETE || status === MenuStatus.FAILED) {
      return;
    }

    // Initial fetch
    fetchMenu();

    // Setup polling
    const interval = setInterval(() => {
      fetchMenu();
    }, 2000); // Poll every 2 seconds

    // Cleanup after 60 seconds max
    const timeout = setTimeout(() => {
      clearInterval(interval);
    }, 60000);

    return () => {
      clearInterval(interval);
      clearTimeout(timeout);
    };
  }, [menuId, status, fetchMenu]);

  return { menu, loading, error, refetch: fetchMenu };
};

// Components
const FileUpload = ({ onFileSelect, disabled }) => {
  const [dragOver, setDragOver] = useState(false);

  const handleDragOver = (e) => {
    e.preventDefault();
    setDragOver(true);
  };

  const handleDragLeave = (e) => {
    e.preventDefault();
    setDragOver(false);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setDragOver(false);
    
    const files = Array.from(e.dataTransfer.files);
    const imageFile = files.find(file => file.type.startsWith('image/'));
    
    if (imageFile) {
      onFileSelect(imageFile);
    }
  };

  const handleFileInput = (e) => {
    const file = e.target.files[0];
    if (file) {
      onFileSelect(file);
    }
  };

  return (
    <div className="upload-container">
      <div
        className={`upload-zone ${dragOver ? 'drag-over' : ''} ${disabled ? 'disabled' : ''}`}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
      >
        <div className="upload-content">
          <svg className="upload-icon" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
          </svg>
          <h3>Upload Menu Image</h3>
          <p>Drag and drop your menu image here, or click to select</p>
          <p className="file-info">Supports JPG, PNG (max 10MB)</p>
        </div>
        <input
          type="file"
          accept="image/*"
          onChange={handleFileInput}
          disabled={disabled}
          className="file-input"
        />
      </div>
    </div>
  );
};

const LoadingSpinner = ({ message = "Processing..." }) => (
  <div className="loading-container">
    <div className="spinner"></div>
    <p className="loading-message">{message}</p>
  </div>
);

const StatusIndicator = ({ status }) => {
  const getStatusInfo = (status) => {
    switch (status) {
      case MenuStatus.PENDING:
        return { text: 'Queued', color: '#fbbf24', icon: '⏳' };
      case MenuStatus.PROCESSING:
        return { text: 'Processing', color: '#3b82f6', icon: '⚡' };
      case MenuStatus.COMPLETE:
        return { text: 'Complete', color: '#10b981', icon: '✅' };
      case MenuStatus.FAILED:
        return { text: 'Failed', color: '#ef4444', icon: '❌' };
      default:
        return { text: 'Unknown', color: '#6b7280', icon: '❓' };
    }
  };

  const { text, color, icon } = getStatusInfo(status);

  return (
    <div className="status-indicator" style={{ color }}>
      <span className="status-icon">{icon}</span>
      <span className="status-text">{text}</span>
    </div>
  );
};

const DishCard = ({ dish }) => (
  <div className="dish-card">
    {dish.image && (
      <div className="dish-image-container">
        <img
          src={dish.image.url}
          alt={dish.name}
          className="dish-image"
          loading="lazy"
        />
      </div>
    )}
    <div className="dish-content">
      <div className="dish-header">
        <h4 className="dish-name">{dish.name}</h4>
        {dish.price && <span className="dish-price">{dish.price}</span>}
      </div>
      {dish.description && (
        <p className="dish-description">{dish.description}</p>
      )}
      {dish.tags && dish.tags.length > 0 && (
        <div className="dish-tags">
          {dish.tags.map((tag, index) => (
            <span key={index} className="dish-tag">
              {tag}
            </span>
          ))}
        </div>
      )}
    </div>
  </div>
);

const MenuSection = ({ section }) => (
  <div className="menu-section">
    <h3 className="section-title">{section.name}</h3>
    <div className="dishes-grid">
      {section.dishes.map((dish) => (
        <DishCard key={dish.id} dish={dish} />
      ))}
    </div>
  </div>
);

const VirtualMenu = ({ menu }) => {
  if (!menu || !menu.sections) {
    return null;
  }

  return (
    <div className="virtual-menu">
      <div className="menu-header">
        <h2>Virtual Menu</h2>
        <StatusIndicator status={menu.status} />
        {menu.generatedAt && (
          <p className="generated-time">
            Generated: {new Date(menu.generatedAt).toLocaleString()}
          </p>
        )}
      </div>
      
      <div className="menu-content">
        {menu.sections.map((section, index) => (
          <MenuSection key={index} section={section} />
        ))}
      </div>
    </div>
  );
};

const ErrorMessage = ({ error, onRetry }) => (
  <div className="error-container">
    <div className="error-content">
      <svg className="error-icon" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
      </svg>
      <h3>Something went wrong</h3>
      <p>{error}</p>
      <button onClick={onRetry} className="retry-button">
        Try Again
      </button>
    </div>
  </div>
);

// Main App Component
const App = () => {
  const [selectedFile, setSelectedFile] = useState(null);
  const [uploading, setUploading] = useState(false);
  const [menuId, setMenuId] = useState(null);
  const [currentStatus, setCurrentStatus] = useState(null);
  const [uploadError, setUploadError] = useState(null);

  const { menu, loading: pollingLoading, error: pollingError, refetch } = usePolling(menuId, currentStatus);

  // Update current status when menu changes
  useEffect(() => {
    if (menu) {
      setCurrentStatus(menu.status);
    }
  }, [menu]);

  const handleFileSelect = (file) => {
    setSelectedFile(file);
    setUploadError(null);
  };

  const handleUpload = async () => {
    if (!selectedFile) return;

    setUploading(true);
    setUploadError(null);

    try {
      const formData = new FormData();
      formData.append('image', selectedFile);

      const response = await fetch(`${API_BASE_URL}/menu`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`Upload failed: ${errorText}`);
      }

      const result = await response.json();
      setMenuId(result.menuId);
      setCurrentStatus(result.status);
      setSelectedFile(null);
    } catch (error) {
      setUploadError(error.message);
    } finally {
      setUploading(false);
    }
  };

  const handleRetry = () => {
    setMenuId(null);
    setCurrentStatus(null);
    setSelectedFile(null);
    setUploadError(null);
  };

  const isProcessing = currentStatus === MenuStatus.PENDING || currentStatus === MenuStatus.PROCESSING;
  const isComplete = currentStatus === MenuStatus.COMPLETE;
  const isFailed = currentStatus === MenuStatus.FAILED;

  return (
    <div className="app">
      <header className="app-header">
        <h1>Menu Generator</h1>
        <p>Transform your restaurant menu photos into interactive digital menus</p>
      </header>

      <main className="app-main">
        {/* Upload Section */}
        {!menuId && (
          <div className="upload-section">
            <FileUpload
              onFileSelect={handleFileSelect}
              disabled={uploading}
            />
            
            {selectedFile && (
              <div className="file-preview">
                <div className="file-info">
                  <span className="file-name">{selectedFile.name}</span>
                  <span className="file-size">
                    {(selectedFile.size / 1024 / 1024).toFixed(2)} MB
                  </span>
                </div>
                <button
                  onClick={handleUpload}
                  disabled={uploading}
                  className="upload-button"
                >
                  {uploading ? 'Uploading...' : 'Process Menu'}
                </button>
              </div>
            )}

            {uploadError && (
              <ErrorMessage error={uploadError} onRetry={handleRetry} />
            )}
          </div>
        )}

        {/* Processing Section */}
        {isProcessing && (
          <div className="processing-section">
            <LoadingSpinner
              message={
                currentStatus === MenuStatus.PENDING
                  ? 'Preparing to process your menu...'
                  : 'Extracting menu items and generating images...'
              }
            />
            <StatusIndicator status={currentStatus} />
            <p className="processing-info">
              This usually takes 20-40 seconds. We're extracting dishes and generating appetizing images for each item.
            </p>
          </div>
        )}

        {/* Results Section */}
        {isComplete && menu && (
          <VirtualMenu menu={menu} />
        )}

        {/* Error Section */}
        {isFailed && (
          <ErrorMessage
            error={menu?.error || 'Menu processing failed'}
            onRetry={handleRetry}
          />
        )}

        {/* Polling Error */}
        {pollingError && !isFailed && (
          <div className="polling-error">
            <p>Connection issue while checking status. <button onClick={refetch} className="link-button">Retry</button></p>
          </div>
        )}

        {/* New Menu Button */}
        {(isComplete || isFailed) && (
          <div className="action-section">
            <button onClick={handleRetry} className="new-menu-button">
              Process Another Menu
            </button>
          </div>
        )}
      </main>

      <footer className="app-footer">
        <p>Powered by AI • Extract menu data and generate dish images automatically</p>
      </footer>
    </div>
  );
};

export default App;
