import React, { useState, useRef } from 'react';
import { FiUpload, FiX, FiFile, FiImage } from 'react-icons/fi';
import './FileUpload.css';

const FileUpload = ({ setFile, file, setError, allowedFileTypes = ['.pdf', '.docx', '.doc', '.txt', '.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif'] }) => {
  const [dragActive, setDragActive] = useState(false);
  const [imageWarning, setImageWarning] = useState(false);
  const inputRef = useRef(null);

  // Function to check if file type is allowed
  const isFileTypeAllowed = (fileName) => {
    const extension = '.' + fileName.split('.').pop().toLowerCase();
    return allowedFileTypes.includes(extension);
  };

  // Function to check if file is an image
  const isImageFile = (fileName) => {
    const extension = '.' + fileName.split('.').pop().toLowerCase();
    const imageExtensions = ['.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif'];
    return imageExtensions.includes(extension);
  };

  // Handle file selection
  const handleFile = (files) => {
    if (files && files[0]) {
      const selectedFile = files[0];
      
      if (!isFileTypeAllowed(selectedFile.name)) {
        setError(`File type not supported. Please upload ${allowedFileTypes.join(', ')} files.`);
        return;
      }
      
      // Check if file is an image and show warning
      if (isImageFile(selectedFile.name)) {
        setImageWarning(true);
      } else {
        setImageWarning(false);
      }
      
      setFile(selectedFile);
      setError(null);
    }
  };

  // Handle drag events
  const handleDrag = (e) => {
    e.preventDefault();
    e.stopPropagation();
    
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true);
    } else if (e.type === 'dragleave') {
      setDragActive(false);
    }
  };

  // Handle drop event
  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    
    handleFile(e.dataTransfer.files);
  };

  // Handle button click
  const handleButtonClick = () => {
    inputRef.current.click();
  };

  // Handle file input change
  const handleChange = (e) => {
    handleFile(e.target.files);
  };

  // Handle file removal
  const handleRemoveFile = () => {
    setFile(null);
    setImageWarning(false);
    if (inputRef.current) {
      inputRef.current.value = '';
    }
  };

  return (
    <div className="file-upload-container">
      {!file ? (
        <div 
          className={`upload-area ${dragActive ? 'drag-active' : ''}`}
          onDragEnter={handleDrag}
          onDragOver={handleDrag}
          onDragLeave={handleDrag}
          onDrop={handleDrop}
        >
          <input
            ref={inputRef}
            type="file"
            id="file-upload"
            className="file-input"
            onChange={handleChange}
            accept={allowedFileTypes.join(',')}
          />
          <div className="upload-content">
            <FiUpload className="upload-icon" />
            <p>Drag & Drop your resume here or</p>
            <button 
              className="browse-button"
              onClick={handleButtonClick}
            >
              Browse Files
            </button>
            <p className="file-types">
              Supported formats: {allowedFileTypes.join(', ')}
            </p>
          </div>
        </div>
      ) : (
        <div className="file-preview">
          {isImageFile(file.name) ? <FiImage className="file-icon" /> : <FiFile className="file-icon" />}
          <div className="file-info">
            <p className="file-name">{file.name}</p>
            <p className="file-size">{(file.size / 1024).toFixed(2)} KB</p>
          </div>
          <button 
            className="remove-button"
            onClick={handleRemoveFile}
          >
            <FiX />
          </button>
        </div>
      )}
      
      {imageWarning && (
        <div className="image-warning">
          <p><strong>Note:</strong> Image files require OCR (Optical Character Recognition) to be properly analyzed. 
          If you encounter issues, please convert your resume to PDF or DOCX format for better results.</p>
        </div>
      )}
    </div>
  );
};

export default FileUpload; 