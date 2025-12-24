import React, { useState, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { useDropzone } from 'react-dropzone';
import axios from 'axios';
import {
  Upload,
  AlertCircle,
  CheckCircle,
  Loader2,
  Info,
  AlertTriangle,
  Shield,
  Eye,
  Sparkles
} from 'lucide-react';
import './App.css';

const API_URL = 'http://localhost:8000';

function App() {
  const [image, setImage] = useState(null);
  const [preview, setPreview] = useState(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  const onDrop = useCallback((acceptedFiles) => {
    const file = acceptedFiles[0];
    if (file) {
      setImage(file);
      setPreview(URL.createObjectURL(file));
      setResult(null);
      setError(null);
    }
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'image/*': ['.jpeg', '.jpg', '.png']
    },
    multiple: false
  });

  const handleAnalyze = async () => {
    if (!image) return;

    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const formData = new FormData();
      formData.append('file', image);

      const response = await axios.post(`${API_URL}/predict`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data'
        }
      });

      setResult(response.data);
    } catch (err) {
      console.error('Error:', err);
      setError(err.response?.data?.detail || 'Failed to analyze image. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  const handleReset = () => {
    setImage(null);
    setPreview(null);
    setResult(null);
    setError(null);
  };

  return (
    <div className="app">
      {/* Animated background particles */}
      <div className="particles">
        {[...Array(20)].map((_, i) => (
          <motion.div
            key={i}
            className="particle"
            animate={{
              y: [0, -1000],
              opacity: [0, 1, 0]
            }}
            transition={{
              duration: Math.random() * 10 + 10,
              repeat: Infinity,
              delay: Math.random() * 5
            }}
            style={{
              left: `${Math.random() * 100}%`,
              width: `${Math.random() * 4 + 2}px`,
              height: `${Math.random() * 4 + 2}px`
            }}
          />
        ))}
      </div>

      <div className="container">
        {/* Header */}
        <motion.div
          className="header"
          initial={{ opacity: 0, y: -50 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8 }}
        >
          <div className="logo">
            <Sparkles className="logo-icon" />
            <h1>Melanoma Detection AI</h1>
          </div>
          <p className="subtitle">
            Advanced AI-powered skin cancer detection with detailed analysis
          </p>
        </motion.div>

        {/* Main Content */}
        <motion.div
          className="main-content"
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ duration: 0.8, delay: 0.2 }}
        >
          {/* Upload Section */}
          {!result && (
            <motion.div
              className="upload-section"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
            >
              <div
                {...getRootProps()}
                className={`dropzone ${isDragActive ? 'active' : ''} ${preview ? 'has-image' : ''}`}
              >
                <input {...getInputProps()} />
                
                {!preview ? (
                  <motion.div
                    className="dropzone-content"
                    whileHover={{ scale: 1.05 }}
                    whileTap={{ scale: 0.95 }}
                  >
                    <Upload className="upload-icon" />
                    <h3>Drop your image here</h3>
                    <p>or click to browse</p>
                    <span className="file-types">Supports: JPG, JPEG, PNG</span>
                  </motion.div>
                ) : (
                  <motion.div
                    className="preview-container"
                    initial={{ opacity: 0, scale: 0.8 }}
                    animate={{ opacity: 1, scale: 1 }}
                  >
                    <img src={preview} alt="Preview" className="preview-image" />
                    <motion.div
                      className="preview-overlay"
                      whileHover={{ opacity: 1 }}
                    >
                      <p>Click to change image</p>
                    </motion.div>
                  </motion.div>
                )}
              </div>

              {preview && !loading && (
                <motion.div
                  className="action-buttons"
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: 0.3 }}
                >
                  <motion.button
                    className="btn btn-primary"
                    onClick={handleAnalyze}
                    whileHover={{ scale: 1.05 }}
                    whileTap={{ scale: 0.95 }}
                  >
                    <Eye className="btn-icon" />
                    Analyze Image
                  </motion.button>
                  <motion.button
                    className="btn btn-secondary"
                    onClick={handleReset}
                    whileHover={{ scale: 1.05 }}
                    whileTap={{ scale: 0.95 }}
                  >
                    Reset
                  </motion.button>
                </motion.div>
              )}

              {loading && (
                <motion.div
                  className="loading"
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                >
                  <Loader2 className="loading-icon" />
                  <h3>Analyzing your image...</h3>
                  <p>Our AI is carefully examining the skin features</p>
                  <motion.div
                    className="progress-bar"
                    initial={{ width: 0 }}
                    animate={{ width: '100%' }}
                    transition={{ duration: 2, repeat: Infinity }}
                  />
                </motion.div>
              )}

              {error && (
                <motion.div
                  className="error-message"
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                >
                  <AlertCircle className="error-icon" />
                  <p>{error}</p>
                </motion.div>
              )}
            </motion.div>
          )}

          {/* Results Section */}
          <AnimatePresence>
            {result && (
              <motion.div
                className="results-section"
                initial={{ opacity: 0, y: 50 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: 50 }}
                transition={{ duration: 0.6 }}
              >
                {/* Result Header */}
                <motion.div
                  className={`result-header ${result.prediction === 'melanoma' ? 'danger' : 'success'}`}
                  initial={{ opacity: 0, scale: 0.8 }}
                  animate={{ opacity: 1, scale: 1 }}
                  transition={{ delay: 0.2 }}
                >
                  {result.prediction === 'melanoma' ? (
                    <AlertTriangle className="result-icon" />
                  ) : (
                    <CheckCircle className="result-icon" />
                  )}
                  <div>
                    <h2>{result.explanation.description}</h2>
                    <p className="confidence">
                      Confidence: {(result.confidence * 100).toFixed(2)}%
                      <span className="confidence-label">
                        ({result.explanation.confidence_level})
                      </span>
                    </p>
                  </div>
                </motion.div>

                {/* Images Comparison */}
                <motion.div
                  className="images-grid"
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  transition={{ delay: 0.4 }}
                >
                  <div className="image-card">
                    <h3>Original Image</h3>
                    <img src={result.images.original} alt="Original" />
                  </div>
                  <div className="image-card">
                    <h3>AI Analysis Heatmap</h3>
                    <img src={result.images.heatmap} alt="Heatmap" />
                    <p className="image-caption">
                      Highlighted areas show regions the AI focused on
                    </p>
                  </div>
                </motion.div>

                {/* Detailed Explanation */}
                <motion.div
                  className="explanation-card"
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: 0.6 }}
                >
                  <h3>
                    <Info className="section-icon" />
                    Detailed Analysis
                  </h3>
                  <ul className="details-list">
                    {result.explanation.details.map((detail, index) => (
                      <motion.li
                        key={index}
                        initial={{ opacity: 0, x: -20 }}
                        animate={{ opacity: 1, x: 0 }}
                        transition={{ delay: 0.7 + index * 0.1 }}
                      >
                        {detail}
                      </motion.li>
                    ))}
                  </ul>
                </motion.div>

                {/* ABCD Criteria */}
                <motion.div
                  className="abcd-card"
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: 0.8 }}
                >
                  <h3>
                    <Shield className="section-icon" />
                    ABCDE Criteria Assessment
                  </h3>
                  <div className="abcd-grid">
                    {Object.entries(result.explanation.abcd_criteria).map(([key, value], index) => (
                      <motion.div
                        key={key}
                        className="abcd-item"
                        initial={{ opacity: 0, scale: 0.8 }}
                        animate={{ opacity: 1, scale: 1 }}
                        transition={{ delay: 0.9 + index * 0.1 }}
                      >
                        <div className="abcd-letter">{key.split('_')[1][0].toUpperCase()}</div>
                        <div className="abcd-content">
                          <h4>{key.split('_')[1].charAt(0).toUpperCase() + key.split('_')[1].slice(1)}</h4>
                          <p>{value}</p>
                        </div>
                      </motion.div>
                    ))}
                  </div>
                </motion.div>

                {/* Recommendation */}
                <motion.div
                  className={`recommendation-card ${result.prediction === 'melanoma' ? 'urgent' : 'normal'}`}
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: 1.2 }}
                >
                  <AlertCircle className="recommendation-icon" />
                  <div>
                    <h3>Medical Recommendation</h3>
                    <p>{result.explanation.recommendation}</p>
                  </div>
                </motion.div>

                {/* Warning */}
                <motion.div
                  className="warning-card"
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  transition={{ delay: 1.4 }}
                >
                  <AlertTriangle className="warning-icon" />
                  <p>{result.warning}</p>
                </motion.div>

                {/* Action Buttons */}
                <motion.div
                  className="action-buttons"
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: 1.6 }}
                >
                  <motion.button
                    className="btn btn-primary"
                    onClick={handleReset}
                    whileHover={{ scale: 1.05 }}
                    whileTap={{ scale: 0.95 }}
                  >
                    Analyze Another Image
                  </motion.button>
                </motion.div>
              </motion.div>
            )}
          </AnimatePresence>
        </motion.div>

        {/* Footer */}
        <motion.footer
          className="footer"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 1, duration: 0.8 }}
        >
          <p>
            <Shield className="footer-icon" />
            This tool is for screening purposes only and does not replace professional medical diagnosis
          </p>
        </motion.footer>
      </div>
    </div>
  );
}

export default App;



