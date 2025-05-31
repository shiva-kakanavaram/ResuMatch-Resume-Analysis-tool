import React, { useState } from 'react';
import { BrowserRouter as Router, Route, Routes } from 'react-router-dom';
import axios from 'axios';
import './App.css';
import LandingPage from './components/LandingPage';
import NotFound from './components/NotFound';
import LoadingSpinner from './components/LoadingSpinner';

// Utility functions
const getScoreClass = (score) => {
  if (score >= 80) return 'score-high';
  if (score >= 60) return 'score-medium';
  return 'score-low';
};

const formatSectionName = (section) => {
  // Convert section key to readable format (e.g., "work_experience" to "Work Experience")
  return section
    .split('_')
    .map(word => word.charAt(0).toUpperCase() + word.slice(1))
    .join(' ');
};

// Main application component with routes
function App() {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<LandingPage />} />
        <Route path="/app" element={<ResuMatchApp />} />
        <Route path="*" element={<NotFound />} />
      </Routes>
    </Router>
  );
}

// The actual application component
function ResuMatchApp() {
  const [resumeFile, setResumeFile] = useState(null);
  const [resumeFiles, setResumeFiles] = useState([]);
  const [jobDescription, setJobDescription] = useState('');
  const [jobTitle, setJobTitle] = useState('');
  const [company, setCompany] = useState('');
  const [userMode, setUserMode] = useState('jobseeker'); // 'jobseeker' or 'hr'
  const [result, setResult] = useState(null);
  const [bulkResults, setBulkResults] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [uploadError, setUploadError] = useState(null);
  
  // Track if we want to show job matching in the jobseeker view (default true)
  const [showJobMatching, setShowJobMatching] = useState(true);

  const validateFile = (file) => {
    const validTypes = ['application/pdf', 'application/vnd.openxmlformats-officedocument.wordprocessingml.document'];
    if (!validTypes.includes(file.type)) {
      return 'Please upload a PDF or DOCX file';
    }
    if (file.size > 10 * 1024 * 1024) { // 10MB limit
      return 'File size must be less than 10MB';
    }
    return null;
  };

  const handleFileChange = (event) => {
    const file = event.target.files[0];
    setUploadError(null);
    setError(null);
    setResult(null);
    
    if (file) {
      const error = validateFile(file);
      if (error) {
        setUploadError(error);
        event.target.value = null; // Reset file input
        return;
      }
      setResumeFile(file);
    }
  };

  const handleBulkFileChange = (event) => {
    const files = Array.from(event.target.files);
    setUploadError(null);
    setError(null);
    setBulkResults(null);
    
    if (files.length === 0) return;
    
    // Validate each file
    const invalidFiles = files.map(file => {
      const error = validateFile(file);
      return error ? { file: file.name, error } : null;
    }).filter(Boolean);
    
    if (invalidFiles.length > 0) {
      setUploadError(`${invalidFiles.length} file(s) are invalid: ${invalidFiles.map(f => f.file).join(', ')}`);
      event.target.value = null; // Reset file input
      return;
    }
    
    setResumeFiles(files);
  };

  const handleUserModeChange = (mode) => {
    setUserMode(mode);
    // Reset all form data when switching modes
    setError(null);
    setResult(null);
    setBulkResults(null);
    setResumeFile(null);
    setResumeFiles([]);
    setJobDescription('');
    setJobTitle('');
    setCompany('');
    setUploadError(null);
    
    // Reset file input elements
    const fileInputs = document.querySelectorAll('input[type="file"]');
    fileInputs.forEach(input => {
      input.value = '';
    });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setResult(null);

    if (userMode === 'jobseeker') {
      if (!resumeFile) {
        setError('Please select a resume file');
        setLoading(false);
        return;
      }

      // If job matching is enabled, require job description
      if (showJobMatching && !jobDescription.trim()) {
        setError('Please enter a job description for job matching');
        setLoading(false);
        return;
      }

      const formData = new FormData();
      formData.append('resume', resumeFile);
      formData.append('job_description', jobDescription);
      formData.append('job_title', jobTitle);
      formData.append('company', company);
      formData.append('analysis_type', showJobMatching ? 'match' : 'ats');
      
      try {
        const response = await axios.post('http://localhost:8000/analyze', formData, {
          headers: {
            'Content-Type': 'multipart/form-data',
          },
        });
        
        if (response.data) {
          setResult(response.data);
        } else {
          throw new Error('No data received from server');
        }
      } catch (err) {
        console.error('Error:', err);
        const errorMessage = err.response?.data?.detail || err.message || 'An error occurred during analysis';
        setError(errorMessage);
      } finally {
        setLoading(false);
      }
    } else if (userMode === 'hr') {
      // HR/Organization bulk analysis
      if (resumeFiles.length === 0) {
        setError('Please select resume files');
        setLoading(false);
        return;
      }

      if (!jobDescription.trim()) {
        setError('Please enter a job description');
        setLoading(false);
        return;
      }

      try {
        // Create a form data with multiple files
        const formData = new FormData();
        resumeFiles.forEach((file, index) => {
          formData.append(`resumes`, file);
        });
        formData.append('job_description', jobDescription);
        formData.append('job_title', jobTitle);
        formData.append('company', company);
        
        let response;
        
        console.log('Starting HR mode resume analysis...');
        
        try {
          // Try the bulk_analyze endpoint first
          console.log('Trying bulk_analyze endpoint...');
          response = await axios.post('http://localhost:8000/bulk_analyze', formData, {
            headers: {
              'Content-Type': 'multipart/form-data',
            },
          });
          console.log('Bulk analyze successful:', response.data);
        } catch (bulkError) {
          console.log('Bulk analyze failed, trying compare_resumes endpoint as fallback...', bulkError);
          
          try {
            // If bulk_analyze fails, try the compare_resumes endpoint as fallback
            response = await axios.post('http://localhost:8000/compare_resumes', formData, {
              headers: {
                'Content-Type': 'multipart/form-data',
              },
            });
            console.log('Compare resumes successful:', response.data);
          } catch (compareError) {
            console.log('Compare resumes also failed:', compareError);
            throw new Error(`Failed to process resumes: ${compareError.response?.data?.detail || compareError.message}`);
          }
        }
        
        if (response && response.data) {
          console.log('Processing API response:', response.data);
          
          // Check if the response contains an error message
          if (response.data.error) {
            console.error('API returned error:', response.data.error);
            setError(response.data.error);
          } else if (response.data.message && !response.data.candidates) {
            console.error('API returned message without candidates:', response.data.message);
            setError(response.data.message);
          } else {
            // Ensure candidates is properly formatted
            if (response.data.candidates && Array.isArray(response.data.candidates)) {
              console.log(`Setting bulkResults with ${response.data.candidates.length} candidates`);
              setBulkResults(response.data);
            } else {
              console.error('API response missing candidates array:', response.data);
              throw new Error('Invalid response format: candidates not found');
            }
          }
        } else {
          console.error('No data received from server');
          throw new Error('No data received from server');
        }
      } catch (err) {
        console.error('Error in HR mode:', err);
        const errorMessage = err.response?.data?.detail || err.message || 'An error occurred during bulk analysis';
        setError(errorMessage);
        
        // Set a minimal result to show something meaningful to the user
        setBulkResults({
          error: errorMessage,
          candidates: [],
          message: 'Unable to complete the analysis. Please check your resumes and job description and try again.'
        });
      } finally {
        setLoading(false);
      }
    }
  };

  const handleJobMatch = async () => {
    if (!result || !result.resume_analysis || !jobDescription.trim()) {
      setError('Resume analysis and job description are required for job matching');
      return;
    }
    
    setLoading(true);
    setError(null);
    
    try {
      const response = await axios.post('http://localhost:8000/job_match', {
        resume_analysis: result.resume_analysis,
        job_description: jobDescription,
        use_enhanced_matching: true
      });
      
      if (response.data) {
        // Update the existing result with job match data
        setResult({
          ...result,
          job_match_analysis: response.data,
          overall_score: response.data.match_score
        });
      } else {
        throw new Error('No data received from server');
      }
    } catch (err) {
      console.error('Error in job matching:', err);
      const errorMessage = err.response?.data?.detail || err.message || 'An error occurred during job matching';
      setError(errorMessage);
    } finally {
      setLoading(false);
    }
  };

  const renderSkillAnalysis = (skillAnalysis) => {
    if (!skillAnalysis) return null;
    
    return Object.entries(skillAnalysis).map(([category, data]) => (
      <div key={category} className="skill-category">
        <h4>{category.replace(/_/g, ' ').toUpperCase()}</h4>
        <div className="skill-details">
          <div className="matched-skills">
            <strong>Matched Skills:</strong>
            <ul>
              {data.matched.map((skill, index) => (
                <li key={`${skill}-${index}`}>{skill}</li>
              ))}
            </ul>
          </div>
          <div className="missing-skills">
            <strong>Missing Skills:</strong>
            <ul>
              {data.missing.map((skill, index) => (
                <li key={`${skill}-${index}`}>{skill}</li>
              ))}
            </ul>
          </div>
          <div className="match-percentage">
            Match: {(data.match_percentage * 100).toFixed(1)}%
          </div>
        </div>
      </div>
    ));
  };

  const renderBulkResults = () => {
    if (!bulkResults) return null;
    
    console.log("Rendering bulk results:", bulkResults);
    
    // Check if we have an error message in the response
    if (bulkResults.error || (bulkResults.message && !bulkResults.candidates)) {
      return (
        <div className="bulk-results error">
          <h3>Top Candidates Analysis</h3>
          <div className="error-message">
            {bulkResults.error || bulkResults.message}
          </div>
        </div>
      );
    }
    
    // Check if we have candidates
    if (!bulkResults.candidates || bulkResults.candidates.length === 0) {
      console.log("No candidates found in bulk results");
      return (
        <div className="bulk-results">
          <h3>Top Matching Candidates</h3>
          <div className="no-results">
            <p>No matching candidates found. Try adjusting your job description or uploading more resumes.</p>
          </div>
        </div>
      );
    }
    
    console.log("Found candidates:", bulkResults.candidates.length);
    console.log("First candidate:", bulkResults.candidates[0]);
    
    return (
      <div className="bulk-results">
        <h3>Top Matching Candidates ({bulkResults.candidates.length})</h3>
        <div className="candidates-list">
          {bulkResults.candidates.map((candidate, index) => {
            // Ensure we have valid data for all fields
            const name = candidate.name || `Candidate ${index + 1}`;
            const filename = candidate.filename || `resume_${index}.pdf`;
            const matchScore = typeof candidate.match_score === 'number' ? candidate.match_score : 0;
            const matchLevel = candidate.match_level || 'Low Match';
            const skills = Array.isArray(candidate.top_skills) ? candidate.top_skills : ['No skills detected'];
            
            return (
              <div key={index} className="candidate-card">
                <div className="candidate-header">
                  <h4 className="candidate-name">
                    {name !== "Unknown" ? name : `Candidate ${index + 1} (${filename.split('.')[0]})`}
                  </h4>
                  <div className="candidate-score">{matchScore.toFixed(1)}%</div>
                </div>
                <div className="candidate-details">
                  <div className="candidate-match-level">{matchLevel}</div>
                  <div className="candidate-skills">
                    <strong>Top Skills:</strong>
                    {skills.length > 0 ? (
                      <ul>
                        {skills.map((skill, idx) => (
                          <li key={idx}>{skill}</li>
                        ))}
                      </ul>
                    ) : (
                      <p className="no-skills">No skills matched with job requirements</p>
                    )}
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      </div>
    );
  };

  const ATSAnalysisComponent = ({ analysis }) => { // eslint-disable-line no-unused-vars
    const { ats_score, section_scores, industry, suggestions } = analysis;
    
    return (
      <div className="ats-analysis">
        <h3>ATS Score Analysis</h3>
        <div className="score-container">
          <div className="main-score">
            <h4>Overall ATS Score</h4>
            <div className="score-circle">
              {Math.round(ats_score * 100)}%
            </div>
          </div>
          <div className="industry-badge">
            <span>Industry: {industry.charAt(0).toUpperCase() + industry.slice(1)}</span>
          </div>
        </div>
        
        <div className="section-scores">
          <h4>Section Scores</h4>
          <div className="score-grid">
            <div className="score-item">
              <span className="label">Skills</span>
              <span className="value">{Math.round(section_scores.skills * 100)}%</span>
            </div>
            <div className="score-item">
              <span className="label">Experience</span>
              <span className="value">{Math.round(section_scores.experience * 100)}%</span>
            </div>
            <div className="score-item">
              <span className="label">Education</span>
              <span className="value">{Math.round(section_scores.education * 100)}%</span>
            </div>
            <div className="score-item">
              <span className="label">Readability</span>
              <span className="value">{Math.round(section_scores.readability * 100)}%</span>
            </div>
            <div className="score-item">
              <span className="label">Achievements</span>
              <span className="value">{Math.round(section_scores.achievements * 100)}%</span>
            </div>
            {section_scores.tfidf !== null && (
              <div className="score-item">
                <span className="label">Job Match (TF-IDF)</span>
                <span className="value">{Math.round(section_scores.tfidf * 100)}%</span>
              </div>
            )}
          </div>
        </div>
        
        <div className="suggestions">
          <h4>Improvement Suggestions</h4>
          <ul>
            {suggestions.map((suggestion, index) => (
              <li key={index}>{suggestion}</li>
            ))}
          </ul>
        </div>
      </div>
    );
  };

  const renderAnalysisResults = (data) => { // eslint-disable-line no-unused-vars
    if (!data) return null;
  
    const { resume_analysis } = data;
    const { overall_score, section_scores = {}, suggestions = [] } = resume_analysis;
    
    // Check if there are OCR-related messages in suggestions
    const hasOcrMessage = suggestions.some(
      msg => msg.includes('OCR') || msg.includes('image')
    );
    
    return (
      <div className="analysis-results">
        <div className="score-container">
          <div className="overall-score">
            <div className="score-label">Overall Score</div>
            <div className="score-ring">
              <div 
                className={`score-value ${getScoreClass(overall_score)}`}
                style={{ '--percentage': `${overall_score}%` }}
              >
                {overall_score}
              </div>
            </div>
          </div>
          
          {hasOcrMessage && (
            <div className="ocr-warning">
              <h4>Image Processing Limited</h4>
              <p>Your resume appears to be an image or scanned document. Without OCR capabilities, 
                 we cannot properly analyze all sections. For best results, please upload a text-based 
                 PDF or DOCX file.</p>
            </div>
          )}
          
          <div className="section-scores">
            <h3>Section Scores</h3>
            {Object.entries(section_scores).length > 0 ? (
              <div className="section-scores-container">
                {Object.entries(section_scores).map(([section, score]) => (
                  <div className="section-score-item" key={section}>
                    <div className="section-name">{formatSectionName(section)}</div>
                    <div className="section-score-bar">
                      <div 
                        className={`section-score-fill ${getScoreClass(score * 100)}`}
                        style={{ width: `${score * 100}%` }}
                      ></div>
                    </div>
                    <div className="section-score-value">{Math.round(score * 100)}</div>
                  </div>
                ))}
              </div>
            ) : (
              <p className="no-sections">No section scores available.</p>
            )}
          </div>
        </div>
        
        <div className="suggestions">
          <h3>Suggestions</h3>
          {suggestions.length > 0 ? (
            <ul className="suggestions-list">
              {suggestions.map((suggestion, index) => (
                <li key={index} className="suggestion-item">
                  {suggestion}
                </li>
              ))}
            </ul>
          ) : (
            <p className="no-suggestions">No suggestions available.</p>
          )}
        </div>
      </div>
    );
  };

  return (
    <div className="App">
      <header className="App-header">
        <div className="header-content">
          <div className="logo">
            <span className="logo-text">ResuMatch</span>
          </div>
          <p>AI-Powered Resume Analysis</p>
        </div>
      </header>

      <div className="user-mode-selector">
        <button 
          className={`mode-btn ${userMode === 'jobseeker' ? 'active' : ''}`}
          onClick={() => handleUserModeChange('jobseeker')}
        >
          Job Seeker Mode
        </button>
        <button 
          className={`mode-btn ${userMode === 'hr' ? 'active' : ''}`}
          onClick={() => handleUserModeChange('hr')}
        >
          HR/Organization Mode
        </button>
      </div>

      <main className="main-content">
        <div className="upload-container">
          <form onSubmit={handleSubmit}>
            <div className="upload-section">
              <h3>{userMode === 'jobseeker' ? 'Upload Your Resume' : 'Upload Resumes in Bulk'}</h3>
              <p>Supported formats: PDF, DOCX</p>
              {userMode === 'jobseeker' ? (
                <input
                  type="file"
                  onChange={handleFileChange}
                  accept=".pdf,.docx"
                  className={uploadError ? 'error' : ''}
                />
              ) : (
                <input
                  type="file"
                  onChange={handleBulkFileChange}
                  accept=".pdf,.docx"
                  multiple
                  className={uploadError ? 'error' : ''}
                />
              )}
              {uploadError && <div className="error-message">{uploadError}</div>}
              {userMode === 'hr' && resumeFiles.length > 0 && (
                <div className="file-list">
                  <p>{resumeFiles.length} file(s) selected</p>
                </div>
              )}
            </div>

            <div className="job-details-section">
              <div className="section-header-with-toggle">
                <h3>Job Details</h3>
                {userMode === 'jobseeker' && (
                  <div className="toggle-container">
                    <label className="toggle">
                      <input 
                        type="checkbox" 
                        checked={showJobMatching} 
                        onChange={() => setShowJobMatching(!showJobMatching)}
                      />
                      <span className="toggle-slider"></span>
                    </label>
                    <span className="toggle-label">Enable Job Matching</span>
                  </div>
                )}
              </div>
              
              <div className="form-group">
                <label>Job Title:</label>
                <input
                  type="text"
                  value={jobTitle}
                  onChange={(e) => setJobTitle(e.target.value)}
                  className="text-input"
                  placeholder="e.g. Software Engineer"
                />
              </div>
              
              <div className="form-group">
                <label>Company:</label>
                <input
                  type="text"
                  value={company}
                  onChange={(e) => setCompany(e.target.value)}
                  className="text-input"
                  placeholder="e.g. Tech Corp"
                />
              </div>
              
              <div className="form-group">
                <label>Job Description:</label>
                <textarea
                  value={jobDescription}
                  onChange={(e) => setJobDescription(e.target.value)}
                  className="text-area"
                  placeholder="Paste the job description here..."
                  rows={6}
                  required={userMode === 'hr' || showJobMatching}
                />
              </div>
            </div>

            <button 
              type="submit" 
              className="submit-btn"
              disabled={loading || 
                (userMode === 'jobseeker' && (!resumeFile || (showJobMatching && !jobDescription))) ||
                (userMode === 'hr' && (resumeFiles.length === 0 || !jobDescription))}
            >
              {loading ? 'Analyzing...' : 
                userMode === 'jobseeker' ? 'Analyze Resume' : 'Find Top Candidates'}
            </button>
          </form>
        </div>

        {loading && (
          <LoadingSpinner message={`Analyzing ${userMode === 'hr' ? 'resumes' : 'your resume'}...`} />
        )}

        {error && (
          <div className="error-message">
            {error}
          </div>
        )}

        {/* Individual Resume Analysis Results */}
        {userMode === 'jobseeker' && result && (
          <div className="results-container">
            <div className="score-overview">
              <h2>Analysis Results</h2>
              <div className="score-grid">
                <div className="score-item">
                  <h3>Overall Score</h3>
                  <div className="score">
                    {showJobMatching && result.job_match_analysis ? 
                      (result.job_match_analysis.match_score !== undefined ? 
                       parseFloat(result.job_match_analysis.match_score).toFixed(1) : '0.0') : 
                      (result.overall_score !== undefined ? 
                       parseFloat(result.overall_score).toFixed(1) : '0.0')}%
                  </div>
                </div>
                {showJobMatching && result.job_match_analysis && (
                  <div className="score-item">
                    <h3>Match Level</h3>
                    <div className="category">{result.job_match_analysis.match_level}</div>
                  </div>
                )}
                {result.resume_analysis?.details?.predicted_category && (
                  <div className="score-item">
                    <h3>Predicted Job Category</h3>
                    <div className="category">{result.resume_analysis.details.predicted_category.replace(/_/g, ' ')}</div>
                  </div>
                )}
              </div>
            </div>

            {/* If we have resume analysis but no job match yet, show direct match button */}
            {!showJobMatching && jobDescription.trim() && !result.job_match_analysis && (
              <div className="action-buttons">
                <button 
                  type="button" 
                  className="match-btn" 
                  onClick={handleJobMatch}
                  disabled={loading}
                >
                  {loading ? 'Matching...' : 'Match with Job Description'}
                </button>
              </div>
            )}

            <div className="detailed-analysis">
              {/* ATS Score Breakdown */}
              {result.resume_analysis?.details?.section_scores && (
                <div className="section">
                  <h3>ATS Score Breakdown</h3>
                  <div className="score-grid">
                    {Object.entries(result.resume_analysis.details.section_scores).map(([key, value]) => (
                      <div key={key} className="score-item">
                        <h4>{key.replace(/_/g, ' ').toUpperCase()}</h4>
                        <div className="sub-score">
                          {value !== undefined ? 
                           (typeof value === 'number' ? 
                            value.toFixed(1) : parseFloat(value).toFixed(1)) : '0.0'}%
                        </div>
                      </div>
                    ))}
                    {result.resume_analysis?.details?.readability?.overall_score && (
                      <div className="score-item">
                        <h4>READABILITY</h4>
                        <div className="sub-score">
                          {(parseFloat(result.resume_analysis.details.readability.overall_score) * 100).toFixed(1)}%
                        </div>
                      </div>
                    )}
                  </div>
                </div>
              )}

              {/* Job Match Details (if enabled) */}
              {showJobMatching && result.job_match_analysis && (
                <div className="section">
                  <h3>Job Match Details</h3>
                  <div className="match-details">
                    <div className="score-grid">
                      <div className="score-item">
                        <h4>Skill Match</h4>
                        <div className="sub-score">
                          {result.job_match_analysis.skill_match_score ? 
                           (typeof result.job_match_analysis.skill_match_score === 'number' ? 
                            result.job_match_analysis.skill_match_score.toFixed(1) : 
                            parseFloat(result.job_match_analysis.skill_match_score).toFixed(1))
                           : '0.0'}%
                        </div>
                      </div>
                      <div className="score-item">
                        <h4>Semantic Match</h4>
                        <div className="sub-score">
                          {result.job_match_analysis.semantic_match_score ? 
                           (typeof result.job_match_analysis.semantic_match_score === 'number' ? 
                            result.job_match_analysis.semantic_match_score.toFixed(1) : 
                            parseFloat(result.job_match_analysis.semantic_match_score).toFixed(1))
                           : '0.0'}%
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              )}

              {/* Skill Analysis (if job matching) */}
              {showJobMatching && result.resume_analysis?.details?.skill_analysis && (
                <div className="section">
                  <h3>Skill Analysis</h3>
                  {renderSkillAnalysis(result.resume_analysis.details.skill_analysis)}
                </div>
              )}
            </div>

            {((result.recommendations && result.recommendations.length > 0) || 
             (result.job_match_analysis && result.job_match_analysis.recommendations && 
              result.job_match_analysis.recommendations.length > 0)) && (
              <div className="section">
                <h3>Recommendations</h3>
                <ul className="recommendations-list">
                  {result.recommendations && result.recommendations.map((recommendation, index) => (
                    <li key={`ats-${index}`} className="recommendation-item">
                      {typeof recommendation === 'object' 
                        ? (recommendation.recommendation || recommendation.message || recommendation.suggestion || JSON.stringify(recommendation)) 
                        : recommendation}
                    </li>
                  ))}
                  {showJobMatching && result.job_match_analysis && result.job_match_analysis.recommendations && 
                   result.job_match_analysis.recommendations.map((recommendation, index) => (
                    <li key={`job-${index}`} className="recommendation-item">
                      {typeof recommendation === 'object' 
                        ? (recommendation.recommendation || recommendation.suggestion || recommendation.message || JSON.stringify(recommendation)) 
                        : recommendation}
                    </li>
                  ))}
                </ul>
              </div>
            )}
          </div>
        )}

        {/* Bulk Analysis Results for HR/Organization Mode */}
        {userMode === 'hr' && bulkResults && (
          <div className="results-container">
            <div className="score-overview">
              <h2>Bulk Analysis Results</h2>
              <p>Analyzed {bulkResults.total_candidates} resumes for {company || 'your company'}</p>
            </div>
            {renderBulkResults()}
          </div>
        )}
      </main>

      <footer className="App-footer">
        <p>&copy; {new Date().getFullYear()} ResuMatch. All rights reserved.</p>
      </footer>
    </div>
  );
}

export default App;
