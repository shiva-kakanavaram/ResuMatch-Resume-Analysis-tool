import React from 'react';
import { useNavigate } from 'react-router-dom';
import './LandingPage.css';

function LandingPage() {
  const navigate = useNavigate();

  const handleGetStarted = () => {
    console.log('Navigating to /app');
    navigate('/app');
  };

  return (
    <div className="landing-page">
      {/* Hero Section */}
      <section className="hero">
        <nav className="navbar">
          <div className="logo">
            <span className="logo-text">ResuMatch</span>
          </div>
          <div className="nav-links">
            <a href="#features">Features</a>
            <a href="#how-it-works">How It Works</a>
            <a href="#testimonials">Testimonials</a>
          </div>
          <button type="button" className="login-btn" onClick={handleGetStarted}>Get Started</button>
        </nav>
        
        <div className="hero-content">
          <div className="hero-text" data-aos="fade-right">
            <h1>Get Your Resume <span className="highlight">AI-Powered</span> Advantage</h1>
            <p>Optimize your resume with our AI-driven ATS analysis and job matching tools to land your dream job faster.</p>
            <div className="hero-btns">
              <button type="button" className="primary-btn" onClick={handleGetStarted}>Get Started</button>
              <button type="button" className="secondary-btn" onClick={handleGetStarted}>Watch Demo</button>
            </div>
            <div className="stats-container">
              <div className="stat-item" data-aos="fade-up" data-aos-delay="100">
                <span className="stat-number">98%</span>
                <span className="stat-label">Accuracy</span>
              </div>
              <div className="stat-item" data-aos="fade-up" data-aos-delay="200">
                <span className="stat-number">10K+</span>
                <span className="stat-label">Users</span>
              </div>
              <div className="stat-item" data-aos="fade-up" data-aos-delay="300">
                <span className="stat-number">87%</span>
                <span className="stat-label">Success Rate</span>
              </div>
            </div>
          </div>
          <div className="hero-image" data-aos="fade-left">
            <div className="resume-preview">
              <div className="resume-header">
                <div className="resume-dots">
                  <span></span>
                  <span></span>
                  <span></span>
                </div>
                <div className="resume-title">Resume Analysis</div>
              </div>
              <div className="resume-content">
                <div className="score-circle">
                  <div className="water-fill"></div>
                  <div className="bubble"></div>
                  <div className="bubble"></div>
                  <div className="bubble"></div>
                  <div className="bubble"></div>
                  <div className="bubble"></div>
                  <div className="percentage">85%</div>
                </div>
                <div className="resume-matches">
                  <div className="match-item">
                    <div className="match-label">Skills Match</div>
                    <div className="match-bar">
                      <div className="match-progress" style={{ width: '78%' }}></div>
                    </div>
                    <div className="match-percent">78%</div>
                  </div>
                  <div className="match-item">
                    <div className="match-label">Experience Match</div>
                    <div className="match-bar">
                      <div className="match-progress" style={{ width: '92%' }}></div>
                    </div>
                    <div className="match-percent">92%</div>
                  </div>
                  <div className="match-item">
                    <div className="match-label">Education Match</div>
                    <div className="match-bar">
                      <div className="match-progress" style={{ width: '85%' }}></div>
                    </div>
                    <div className="match-percent">85%</div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
        
        <div className="wave-separator">
          <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1440 320">
            <path fill="#f9f9fb" fillOpacity="1" d="M0,64L48,80C96,96,192,128,288,122.7C384,117,480,75,576,80C672,85,768,139,864,144C960,149,1056,107,1152,96C1248,85,1344,107,1392,117.3L1440,128L1440,320L1392,320C1344,320,1248,320,1152,320C1056,320,960,320,864,320C768,320,672,320,576,320C480,320,384,320,288,320C192,320,96,320,48,320L0,320Z"></path>
          </svg>
        </div>
      </section>

      {/* Features Section */}
      <section id="features" className="features">
        <div className="section-header" data-aos="fade-up">
          <h2>Powerful Features</h2>
          <p>Everything you need to optimize your resume and find the perfect job match</p>
        </div>
        
        <div className="features-grid">
          <div className="feature-card" data-aos="zoom-in" data-aos-delay="100">
            <div className="feature-icon">
              <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
            </div>
            <h3>ATS Compatibility Analysis</h3>
            <p>Ensure your resume gets past automated tracking systems with our in-depth scanning and optimization.</p>
          </div>
          
          <div className="feature-card" data-aos="zoom-in" data-aos-delay="200">
            <div className="feature-icon">
              <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17 8h2a2 2 0 012 2v6a2 2 0 01-2 2h-2v4l-4-4H9a1.994 1.994 0 01-1.414-.586m0 0L11 14h4a2 2 0 002-2V6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2v4l.586-.586z" />
              </svg>
            </div>
            <h3>Personalized Recommendations</h3>
            <p>Get tailored suggestions to improve your resume for specific job positions and industries.</p>
          </div>
          
          <div className="feature-card" data-aos="zoom-in" data-aos-delay="300">
            <div className="feature-icon">
              <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M11 3.055A9.001 9.001 0 1020.945 13H11V3.055z" />
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M20.488 9H15V3.512A9.025 9.025 0 0120.488 9z" />
              </svg>
            </div>
            <h3>Job Matching Score</h3>
            <p>See how well your resume matches specific job descriptions with detailed scoring metrics.</p>
          </div>
          
          <div className="feature-card" data-aos="zoom-in" data-aos-delay="400">
            <div className="feature-icon">
              <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 012-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10" />
              </svg>
            </div>
            <h3>Bulk Resume Processing</h3>
            <p>HR professionals can analyze multiple resumes at once to find the best candidates for the job.</p>
          </div>
        </div>
      </section>

      {/* How It Works Section */}
      <section id="how-it-works" className="how-it-works">
        <div className="wave-separator top">
          <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1440 320">
            <path fill="#4361ee" fillOpacity="0.05" d="M0,64L48,80C96,96,192,128,288,122.7C384,117,480,75,576,80C672,85,768,139,864,144C960,149,1056,107,1152,96C1248,85,1344,107,1392,117.3L1440,128L1440,320L1392,320C1344,320,1248,320,1152,320C1056,320,960,320,864,320C768,320,672,320,576,320C480,320,384,320,288,320C192,320,96,320,48,320L0,320Z"></path>
          </svg>
        </div>
        
        <div className="section-header" data-aos="fade-up">
          <h2>How It Works</h2>
          <p>Just a few simple steps to optimize your resume and boost your job search</p>
        </div>
        
        <div className="steps-container">
          <div className="step" data-aos="fade-right">
            <div className="step-number">1</div>
            <div className="step-content">
              <h3>Upload Your Resume</h3>
              <p>Simply upload your resume in PDF or DOCX format to our secure platform.</p>
            </div>
          </div>
          
          <div className="step" data-aos="fade-left">
            <div className="step-number">2</div>
            <div className="step-content">
              <h3>Enter Job Description</h3>
              <p>Paste the job description you're interested in to get a personalized match score.</p>
            </div>
          </div>
          
          <div className="step" data-aos="fade-right">
            <div className="step-number">3</div>
            <div className="step-content">
              <h3>Get Detailed Analysis</h3>
              <p>Receive comprehensive feedback on your resume's ATS compatibility and job match.</p>
            </div>
          </div>
          
          <div className="step" data-aos="fade-left">
            <div className="step-number">4</div>
            <div className="step-content">
              <h3>Implement Recommendations</h3>
              <p>Follow our AI-powered suggestions to optimize your resume for better results.</p>
            </div>
          </div>
        </div>
        
        <div className="cta-container" data-aos="zoom-in">
          <h3>Ready to supercharge your job search?</h3>
          <button type="button" className="primary-btn" onClick={handleGetStarted}>Try It For Free</button>
        </div>
      </section>

      {/* Testimonials Section */}
      <section id="testimonials" className="testimonials">
        <div className="section-header" data-aos="fade-up">
          <h2>What Our Users Say</h2>
          <p>Join thousands of job seekers who have improved their resume with ResuMatch</p>
        </div>
        
        <div className="testimonials-grid">
          <div className="testimonial-card" data-aos="fade-up" data-aos-delay="100">
            <div className="testimonial-content">
              <p>"ResuMatch helped me optimize my resume for ATS systems and I got 3 interviews in my first week of applying!"</p>
            </div>
            <div className="testimonial-author">
              <div className="author-avatar">JS</div>
              <div className="author-info">
                <h4>Jane Smith</h4>
                <p>Software Developer</p>
              </div>
            </div>
          </div>
          
          <div className="testimonial-card" data-aos="fade-up" data-aos-delay="200">
            <div className="testimonial-content">
              <p>"As a recruiter, the bulk resume analysis feature saves me hours of work every day. I can quickly find the best candidates for each position."</p>
            </div>
            <div className="testimonial-author">
              <div className="author-avatar">MJ</div>
              <div className="author-info">
                <h4>Mark Johnson</h4>
                <p>HR Manager</p>
              </div>
            </div>
          </div>
          
          <div className="testimonial-card" data-aos="fade-up" data-aos-delay="300">
            <div className="testimonial-content">
              <p>"The job matching feature is a game-changer! I can now see exactly how my skills align with job requirements and focus on improving what matters."</p>
            </div>
            <div className="testimonial-author">
              <div className="author-avatar">AL</div>
              <div className="author-info">
                <h4>Aisha Lopez</h4>
                <p>Marketing Specialist</p>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="footer">
        <div className="footer-content">
          <div className="footer-logo">
            <span className="logo-text">ResuMatch</span>
          </div>
          
          <div className="footer-links">
            <div className="footer-column">
              <h4>Company</h4>
              <ul>
                <li><a href="#about">About Us</a></li>
                <li><a href="#careers">Careers</a></li>
                <li><a href="#contact">Contact</a></li>
              </ul>
            </div>
            
            <div className="footer-column">
              <h4>Resources</h4>
              <ul>
                <li><a href="#blog">Blog</a></li>
                <li><a href="#tutorials">Tutorials</a></li>
                <li><a href="#faq">FAQ</a></li>
              </ul>
            </div>
            
            <div className="footer-column">
              <h4>Legal</h4>
              <ul>
                <li><a href="#privacy">Privacy Policy</a></li>
                <li><a href="#terms">Terms of Service</a></li>
                <li><a href="#security">Security</a></li>
              </ul>
            </div>
            
            <div className="footer-column">
              <h4>Connect</h4>
              <div className="social-icons">
                <a href="#twitter" className="social-icon">
                  <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="currentColor">
                    <path d="M23.953 4.57a10 10 0 01-2.825.775 4.958 4.958 0 002.163-2.723c-.951.555-2.005.959-3.127 1.184a4.92 4.92 0 00-8.384 4.482C7.69 8.095 4.067 6.13 1.64 3.162a4.822 4.822 0 00-.666 2.475c0 1.71.87 3.213 2.188 4.096a4.904 4.904 0 01-2.228-.616v.06a4.923 4.923 0 003.946 4.827 4.996 4.996 0 01-2.212.085 4.936 4.936 0 004.604 3.417 9.867 9.867 0 01-6.102 2.105c-.39 0-.779-.023-1.17-.067a13.995 13.995 0 007.557 2.209c9.053 0 13.998-7.496 13.998-13.985 0-.21 0-.42-.015-.63A9.935 9.935 0 0024 4.59z" />
                  </svg>
                </a>
                <a href="#linkedin" className="social-icon">
                  <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="currentColor">
                    <path d="M20.447 20.452h-3.554v-5.569c0-1.328-.027-3.037-1.852-3.037-1.853 0-2.136 1.445-2.136 2.939v5.667H9.351V9h3.414v1.561h.046c.477-.9 1.637-1.85 3.37-1.85 3.601 0 4.267 2.37 4.267 5.455v6.286zM5.337 7.433a2.062 2.062 0 01-2.063-2.065 2.064 2.064 0 112.063 2.065zm1.782 13.019H3.555V9h3.564v11.452zM22.225 0H1.771C.792 0 0 .774 0 1.729v20.542C0 23.227.792 24 1.771 24h20.451C23.2 24 24 23.227 24 22.271V1.729C24 .774 23.2 0 22.222 0h.003z" />
                  </svg>
                </a>
                <a href="#facebook" className="social-icon">
                  <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="currentColor">
                    <path d="M24 12.073c0-6.627-5.373-12-12-12s-12 5.373-12 12c0 5.99 4.388 10.954 10.125 11.854v-8.385H7.078v-3.47h3.047V9.43c0-3.007 1.792-4.669 4.533-4.669 1.312 0 2.686.235 2.686.235v2.953H15.83c-1.491 0-1.956.925-1.956 1.874v2.25h3.328l-.532 3.47h-2.796v8.385C19.612 23.027 24 18.062 24 12.073z" />
                  </svg>
                </a>
              </div>
            </div>
          </div>
        </div>
        
        <div className="copyright">
          <p>&copy; {new Date().getFullYear()} ResuMatch. All rights reserved.</p>
        </div>
      </footer>
    </div>
  );
}

export default LandingPage; 