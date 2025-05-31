# ResuMatch

ResuMatch is a modern, AI-powered web application designed to provide comprehensive resume analysis and job matching. Leveraging advanced natural language processing and machine learning techniques, it helps job seekers optimize their resumes and assists HR organizations in identifying the best candidates.

## ✨ Features

### For Job Seekers:
- **ATS Score Analysis**: Get detailed feedback on how your resume performs against Applicant Tracking Systems
- **Job Match Analysis**: See how well your resume matches specific job descriptions
- **Detailed Insights**: Receive section-by-section scores and actionable recommendations

### For HR/Organizations:
- **Bulk Resume Processing**: Upload and analyze multiple resumes at once
- **Candidate Ranking**: Automatically rank candidates based on job description match
- **Top Skills Identification**: Identify key matched and missing skills for each candidate

## 🛠️ Technology Stack

### Backend:
- **FastAPI**: Modern, high-performance web framework
- **Machine Learning**: BERT embeddings, TF-IDF vectorization
- **Custom Resume Parser**: Advanced section identification and content extraction
- **Job Matching Algorithm**: Semantic and skill-based matching

### Frontend:
- **React**: Component-based UI with React Router for navigation
- **Animation Libraries**: AOS and Animate.css for smooth transitions
- **Responsive Design**: Mobile-first approach with modern CSS
- **Axios**: Promise-based HTTP client for API requests

## 🔧 Setup and Installation

### Backend Setup

```bash
# Navigate to the backend directory
cd ResuMatch/backend

# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Start the FastAPI server
uvicorn main:app --reload
```

### Frontend Setup

```bash
# Navigate to the frontend directory
cd ResuMatch/frontend

# Install dependencies
npm install

# Start the development server
npm start
```


### Job Seeker Mode:
1. Visit the landing page and click "Get Started" or navigate to `/app`
2. Choose "Job Seeker Mode"
3. Upload your resume (PDF or DOCX)
4. Toggle "Enable Job Matching" to analyze for a specific job
5. Enter job details (title, company, description)
6. Click "Analyze Resume"
7. View your detailed analysis results

### HR/Organization Mode:
1. Choose "HR/Organization Mode" from the mode selector
2. Upload multiple resumes (PDF or DOCX)
3. Enter job details for the position
4. Click "Find Top Candidates"
5. View the ranked list of candidates with their match scores and top skills


