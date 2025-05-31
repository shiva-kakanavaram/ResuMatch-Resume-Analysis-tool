# Images Directory

This directory contains images used throughout the ResuMatch application. These include:

## SVG Mockups

### User Interface Mockups
- `jobseeker_upload.svg`: A visual representation of the job seeker interface, showing the resume upload functionality with drag-and-drop capability and supported file formats.
- `hr_upload.svg`: A visual representation of the HR/Organization interface, displaying the job posting creation form with fields for job details and document upload.

### Feature Visualization
- `resume_analysis.svg`: A visualization of the resume analysis results screen, showing skills identification, experience matching, and personalized recommendations.
- `job_matching.svg`: A visualization of the job matching results interface, displaying job listings with match percentages and filtering options.
- `hr_dashboard.svg`: A visualization of the HR dashboard, showing analytics, job postings statistics, and recent applications with match scores.

## Screenshots
These screenshots can be used in documentation, presentations, or marketing materials:

- **resume_analysis.png**: Screenshot of the resume analysis result screen, showing ATS score, formatting suggestions, and content improvements.

- **job_matching.png**: Screenshot of the job matching results, showing skills match percentage, matched skills list, and missing skills that could improve the match.

- **hr_dashboard.png**: Screenshot of the HR dashboard, showing ranked candidates from bulk resume analysis and their respective match scores.

## Usage
To use these images in your React components:
```jsx
import { ReactComponent as JobseekerUpload } from '../images/jobseeker_upload.svg';
import hrUploadImg from '../images/hr_upload.svg';
import screenshotImg from '../images/resume_analysis.png';

// For SVG as React component (allows styling with CSS)
<JobseekerUpload className="custom-class" />

// For regular image tag
<img src={hrUploadImg} alt="HR Upload Interface" />
<img src={screenshotImg} alt="Resume Analysis Screenshot" />
```

## Contributing Images
When adding new images to this directory:
1. Use descriptive filenames that clearly indicate the content of the image
2. Optimize images for web (compress PNGs, use appropriate SVG optimization)
3. Update this README with a description of any new images
4. Ensure all images are properly licensed for use in this project 

For SVG files, follow these additional guidelines:
- Use consistent styling (colors, fonts, etc.) that match the application's design system
- Include appropriate comments in the SVG code to explain complex elements
- Test SVG rendering in multiple browsers to ensure compatibility 