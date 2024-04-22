// Function to display recommended jobs
function displayRecommendedJobs(jobs) {
    const jobList = document.getElementById("job-list");
    jobList.innerHTML = ""; // Clear previous job list
    
    jobs.forEach(job => {
        const listItem = document.createElement("li");
        listItem.textContent = `${job.job_title} - Similarity Score: ${job.similarity_score}`;
        jobList.appendChild(listItem);
    });
}

// Event listener for resume upload form submission
document.getElementById("resume-upload-form").addEventListener("submit", async function(event) {
    event.preventDefault(); // Prevent default form submission
    
    const formData = new FormData(this);
    
    try {
        // Send resume data to backend for processing
        const response = await fetch("/upload-resume", {
            method: "POST",
            body: formData
        });
        
        if (!response.ok) {
            throw new Error("Failed to upload resume");
        }
        
        const recommendedJobs = await response.json(); // Get recommended jobs from response
        displayRecommendedJobs(recommendedJobs); // Display recommended jobs on the page
    } catch (error) {
        console.error("Error:", error);
    }
});
