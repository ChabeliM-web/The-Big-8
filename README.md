<!DOCTYPE html>
<html lang="en">

<body>

<h1>EduPortal</h1>

<p><strong>EduPortal</strong> is an AI-powered platform designed to help high school and primary school teachers automate the marking of learner scripts, reducing effort and time spent on grading. The platform provides an intuitive interface for uploading question papers and answer sheets, automatically processes the scripts, and provides detailed feedback.</p>

<h2>Features</h2>
<ul>
    <li><strong>AI-Powered Auto-Marking:</strong> Upload scripts and answer sheets, and the system will automatically evaluate and mark them.</li>
    <li><strong>Dashboard:</strong> Access a user-friendly dashboard to manage scripts, grades, and feedback.</li>
    <li><strong>Gradebook:</strong> Store and manage grades efficiently, with the option to modify grades when necessary.</li>
    <li><strong>User Authentication:</strong> Teachers can securely log in, manage their accounts, and log out when finished.</li>
    <li><strong>Downloadable Feedback:</strong> Teachers can download detailed feedback and results for each learner.</li>
    <li><strong>Multiple Styles:</strong> Supports a range of custom CSS styles to ensure a user-friendly and visually appealing interface.</li>
</ul>

<h2>Installation</h2>
<ol>
    <li>Clone the repository:
        <pre><code>git clone https://github.com/ChabeliM-web/The-Big-8/EduPortalCBS.git</code></pre>
    </li>
    <li>Navigate to the project directory:
        <pre><code>cd EduPortalCBS</code></pre>
    </li>
    <li>Set up your database by running the following SQL files in your MySQL database:
        <ul>
            <li><code>gradebook.sql</code></li>
            <li><code>logindetails.sql</code></li>
        </ul>
    </li>
    <li>Configure your database connection in <code>db_connection.php</code>.</li>
    <li>Ensure you have Python, Py2PDF, Selenium + Edge WebDriver & Groq installed for running the AI marking scripts:
        <ul>
            <li><code>auto_mark.py</code></li>
        </ul>
    </li>
    <li>Start your local server and ensure all dependencies are met.</li>
</ol>

<h2>Files and Directories</h2>
<ul>
    <li><code>auto_mark.php</code>: Handles the backend for auto-marking scripts using the AI model.</li>
    <li><code>auto_mark.py</code>: Python script that processes uploaded answer sheets and performs auto-marking using AI.</li>
    <li><code>db_connection.php</code>: Database connection configuration.</li>
    <li><code>dashboard.html</code>: Main dashboard for teachers to manage scripts, grades, and feedback.</li>
    <li><code>index.html</code>: Landing page for the EduPortalCBS.</li>
    <li><code>Login Form.html</code>: Login form for teachers to access the system.</li>
    <li><code>grades.php</code>: Page to display and manage student grades.</li>
    <li><code>papers.php</code>: Page to upload question papers and answer sheets.</li>
    <li><code>logout.php</code>: Handles logging out from the system.</li>
    <li><code>save_feedback.php</code>: Saves the feedback generated by the auto-marking process.</li>
    <li><code>style.css</code>, <code>Style2.css</code>, <code>Style3.css</code>, <code>Style4.css</code>: CSS files for different styles applied throughout the platform.</li>
</ul>

<h2>Database</h2>
<p>The project uses a MySQL database to manage users and store grades. Import the provided SQL files (<code>gradebook.sql</code> and <code>logindetails.sql</code>) to set up your database schema.</p>

<h2>Usage</h2>
<ul>
    <li><strong>Login:</strong> Teachers can log in using their credentials.</li>
    <li><strong>Upload Scripts:</strong> Navigate to the "Upload" section to upload question papers and answer sheets.</li>
    <li><strong>Auto-Mark:</strong> The AI will process the scripts and provide immediate grading.</li>
    <li><strong>View Grades:</strong> Teachers can view and modify grades as necessary.</li>
    <li><strong>Download Feedback:</strong> After auto-marking, teachers can download the detailed feedback for each learner.</li>
</ul>

<h2>License</h2>
<p>This project is licensed under the MIT License.</p>

</body>
</html>
