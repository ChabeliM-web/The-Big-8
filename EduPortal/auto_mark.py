import os
import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import sys
from PyPDF2 import PdfReader
from groq import Groq

# Set the correct path to the Edge Selenium WebDriver
edge_driver_path = r"C:/Selenium_Driver/msedgedriver.exe"
os.environ['PATH'] += os.pathsep + edge_driver_path

# Create a WebDriver instance for Microsoft Edge
options = webdriver.EdgeOptions()
options.add_experimental_option("detach", True)  # Keep the browser open

driver = webdriver.Edge(options=options)
driver.set_window_size(1024, 768)

# Function to extract text from a PDF file
def extract_text_from_pdf(pdf_file):
    
    reader = PdfReader(pdf_file)
    extracted_text = ""
    for page in reader.pages:
        extracted_text += page.extract_text()
    
    return extracted_text

# Function to upload an image file and submit it on the website
def upload_image_to_website(image_file):
    
    driver.get("https://www.pen-to-print.com/App/notes/")
    wait = WebDriverWait(driver, 30)
    file_input = wait.until(EC.presence_of_element_located((By.XPATH, "//input[@type='file']")))
    file_input.send_keys(image_file)
    
    submit_button = wait.until(EC.element_to_be_clickable((By.CLASS_NAME, "convert-button")))
    submit_button.click()

    time.sleep(10)
    textarea = wait.until(EC.presence_of_element_located((By.CLASS_NAME, "scanline-cell-content")))
    extracted_text = textarea.get_attribute("value")
    
     
    return extracted_text

# Function to enter extracted questions and answers into Groq AI for evaluation
def submit_questions_and_answers_to_groq(q_paper_text, a_sheet_text):


    # Set the API key environment variable for Groq
    os.environ['GROQ_API_KEY'] = "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"

    # Initialize the Groq client
    client = Groq(
        api_key=os.environ.get("GROQ_API_KEY"),
    )

   # Define the submission content
    submit_content = (
        f"Please mark the provided exam paper and answers.\n\n"
        f"Questions:\n{q_paper_text}\n\n"
        f"Answers:\n{a_sheet_text}\n\n"
        "Mark each sub-question according to the marks provided in parentheses, for example: (3). "
        "The total score for each section is given in square brackets, for example: [30]. "
        "Please calculate the marks accordingly and ensure the grand total is consistent.\n\n"
        "Before giving feedback, always read back the following information in this exact format, separated by a new line. Do not add any astericks or stars around results, keep it clean:\n"
        "- Name: Learner Name\n"
        "- ID: Learner ID\n"
        "- Grade: Learner Grade\n"
        "- Subject: Subject Name\n"
        "- Total Score: [Attained Test Mark In Square Brackets/The total for the test]\n\n"
        "At the end of the marking, provide an overall feedback on the learner's performance, but do not provide feedback per question.\n"
        "Focus on giving a clear and concise summary of the learner's overall performance."
    )

    # Pass the content to Groq AI and get the response
    response = client.chat.completions.create(
        messages=[
            {"role": "user", "content": submit_content}
        ],
        model="llama3-8b-8192",
        timeout=120  # Set a timeout of 120 seconds
    )

    # Extract the Groq AI's response and format it
    marked_results = response.choices[0].message.content.strip()

    return marked_results

# Function to preprocess the image and extract text
def extract_text_from_image(image_file):
    extracted_text = upload_image_to_website(image_file)
    
    return extracted_text

# Main logic to handle file uploads, text extraction, and marking
def process_and_mark(q_paper_path, a_sheet_path):

    q_paper_text = extract_text_from_pdf(q_paper_path)

    if a_sheet_path.endswith('.pdf'):
        a_sheet_text = extract_text_from_pdf(a_sheet_path)
    else:
        a_sheet_text = extract_text_from_image(a_sheet_path)

    marking_result = submit_questions_and_answers_to_groq(q_paper_text, a_sheet_text)

    # Here, you can format the result to display the questions, marks per question, and total marks in a neat way
    formatted_result = format_results(marking_result)
    return formatted_result

# Function to format results neatly with marks
def format_results(results):
    formatted_text = ""
    questions_and_marks = results.split('\n')

    for line in questions_and_marks:
        if line.strip():  # Ensure there's content in the line
            formatted_text += line + "\n"

    return formatted_text

if __name__ == '__main__':

    q_paper_path = sys.argv[1]
    a_sheet_path = sys.argv[2]

    # Printing the extracted information
    print(f"Question Paper Path: {q_paper_path}")
    print(f"Answer Sheet Path: {a_sheet_path}")

    result = process_and_mark(q_paper_path, a_sheet_path)
    print(f" ")
    print(result)



