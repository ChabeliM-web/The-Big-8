<?php
include("./db_connection.php");

$message = "";

if ($_SERVER['REQUEST_METHOD'] == 'POST') {
    // Define paths for uploaded files
    $q_paper_path = 'uploads/' . basename($_FILES['q_paper']['name']);
    $a_sheet_path = 'uploads/' . basename($_FILES['a_sheet']['name']);

    // Move uploaded files to the server
    if (move_uploaded_file($_FILES['q_paper']['tmp_name'], $q_paper_path) && move_uploaded_file($_FILES['a_sheet']['tmp_name'], $a_sheet_path)) {
        // Prepare command with full paths and correct escape
        $command = escapeshellcmd("python auto_mark.py " . escapeshellarg(realpath($q_paper_path)) . " " . escapeshellarg(realpath($a_sheet_path)));
        $output = shell_exec($command);

        // Debugging output
        if ($output === null) {
            echo "No output returned from Python script. Command executed: $command";
        } else {
            // Save feedback in a variable
            $feedback = htmlentities($output);

            // Display the result
            echo '<div class="feedback-container">';
            echo "<h3>Marking Feedback</h3>";
            echo "<div class='message success' id='message-box' style='display:none;'>$message</div>";
            echo "<pre id='feedback'>" . $feedback . "</pre>";

            // Buttons for downloading and saving to the database
            echo '<div class="action-buttons">';
            echo '<button id="downloadBtn" class="btn">Download as TXT</button>';
            echo '<button id="saveBtn" class="btn">Save to Database</button>'; // Changed to a button
            echo '</div>';
            // Link to mark another script
            echo '<div class="action-buttons">';
            echo '<a href="dashboard.html" class="btn">Mark Another Script</a>';
            echo '</div>';
            echo '</div>';
        }
    } else {
        echo "Failed to upload files.";
    }
} else {
    echo "Invalid request.";
}


if (isset($_POST['save_to_db'])) {
    $feedback = $_POST['feedback'];

    // Use regex patterns to extract the name, ID number, grade, and subject
    $name_pattern = "/Name:\s*(.+)/";
    $id_pattern = "/ID:\s*(.+)/";
    $grade_pattern = "/Grade:\s*(.+)/";
    $subject_pattern = "/Subject:\s*(.+)/";
    $mark_pattern = "/Total Score:\s*(\[[0-9]+\/[0-9]+\])/";  // Regex to extract the total mark

    // Extract Name
    if (preg_match($name_pattern, $feedback, $name_matches)) {
        $name = trim($name_matches[1]);
    } else {
        $name = "Unknown";
    }

    // Extract ID Number
    if (preg_match($id_pattern, $feedback, $id_matches)) {
        $id_number = trim($id_matches[1]);
    } else {
        $id_number = "Unknown";
    }

    // Extract Grade
    if (preg_match($grade_pattern, $feedback, $grade_matches)) {
        $grade = trim($grade_matches[1]);
    } else {
        $grade = "Unknown";
    }

    // Extract Subject
    if (preg_match($subject_pattern, $feedback, $subject_matches)) {
        $subject = trim($subject_matches[1]);
    } else {
        $subject = "Unknown";
    }

    // Extract Total Mark with brackets
    if (preg_match($mark_pattern, $feedback, $mark_matches)) {
        $mark_with_brackets = trim($mark_matches[1]);
    } else {
        $mark_with_brackets = "[0/0]";  // Default if not found
    }

    // Check if the student with the same ID number already exists
    $sql_check = "SELECT * FROM gradebook WHERE id_number = ?";
    $stmt = $conn->prepare($sql_check);
    $stmt->bind_param("s", $id_number);
    $stmt->execute();
    $result = $stmt->get_result();

    if ($result->num_rows > 0) {
        // Student exists, perform an UPDATE
        $sql_update = "UPDATE gradebook SET Name=?, Subject=?, Grade=?, Mark=? WHERE id_number=?";
        $stmt_update = $conn->prepare($sql_update);
        // Use "sssss" because all values are strings, including Mark
        $stmt_update->bind_param("sssss", $name, $subject, $grade, $mark_with_brackets, $id_number);

        if ($stmt_update->execute()) {
            $message = "Record updated successfully";
        } else {
            $message = "Error updating record: " . $conn->error;
        }

        $stmt_update->close();
    } else {
        // Student doesn't exist, perform an INSERT
        $sql_insert = "INSERT INTO gradebook (Name, Subject, id_number, Grade, Mark) VALUES (?, ?, ?, ?, ?)";
        $stmt_insert = $conn->prepare($sql_insert);
        // Use "sssss" because all values are strings
        $stmt_insert->bind_param("sssss", $name, $subject, $id_number, $grade, $mark_with_brackets);

        if ($stmt_insert->execute()) {
            $message = "Record inserted successfully";
        } else {
            $message = "Error inserting record: " . $conn->error;
        }

        $stmt_insert->close();
    }

    $stmt->close();
    $conn->close();
}

?>

<!DOCTYPE html>
<html lang="en">

<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Marking Feedback</title>

    <style>
        body {
            font-family: 'Poppins', sans-serif;
            background-color: #f4f4f9;
            color: #333;
            margin: 0;
            padding: 0;
        }

        .feedback-container {
            width: 80%;
            margin: 50px auto;
            padding: 20px;
            background-color: white;
            border-radius: 10px;
            box-shadow: 0px 0px 15px rgba(0, 0, 0, 0.1);
            text-align: center;
        }

        h3 {
            color: #333;
            font-size: 24px;
        }

        pre {
            background-color: #f1f1f1;
            padding: 20px;
            border: 2px solid #ddd;
            border-radius: 5px;
            color: #333;
            white-space: pre-wrap;
            text-align: left;
            max-height: 400px;
            overflow-y: scroll;
        }

        .action-buttons {
            margin-top: 20px;
        }

        .btn {
            padding: 10px 20px;
            background-color: #4CAF50;
            color: white;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            font-size: 16px;
            transition: background-color 0.3s;
            margin-right: 10px;
        }

        .btn:hover {
            background-color: #45a049;
        }

        .message {
            padding: 10px;
            margin: 10px 0;
            border: 1px solid transparent;
            border-radius: 5px;
            color: #fff;
            font-size: 16px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            position: relative;
        }

        .message.success {
            background-color: #4caf50;
        }

        .message.error {
            background-color: #f44336;
        }

        .message .close-btn {
            background: transparent;
            border: none;
            color: white;
            font-weight: bold;
            cursor: pointer;
            font-size: 18px;
            margin-left: 10px;
        }

        .message .close-btn:hover {
            color: #ddd;
        }
    </style>
</head>

<body>
    <script>
        document.getElementById('downloadBtn').addEventListener('click', function() {
            const feedbackText = document.getElementById('feedback').textContent;
            const blob = new Blob([feedbackText], {
                type: 'text/plain'
            });
            const link = document.createElement('a');
            link.href = window.URL.createObjectURL(blob);
            link.download = 'marking_feedback.txt';
            link.click();
        });

        document.addEventListener("DOMContentLoaded", function() {
            var messageBox = document.getElementById("message-box");

            // Automatically close after 5 seconds
            if (messageBox) {
                setTimeout(function() {
                    messageBox.style.display = 'none';
                }, 5000); // 5 seconds in milliseconds
            }
        });

        // Save to Database Button with AJAX
        document.getElementById('saveBtn').addEventListener('click', function() {
            var feedback = document.getElementById('feedback').textContent;

            var xhr = new XMLHttpRequest();
            xhr.open("POST", "save_feedback.php", true); // Use a separate file for saving
            xhr.setRequestHeader("Content-Type", "application/x-www-form-urlencoded");

            xhr.onreadystatechange = function() {
                if (xhr.readyState == 4 && xhr.status == 200) {
                    // Alert the user on success
                    alert("Feedback saved successfully");
                }
            };

            // Send the feedback as a POST request
            xhr.send("feedback=" + encodeURIComponent(feedback));
        });
    </script>
</body>

</html>