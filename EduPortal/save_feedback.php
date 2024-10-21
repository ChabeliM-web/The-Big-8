<?php
include("./db_connection.php");

if ($_SERVER['REQUEST_METHOD'] == 'POST' && isset($_POST['feedback'])) {
    $feedback = $_POST['feedback'];

    // Use regex patterns to extract the name, ID number, grade, and subject
    $name_pattern = "/Name:\s*(.+)/";
    $id_pattern = "/ID:\s*(.+)/";
    $grade_pattern = "/Grade:\s*(.+)/";
    $subject_pattern = "/Subject:\s*(.+)/";
    $mark_pattern = "/Total Score:\s*(\[[0-9]+\/[0-9]+\])/";  // Regex to extract the total mark

    // Initialize variables
    $name = "Unknown";
    $id_number = "Unknown";
    $grade = "Unknown";
    $subject = "Unknown";
    $mark_with_brackets = "[0/0]";  // Default if not found

    // Function to clean inputs (remove asterisks and unwanted characters)
    function clean_input($input) {
        // Remove asterisks and trim whitespace
        return trim(str_replace("*", "", $input));
    }

    // Extract and clean Name
    if (preg_match($name_pattern, $feedback, $name_matches)) {
        $name = clean_input($name_matches[1]);
    }

    // Extract and clean ID Number
    if (preg_match($id_pattern, $feedback, $id_matches)) {
        $id_number = clean_input($id_matches[1]);
    }

    // Extract and clean Grade
    if (preg_match($grade_pattern, $feedback, $grade_matches)) {
        $grade = clean_input($grade_matches[1]);
    }

    // Extract and clean Subject
    if (preg_match($subject_pattern, $feedback, $subject_matches)) {
        $subject = clean_input($subject_matches[1]);
    }

    // Extract and clean Total Mark with brackets
    if (preg_match($mark_pattern, $feedback, $mark_matches)) {
        $mark_with_brackets = clean_input($mark_matches[1]);
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
        $stmt_update->bind_param("sssss", $name, $subject, $grade, $mark_with_brackets, $id_number);

        if ($stmt_update->execute()) {
            echo "Record updated successfully";
        } else {
            echo "Error updating record: " . $conn->error;
        }

        $stmt_update->close();
    } else {
        // Student doesn't exist, perform an INSERT
        $sql_insert = "INSERT INTO gradebook (Name, Subject, id_number, Grade, Mark) VALUES (?, ?, ?, ?, ?)";
        $stmt_insert = $conn->prepare($sql_insert);
        $stmt_insert->bind_param("sssss", $name, $subject, $id_number, $grade, $mark_with_brackets);

        if ($stmt_insert->execute()) {
            echo "Record inserted successfully";
        } else {
            echo "Error inserting record: " . $conn->error;
        }

        $stmt_insert->close();
    }

    $stmt->close();
} else {
    echo "Invalid request.";
}

$conn->close();
?>
