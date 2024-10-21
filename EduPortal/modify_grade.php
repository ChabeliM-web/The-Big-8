<?php
include("./db_connection.php");

if (isset($_POST['submit'])) {
    $id_number = $_POST['id_number'];
    $subject = $_POST['subject'];
    $mark = $_POST['mark'];

    // Update the grade in the database
    $query = "UPDATE gradebook SET subject = '$subject', mark = '$mark' WHERE id_number = '$id_number'";

    if (mysqli_query($conn, $query)) {
        echo "Record updated successfully";
        header("Location: grades.php"); // Redirect back to the main page after modification
    } else {
        echo "Error updating record: " . mysqli_error($conn);
    }
}
?>
