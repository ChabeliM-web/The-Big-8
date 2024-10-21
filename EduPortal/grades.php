<?php
include("./db_connection.php");

// Fetch data from gradebook table
$query = "SELECT id_number, name, subject, grade, mark FROM gradebook";
$result = mysqli_query($conn, $query);
?>

<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Student Grades Table</title>
    <link rel="stylesheet" href="Style4.css">
</head>
<body>

    <h2>Student Grades</h2>
    <table class="grades-table">
        <thead>
            <tr>
                <th>Student ID</th>
                <th>Name</th>
                <th>Subject</th>
                <th>Grade</th>
                <th>Attained Mark</th>
                <th>Modify Marks</th>
            </tr>
        </thead>
        <tbody>
            <?php
            // Loop through the gradebook data and populate the table
            while ($row = mysqli_fetch_assoc($result)) {
                echo "<tr>";
                echo "<form action='modify_grade.php' method='post'>";
                echo "<td>" . $row['id_number'] . "</td>";
                echo "<td>" . $row['name'] . "</td>";
                echo "<td><select id='subject' name='subject' required>";
                echo "<option value='" . $row['subject'] . "' selected>" . $row['subject'] . "</option>";
                echo "<option value='Math'>Math</option>";
                echo "<option value='English'>English</option>";
                echo "<option value='Science'>Science</option>";
                echo "<option value='History'>History</option>";
                echo "<option value='Geography'>Geography</option>";
                echo "<option value='Art'>Art</option>";
                echo "<option value='Computer Science'>Computer Science</option>";
                echo "</select></td>";
                echo "<td>" . $row['grade'] . "</td>";
                echo "<td><input type='text' name='mark' value='" . $row['mark'] . "'></td>";
                echo "<td><button type='submit' name='submit'>Modify</button></td>";
                echo "<input type='hidden' name='id_number' value='" . $row['id_number'] . "'>";
                echo "</form>";
                echo "</tr>";
            }
            ?>
        </tbody>
    </table>

</body>
</html>
