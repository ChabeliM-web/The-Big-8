<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Download Question Papers</title>
    <link rel="stylesheet" href="index.css">
    <style>
        body {
            font-family: 'Poppins', sans-serif;
            background-color: #f4f4f9;
            color: #333;
            margin: 0;
            padding: 0;
        }

        .container {
            width: 80%;
            max-width: 800px;
            margin: 50px auto;
            padding: 20px;
            background-color: #ffffff;
            border-radius: 10px;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
        }

        h1 {
            text-align: center;
            color: #4caf50;
            margin-bottom: 20px;
        }

        table {
            width: 100%;
            border-collapse: collapse;
            margin-bottom: 20px;
        }

        thead {
            background-color: #4caf50;
            color: white;
        }

        th, td {
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }

        tr:hover {
            background-color: #f1f1f1;
        }

        a {
            text-decoration: none;
            color: #4caf50;
            font-weight: bold;
            transition: color 0.3s;
        }

        a:hover {
            color: #45a049;
        }

        /* Responsive design */
        @media (max-width: 600px) {
            .container {
                width: 95%;
            }

            th, td {
                font-size: 14px;
                padding: 10px;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Download Question Papers</h1>
        <table>
            <thead>
                <tr>
                    <th>Subject</th>
                    <th>Action</th>
                    <th>Year</th>
                </tr>
            </thead>
            <tbody>
                <?php
                // Directory containing the PDFs
                $directory = 'uploads/';
                
                // Open the directory
                if ($handle = opendir($directory)) {
                    // Loop through the files in the directory
                    while (false !== ($file = readdir($handle))) {
                        // Only process PDF files
                        if (pathinfo($file, PATHINFO_EXTENSION) === 'pdf') {
                            // Extract the subject and year from the filename
                            $parts = explode('_', pathinfo($file, PATHINFO_FILENAME));
                            $subject = ucfirst($parts[0]); // Capitalize first letter
                            $year = end($parts); // Last part as the year

                            echo '<tr>';
                            echo '<td>' . htmlspecialchars($subject) . '</td>';
                            echo '<td><a href="download.php?file=' . urlencode($file) . '">Download</a></td>';
                            echo '<td>' . htmlspecialchars($year) . '</td>';
                            echo '</tr>';
                        }
                    }
                    closedir($handle);
                } else {
                    echo '<tr><td colspan="3">Unable to open the uploads directory.</td></tr>';
                }
                ?>
            </tbody>
        </table>
        <a href="dashboard.html">Back To Dashboard</a>
    </div>
</body>
</html>

