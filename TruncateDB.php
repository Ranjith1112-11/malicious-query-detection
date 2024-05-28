<?php 
$dbhost = "localhost";
$dbuser = "id17784130_root";
$dbpass = "rootuser--123S";
$db = "id17784130_queries";
$conn = new mysqli($dbhost, $dbuser, $dbpass,$db) or die("Connect failed: %s\n". $conn -> error);
if($conn)
{
    
    $sql = "TRUNCATE TABLE temp";
if ($conn->query($sql) === TRUE) {
  echo "Truncated Successfully";
} 
else {
  echo "Error: " . $sql . "<br>" . $conn->error;
    }
}
else{
    echo "Failure";
}