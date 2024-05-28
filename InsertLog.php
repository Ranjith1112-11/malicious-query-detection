<?php 
$dbhost = "localhost";
$dbuser = "id17784130_root";
$dbpass = "rootuser--123S";
$db = "id17784130_queries";
$conn = new mysqli($dbhost, $dbuser, $dbpass,$db) or die("Connect failed: %s\n". $conn -> error);
if($conn && isset($_GET['qry']) && isset($_GET['type']))
{
    echo "Success";
    $sql = "INSERT INTO Query VALUES ('".$_GET['qry']."','".$_GET['type']."')";
if ($conn->query($sql) === TRUE) {
  echo "\nNew record created successfully";
} 
else {
  echo "Error: " . $sql . "<br>" . $conn->error;
    }
}
else{
    echo "Failure";
}
?>