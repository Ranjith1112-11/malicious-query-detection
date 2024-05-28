<?php 
$dbhost = "localhost";
$dbuser = "id17784130_root";
$dbpass = "rootuser--123S";
$db = "id17784130_queries";
if(isset($_GET['query']))
{
    $conn = new mysqli($dbhost, $dbuser, $dbpass,$db) or die("Connect failed: %s\n". $conn -> error);
 if($conn)   {
    echo "Success";
    echo $_GET['query'];
    $sql = "INSERT INTO temp
VALUES ('".$_GET['query']."')";
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
}
?>
<html>
    <head></head>
    <body>
        <form action="Home.php" method="POST">
            <label for="fname">Enter Query:</label>
<input type="text" id="query" name="fname">
<a href = "javascript:;" onclick = "this.href='Home.php?query=' + document.getElementById('query').value">Click</a>
        </form>
        
    </body>
</html>