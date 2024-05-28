<?php 
$dbhost = "localhost";
$dbuser = "id17784130_root";
$dbpass = "rootuser--123S";
$db = "id17784130_queries";
$conn = new mysqli($dbhost, $dbuser, $dbpass,$db) or die("Connect failed: %s\n". $conn -> error);

$sql = "SELECT query FROM temp";
$result = mysqli_query($conn, $sql);

if (mysqli_num_rows($result) > 0) {
  while($row = mysqli_fetch_assoc($result)) {
      $myVar = htmlspecialchars( $row["query"], ENT_QUOTES);
      echo("<h1 id='Query'>".$myVar."</h1>");
  }
} else {
 echo("<h1 id='Query'>0</h1>");
}

mysqli_close($conn);