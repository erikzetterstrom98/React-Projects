import { Button } from "./components/Button";
import { Alert } from "./components/Alert";
import { useState } from "react";

const alertStrong = 'Satans jävla skit!';
const alertText = "Klicka bort den här skit'n.";
//const buttonChild = <Button onClick={handleClick}>Aej vafan...</Button>;

function App() {
  const [alertVisible, setAlertVisbility] = useState(false);
  const handleClick = () => {
    setAlertVisbility(true);
  }
  const handleClose = () => {
    setAlertVisbility(false);
  }
  return (
  <div>
    { alertVisible && <Alert strong={alertStrong} text={alertText} onClick={handleClose}/>}
    <Button onClick={handleClick}>Aej men vafan...</Button>
  </div>)
}

export default App;