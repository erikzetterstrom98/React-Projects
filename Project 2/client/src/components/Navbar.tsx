import React, { useState } from 'react'
import { Button } from './Button';

interface Wohoo {
  inMainMenu: boolean
  goBackToMainMenu: () => void;
}

export const Navbar = ({goBackToMainMenu, inMainMenu}: Wohoo) => {
  // const [backButton, setBackButton] = useState(false);

  // const handleBackClick = () => {
  //   setBackButton(false);
  //   goBackToMainMenu();
  // }

  return (
    <nav className="navbar navbar-expand-lg bg-body-tertiary">
      <div className="container navbar-container">
        <div>
          {
            !inMainMenu &&
            <Button onClick={goBackToMainMenu}>
              {'\u2190'}
            </Button>
          }
        </div>
        <div>
          <h1>Draw AI</h1>
        </div>
        <div> </div>
      </div>
    </nav>
  )
}
