import React, { useState, useEffect } from 'react'
import { Button } from './components/Button';
import { Navbar } from './components/Navbar';
import { PredictionThing } from './components/PredictionThing';
import { Startscreen } from './components/Startscreen';

export default function App() {

  const [inStartMenu, setContent] = useState(true);

  function handleStart () {
    setContent(!inStartMenu);
  }

  return (
    <main>
      <Navbar goBackToMainMenu={handleStart} inMainMenu={inStartMenu}></Navbar>
      {inStartMenu ?
      <Startscreen onClick={handleStart}></Startscreen> :
      <PredictionThing></PredictionThing>}
    </main>
  )
}
