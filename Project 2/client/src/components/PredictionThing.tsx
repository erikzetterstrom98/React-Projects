import {useEffect, useRef, useState} from 'react'
import { Button } from './Button'
import { Canvas } from './Canvas'

const CANVAS_HEIGHT = 200;
const CANVAS_WIDTH = 200;

const BACKEND_URL = 'http://localhost:5000/predict'

export const PredictionThing = () => {

  const contextReference = useRef<CanvasRenderingContext2D | null>(null);

  const [guess, setGuess] = useState(-1);

  // useEffect(
  //   () => {
  //     console.log(contextReference.current?.getImageData(0, 0, 200, 200));
  //   },
  //   [contextReference]
  // )

  function getCanvasData(): number[] | undefined {
    if (!contextReference.current) {
      return;
    }
    const imageData = contextReference.current.getImageData(0, 0, 200, 200);
    const data = imageData.data;

    let array: number[] = [];

    for (let index = 0; index < CANVAS_HEIGHT*CANVAS_WIDTH*4; index += 4){
      const r = data[index+0];
      const g = data[index+1];
      const b = data[index+2];
      const a = data[index+3];

      if (r == 0 && g == 0 && b == 0){
        array.push(a/255); // Only push black pixels
      }
      else {
        array.push(0);
      }
    }
    return array;
  }

  function callBackend() {
    setGuess(-1);
    const imageData = getCanvasData();
    //const imageData = [6, 2, 0, 1, 1, 4];
    if (!imageData) {
      return;
    }

    var requestHeaders = new Headers();
    requestHeaders.append("Content-Type", "application/json");

    var bodyData = JSON.stringify(
      {
        "image_height": CANVAS_HEIGHT,
        "image_width": CANVAS_WIDTH,
        "data": imageData
      }
    );

    var requestOptions: RequestInit = {
      method: 'POST',
      headers: requestHeaders,
      body: bodyData,
      redirect: 'follow'
    }

    fetch(
      BACKEND_URL,
      requestOptions
    )
      .then(
        response => response.json()
      )
      .then(
        result => setGuess(result["guess"])
      )
      .catch(
        error => console.log('Error:', error)
      )
  }

  return (
    <>
    <div className='container text-center'>
      <Canvas 
        contextReference={contextReference}
        canvas_height={CANVAS_HEIGHT}
        canvas_width={CANVAS_WIDTH}
      >
      </Canvas>
    </div>
    <div className='container text-center'><Button onClick={callBackend}>
    Guess!
  </Button></div>

  <div className='container text-center guess'>
  {guess !== -1 && 'My guess is:' + guess}</div>
  </>
  )
}
