import React from 'react'
import { Button } from './Button'

interface Props {
  onClick: React.MouseEventHandler<HTMLButtonElement>;
}

export const Startscreen = ({onClick}: Props) => {
  return (
    <div className="container-fluid justify-content-center row-thing">
        <span>Press the button to start</span>
        <Button onClick={onClick}>
            Start drawing!
        </Button>
    </div>
  )
}
